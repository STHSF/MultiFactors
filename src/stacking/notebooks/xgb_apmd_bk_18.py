#!/usr/bin/env python
# coding: utf-8
import sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import gc, time, sqlite3
import pandas as pd
import numpy as np
from m1_xgb import *
from datetime import datetime, timedelta
from src.conf.configuration import regress_conf
from src.stacking import factor_store, feature_list
from PyFin.api import *
from alphamind.api import *
from src.conf.models import *
from alphamind.execution.naiveexecutor import NaiveExecutor

data_source = 'postgresql+psycopg2://alpha:alpha@180.166.26.82:8889/alpha'
engine = SqlEngine(data_source)

# 是否使用严格的组合优化条件
strict = False
universe = Universe('zz500')
freq = '5b'
benchmark_code = 905
start_date = '2010-01-01'  # 训练集的起始时间
back_start_date = '2010-01-01'  # 模型回测的起始时间
end_date = '2019-10-01'
ref_dates = makeSchedule(start_date, end_date, freq, 'china.sse')
back_ref_dates = makeSchedule(back_start_date, end_date, freq, 'china.sse')
horizon = map_freq(freq)
industry_name = 'sw'
industry_level = 1
alpha_logger.info('读取数据。。。。。。。。。')
# uqer因子列表
basic_factor_store = factor_store.basic_factor_store
# alpha191因子列表
alpha_factor_store = factor_store.alpha_factor_store

# 提取Uqer因子
alpha_logger.info('loading basic_factor_org ...........')
basic_factor_org = engine.fetch_factor_range(universe, basic_factor_store, dates=ref_dates)
alpha_logger.info('basic_factor_org loading success')
# 提取alpha191因子
alpha_logger.info('loading alpha191_factor_org ...........')
alpha191_factor_org = engine.fetch_factor_range(universe, alpha_factor_store, dates=ref_dates,
                                                used_factor_tables=[Alpha191])

# 合并所有的因子
factor_data_org = pd.merge(basic_factor_org, alpha191_factor_org, on=['trade_date', 'code'], how='outer')

# 获取
alpha_logger.info('loading industry_total data ...........')
industry = engine.fetch_industry_range(universe, dates=ref_dates)
factor_data = pd.merge(factor_data_org, industry, on=['trade_date', 'code']).fillna(0.)
risk_total = engine.fetch_risk_model_range(universe, dates=ref_dates)[1]

return_data = engine.fetch_dx_return_range(universe, dates=ref_dates, horizon=horizon, offset=0,
                                           benchmark=benchmark_code)

benchmark_total = engine.fetch_benchmark_range(dates=ref_dates, benchmark=benchmark_code)
industry_total = engine.fetch_industry_matrix_range(universe, dates=ref_dates, category=industry_name,
                                                    level=industry_level)
alpha_logger.info('industry_total loading success')

train_data = pd.merge(factor_data, return_data, on=['trade_date', 'code']).dropna()

# Constraintes settings
industry_names = industry_list(industry_name, industry_level)
constraint_risk = ['EARNYILD', 'LIQUIDTY', 'GROWTH', 'SIZE', 'SIZENL', 'BETA', 'MOMENTUM'] + industry_names
# constraint_risk = ['EARNYILD', 'LIQUIDTY', 'GROWTH', 'SIZE', 'BETA', 'MOMENTUM'] + industry_names

total_risk_names = constraint_risk + ['benchmark', 'total']
b_type = []
l_val = []
u_val = []

if strict:
    for name in total_risk_names:
        if name == 'benchmark':
            b_type.append(BoundaryType.RELATIVE)
            l_val.append(0.0)
            u_val.append(1.0)
        elif name == 'total':
            b_type.append(BoundaryType.ABSOLUTE)
            l_val.append(-0.0)
            u_val.append(0.0)
        elif name == 'SIZE':
            b_type.append(BoundaryType.ABSOLUTE)
            l_val.append(-0.1)
            u_val.append(0.1)
        elif name == 'SIZENL':
            b_type.append(BoundaryType.ABSOLUTE)
            l_val.append(-0.1)
            u_val.append(-0.1)
        elif name in industry_names:
            b_type.append(BoundaryType.ABSOLUTE)
            l_val.append(-0.005)
            u_val.append(0.005)
        else:
            b_type.append(BoundaryType.ABSOLUTE)
            l_val.append(-1.0)
            u_val.append(1.0)
else:
    for name in total_risk_names:
        if name == 'benchmark':
            b_type.append(BoundaryType.RELATIVE)
            l_val.append(0.0)
            u_val.append(1.0)
        elif name == 'total':
            b_type.append(BoundaryType.ABSOLUTE)
            l_val.append(.0)
            u_val.append(.0)
        else:
            b_type.append(BoundaryType.ABSOLUTE)
            l_val.append(-1.0)
            u_val.append(1.0)

bounds = create_box_bounds(total_risk_names, b_type, l_val, u_val)

# 获取特征名
features = list(basic_factor_store.keys())
alpha_features = list(alpha_factor_store.keys())
features.extend(alpha_features)
label = ['dx']


def create_scenario():
    weight_gap = 1
    transact_cost = 0.003
    GPU_device = True

    executor = NaiveExecutor()
    trade_dates = []
    current_pos = pd.DataFrame()
    total_pos = pd.DataFrame()
    previous_pos = pd.DataFrame()
    tune_record = pd.DataFrame()
    rets = []
    net_rets = []
    turn_overs = []
    leverags = []
    ics = []

    # take ref_dates[i] as an example
    for i, ref_date in enumerate(back_ref_dates):
        alpha_logger.info('{0} is start'.format(ref_date))
        # machine learning model
        # Filter Training data
        # train data
        trade_date_pre = ref_date - timedelta(days=1)
        trade_date_pre_80 = ref_date - timedelta(days=80)
        # train = train_data[(train_data.trade_date <= trade_date_pre) & (trade_date_pre_80 <= train_data.trade_date)].dropna()
        # 训练集构造, 选择当天之前(不含当天)的因子数据作为训练集.
        train = train_data[train_data.trade_date <= trade_date_pre].dropna()

        if len(train) <= 0:
            continue
        x_train = train[features]
        y_train = train[label]
        alpha_logger.info('len_x_train: {0}, len_y_train: {1}'.format(len(x_train.values), len(y_train.values)))
        alpha_logger.info('X_train.shape={0}, X_test.shape = {1}'.format(np.shape(x_train), np.shape(y_train)))

        # xgb_configuration
        regress_conf.xgb_config_r()
        regress_conf.cv_folds = None
        regress_conf.early_stop_round = 10
        regress_conf.max_round = 800
        tic = time.time()
        # training
        xgb_model = XGBooster(regress_conf)
        if GPU_device:
            xgb_model.set_params(tree_method='gpu_hist', max_depth=5)
        else:
            xgb_model.set_params(max_depth=5)
        alpha_logger.info(xgb_model.get_params())
        best_score, best_round, cv_rounds, best_model = xgb_model.fit(x_train, y_train)
        alpha_logger.info('Training time cost {}s'.format(time.time() - tic))
        alpha_logger.info('best_score = {}, best_round = {}'.format(best_score, best_round))

        # 测试集, 取当天的因子数据作为输入.
        total_data_test_excess = train_data[train_data.trade_date == ref_date]
        alpha_logger.info('{0} total_data_test_excess: {1}'.format(ref_date, len(total_data_test_excess)))
        if len(total_data_test_excess) <= 0:
            alpha_logger.info('{0} HAS NO DATA!!!'.format(ref_date))
            continue

        # 获取当天的行业, 风险模型和基准数据
        industry_matrix = industry_total[industry_total.trade_date == ref_date]
        benchmark_w = benchmark_total[benchmark_total.trade_date == ref_date]
        risk_matrix = risk_total[risk_total.trade_date == ref_date]

        total_data = pd.merge(industry_matrix, benchmark_w, on=['code'], how='left').fillna(0.)
        total_data = pd.merge(total_data, risk_matrix, on=['code'])
        alpha_logger.info('{0} len_of_total_data: {1}'.format(ref_date, len(total_data)))

        total_data_test_excess = pd.merge(total_data, total_data_test_excess, on=['code'])
        alpha_logger.info('{0} len_of_total_data_test_excess: {1}'.format(ref_date, len(total_data_test_excess)))

        codes = total_data_test_excess.code.values.tolist()
        alpha_logger.info('{0} full re-balance: {1}'.format(ref_date, len(codes)))
        # 获取调仓日当天的股票收益
        dx_returns = return_data[return_data.trade_date == ref_date][['code', 'dx']]

        benchmark_w = total_data_test_excess.weight.values
        alpha_logger.info('shape_of_benchmark_w: {}'.format(np.shape(benchmark_w)))
        is_in_benchmark = (benchmark_w > 0.).astype(float).reshape((-1, 1))
        total_risk_exp = np.concatenate([total_data_test_excess[constraint_risk].values.astype(float),
                                         is_in_benchmark,
                                         np.ones_like(is_in_benchmark)],
                                        axis=1)
        alpha_logger.info('shape_of_total_risk_exp_pre: {}'.format(np.shape(total_risk_exp)))
        total_risk_exp = pd.DataFrame(total_risk_exp, columns=total_risk_names)
        alpha_logger.info('shape_of_total_risk_exp: {}'.format(np.shape(total_risk_exp)))
        constraints = LinearConstraints(bounds, total_risk_exp, benchmark_w)
        alpha_logger.info('constraints: {0} in {1}'.format(np.shape(constraints.risk_targets()), ref_date))

        lbound = np.maximum(0., benchmark_w - weight_gap)
        ubound = weight_gap + benchmark_w
        alpha_logger.info('lbound: {0} in {1}'.format(np.shape(lbound), ref_date))
        alpha_logger.info('ubound: {0} in {1}'.format(np.shape(ubound), ref_date))

        # predict
        x_pred = total_data_test_excess[features]
        predict_xgboost = xgb_model.predict(best_model, x_pred)
        a = np.shape(predict_xgboost)
        predict_xgboost = np.reshape(predict_xgboost, (a[0], -1)).astype(np.float64)
        alpha_logger.info('shape_of_predict_xgboost: {}'.format(np.shape(predict_xgboost)))
        del xgb_model
        del best_model
        gc.collect()

        # 股票过滤, 组合优化之前过滤掉(未完成)
        
        # 保证当前持仓和优化前的持仓数据顺序一致
        previous_pos = total_data_test_excess[['code']].merge(previous_pos, on='code', how='left').fillna(0)
        # backtest
        try:
            target_pos, _ = er_portfolio_analysis(predict_xgboost,
                                                  total_data_test_excess['industry'].values,
                                                  None,
                                                  constraints,
                                                  False,
                                                  benchmark_w,
                                                  method='risk_neutral',
                                                  lbound=lbound,
                                                  ubound=ubound,
                                                  turn_over_target=0.5,
                                                  current_pos=previous_pos)
        except:
            target_pos = None
            alpha_logger.info('target_pos: error')
        alpha_logger.info('target_pos_shape: {}'.format(np.shape(target_pos)))
        alpha_logger.info('len_codes:{}'.format(np.shape(codes)))
        target_pos['code'] = codes

        result = pd.merge(target_pos, dx_returns, on=['code'])
        result['trade_date'] = ref_date
        tune_record = tune_record.append(result)
        alpha_logger.info('len_result: {}'.format(len(result)))

        # excess_return = np.exp(result.dx.values) - 1. - index_return.loc[ref_date, 'dx']
        excess_return = np.exp(result.dx.values) - 1.
        ret = result.weight.values @ excess_return

        trade_dates.append(ref_date)
        rets.append(np.log(1. + ret))
        alpha_logger.info('len_rets: {}, len_trade_dates: {}'.format(len(rets), len(trade_dates)))

        turn_over_org, current_pos = executor.execute(target_pos=target_pos)
        turn_over = turn_over_org / sum(target_pos.weight.values)
        alpha_logger.info('turn_over: {}'.format(turn_over))
        turn_overs.append(turn_over)
        alpha_logger.info('turn_over: {}'.format(turn_over))
        previous_pos = executor.set_current(current_pos)
        previous_pos['trade_date'] = ref_date
        total_pos.append(previous_pos, ignore_index=True)
        net_rets.append(np.log(1. + ret - transact_cost * turn_over))
        alpha_logger.info('len_net_rets: {}, len_trade_dates: {}'.format(len(net_rets), len(trade_dates)))
        alpha_logger.info('{} is finished'.format(ref_date))

    # ret_df = pd.DataFrame({'xgb_regress': rets}, index=trade_dates)
    ret_df = pd.DataFrame({'xgb_regress': rets, 'net_xgb_regress': net_rets}, index=trade_dates)
    ret_df.loc[advanceDateByCalendar('china.sse', ref_dates[-1], freq).strftime('%Y-%m-%d')] = 0.
    ret_df = ret_df.shift(1)
    ret_df.iloc[0] = 0.
    return ret_df, tune_record, rets, net_rets, total_pos


ret_df, tune_record, rets, net_rets, total_pos = create_scenario()

# 调仓记录保存
# con = sqlite3.connect('./tune_record.db')
# 约束条件比较严格
# table_name = 'tune_record_whole_strict'
# table_name = 'tune_record_strict'
# 约束条件比较宽松
# table_name = 'tune_record_whole_loose'
# table_name = 'tune_record_loose'
# alpha_logger.info('save tune_record to {}'.format(table_name))
# tune_record.to_sql(table_name, con=con, if_exists='replace', index=False)