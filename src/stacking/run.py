#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: run.py
@time: 2019-08-05 19:55
"""

import time
import sys
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../')
import pandas as pd
from PyFin.api import *
from alphamind.api import *
from src.conf.models import *
import numpy as np
from alphamind.execution.naiveexecutor import NaiveExecutor
from datetime import datetime, timedelta
from m1_xgb import *
from src.conf.configuration import regress_conf
import xgboost as xgb
import gc
import json
from flask import Flask

app = Flask(__name__)


def create_scenario(train_data, weight_gap, return_data, risk_total,
                    benchmark_total, industry_total, bounds, constraint_risk, total_risk_names):
    executor = NaiveExecutor()
    trade_dates = []
    transact_cost = 0.003
    previous_pos = pd.DataFrame()
    tune_record = pd.DataFrame()
    current_pos = pd.DataFrame()
    turn_overs = []
    leverags = []
    rets = []
    net_rets = []
    turn_overs = []
    leverags = []
    ics = []
    # take ref_dates[i] as an example
    for i, ref_date in enumerate(ref_dates):
        alpha_logger.info('{0} is start'.format(ref_date))

        # machine learning model
        # Filter Training data
        # train data
        trade_date_pre = ref_date - timedelta(days=1)
        trade_date_pre_80 = ref_date - timedelta(days=80)

        # train = train_data[(train_data.trade_date <= trade_date_pre) & (trade_date_pre_80 <= train_data.trade_date)].dropna()
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
        regress_conf.max_round = 10
        tic = time.time()
        # training
        xgb_model = XGBooster(regress_conf)
        # xgb_model.set_params(tree_method='gpu_hist', max_depth=5)
        xgb_model.set_params(max_depth=5)
        print(xgb_model.get_params)
        best_score, best_round, cv_rounds, best_model = xgb_model.fit(x_train, y_train)
        alpha_logger.info('Training time cost {}s'.format(time.time() - tic))
        alpha_logger.info('best_score = {}, best_round = {}'.format(best_score, best_round))

        # Test data
        total_data_test_excess = train_data[train_data.trade_date == ref_date]
        alpha_logger.info('{0} total_data_test_excess: {1}'.format(ref_date, len(total_data_test_excess)))

        if len(total_data_test_excess) <= 0:
            alpha_logger.info('{0} HAS NO DATA!!!'.format(ref_date))
            continue

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
        # alpha_logger.info('lbound: \n{}'.format(lbound))
        # alpha_logger.info('ubound: \n{}'.format(ubound))

        # predict
        x_pred = total_data_test_excess[features]
        dpred = xgb.DMatrix(x_pred.values)
        predict_xgboost = best_model.predict(dpred)
        a = np.shape(predict_xgboost)
        predict_xgboost = np.reshape(predict_xgboost, (a[0], -1)).astype(np.float64)
        alpha_logger.info('shape_of_predict_xgboost: {}'.format(np.shape(predict_xgboost)))
        # alpha_logger.info('predict_xgboost: {}'.format(predict_xgboost))
        del xgb_model
        del best_model
        gc.collect()

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
                                                  ubound=ubound)
        except:
            import pdb
            pdb.set_trace()
            alpha_logger.info('target_pos: {}'.format(target_pos))
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
        executor.set_current(current_pos)
        net_rets.append(np.log(1. + ret - transact_cost * turn_over))
        alpha_logger.info('len_net_rets: {}, len_trade_dates: {}'.format(len(net_rets), len(trade_dates)))

        alpha_logger.info('{} is finished'.format(ref_date))

    # ret_df = pd.DataFrame({'xgb_regress': rets}, index=trade_dates)
    ret_df = pd.DataFrame({'xgb_regress': rets, 'net_xgb_regress': net_rets}, index=trade_dates)
    ret_df.loc[advanceDateByCalendar('china.sse', ref_dates[-1], freq).strftime('%Y-%m-%d')] = 0.
    ret_df = ret_df.shift(1)
    ret_df.iloc[0] = 0.
    return ret_df, tune_record


@app.route('/')
def index():
    return '<h1>hello Test<h1>'


@app.route('/runxgb')
def first_flask():
    print('backtesting>>>>>>>>>>>>>>>>>')
    # 获取因子数据
    # factor_data_org = engine.fetch_factor_range(universe, basic_factor_store,
    #                                             dates=ref_dates, used_factor_tables=[Alpha191])
    factor_data_org = engine.fetch_factor_range(universe, basic_factor_store, dates=ref_dates)

    # 获取行业数据， 风险因子
    industry = engine.fetch_industry_range(universe, dates=ref_dates)
    factor_data = pd.merge(factor_data_org, industry, on=['trade_date', 'code']).fillna(0.)

    risk_total = engine.fetch_risk_model_range(universe, dates=ref_dates)[1]

    # 获取收益率
    return_data = engine.fetch_dx_return_range(universe, dates=ref_dates,
                                               horizon=horizon, offset=0,
                                               benchmark=benchmark_code)

    # 获取benchmark
    benchmark_total = engine.fetch_benchmark_range(dates=ref_dates, benchmark=benchmark_code)
    industry_total = engine.fetch_industry_matrix_range(universe, dates=ref_dates, category=industry_name,
                                                        level=industry_level)

    # Constraintes settings
    weight_gap = 1

    industry_names = industry_list(industry_name, industry_level)
    constraint_risk = ['EARNYILD', 'LIQUIDTY', 'GROWTH', 'SIZE', 'BETA', 'MOMENTUM'] + industry_names
    total_risk_names = constraint_risk + ['benchmark', 'total']

    b_type = []
    l_val = []
    u_val = []

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
            l_val.append(-1.005)
            u_val.append(1.005)

    bounds = create_box_bounds(total_risk_names, b_type, l_val, u_val)  # # Constraintes settings

    train_data = pd.merge(factor_data, return_data, on=['trade_date', 'code']).dropna()
    print('data load success >>>>>>>>>>>>')
    ret_df, tune_record = create_scenario(train_data, weight_gap, return_data, risk_total,
                                          benchmark_total, industry_total, bounds, constraint_risk, total_risk_names)

    result_dic = {'ret_df': ret_df.to_json(), 'tune_record': tune_record.reset_index().to_json()}
    return result_dic


if __name__ == '__main__':
    data_source = 'postgresql+psycopg2://alpha:alpha@180.166.26.82:8889/alpha'
    engine = SqlEngine(data_source)

    universe = Universe('zz500')
    freq = '2b'
    benchmark_code = 905
    start_date = '2019-01-01'
    end_date = '2019-08-01'
    ref_dates = makeSchedule(start_date, end_date, freq, 'china.sse')
    horizon = map_freq(freq)
    industry_name = 'sw'
    industry_level = 1

    basic_factor_store = {
        'f0': CSQuantiles(LAST('AccountsPayablesTDays'), groups='sw1'),
        'f1': CSQuantiles(LAST('AccountsPayablesTRate'), groups='sw1'),
        'f2': CSQuantiles(LAST('AdminiExpenseRate'), groups='sw1'),
        'f3': CSQuantiles(LAST('ARTDays'), groups='sw1'),
        'f4': CSQuantiles(LAST('ARTRate'), groups='sw1'),
        'f5': CSQuantiles(LAST('ASSI'), groups='sw1'),
        'f6': CSQuantiles(LAST('BLEV'), groups='sw1'),
        'f7': CSQuantiles(LAST('BondsPayableToAsset'), groups='sw1'),
        'f8': CSQuantiles(LAST('CashRateOfSales'), groups='sw1'),
        'f9': CSQuantiles(LAST('CashToCurrentLiability'), groups='sw1'),
        'f10': CSQuantiles(LAST('CMRA'), groups='sw1'),
        'f11': CSQuantiles(LAST('CTOP'), groups='sw1'),
        'f12': CSQuantiles(LAST('CTP5'), groups='sw1'),
        'f13': CSQuantiles(LAST('CurrentAssetsRatio'), groups='sw1'),
        'f14': CSQuantiles(LAST('CurrentAssetsTRate'), groups='sw1'),
        'f15': CSQuantiles(LAST('CurrentAssetsTRate'), groups='sw1'),
        'f16': CSQuantiles(LAST('DAVOL10'), groups='sw1'),
        'f17': CSQuantiles(LAST('DAVOL20'), groups='sw1'),
        'f18': CSQuantiles(LAST('DAVOL5'), groups='sw1'),
        'f19': CSQuantiles(LAST('DDNBT'), groups='sw1'),
        'f20': CSQuantiles(LAST('DDNCR'), groups='sw1'),
        'f21': CSQuantiles(LAST('DDNSR'), groups='sw1'),
        'f22': CSQuantiles(LAST('DebtEquityRatio'), groups='sw1'),
        'f23': CSQuantiles(LAST('DebtsAssetRatio'), groups='sw1'),
        'f24': CSQuantiles(LAST('DHILO'), groups='sw1'),
        'f25': CSQuantiles(LAST('DilutedEPS'), groups='sw1'),
        'f26': CSQuantiles(LAST('DVRAT'), groups='sw1'),
        'f27': CSQuantiles(LAST('EBITToTOR'), groups='sw1'),
        'f28': CSQuantiles(LAST('EGRO'), groups='sw1'),
        'f29': CSQuantiles(LAST('EMA10'), groups='sw1'),
        'f30': CSQuantiles(LAST('EMA120'), groups='sw1'),
        'f31': CSQuantiles(LAST('EMA20'), groups='sw1'),
        'f32': CSQuantiles(LAST('EMA5'), groups='sw1'),
        'f33': CSQuantiles(LAST('EMA60'), groups='sw1'),
        'f34': CSQuantiles(LAST('EPS'), groups='sw1'),
        'f35': CSQuantiles(LAST('EquityFixedAssetRatio'), groups='sw1'),
        'f36': CSQuantiles(LAST('EquityToAsset'), groups='sw1'),
        'f37': CSQuantiles(LAST('EquityTRate'), groups='sw1'),
        'f38': CSQuantiles(LAST('ETOP'), groups='sw1'),
        'f39': CSQuantiles(LAST('ETP5'), groups='sw1'),
        'f40': CSQuantiles(LAST('FinancialExpenseRate'), groups='sw1'),
        'f41': CSQuantiles(LAST('FinancingCashGrowRate'), groups='sw1'),
        'f42': CSQuantiles(LAST('FixAssetRatio'), groups='sw1'),
        'f43': CSQuantiles(LAST('FixedAssetsTRate'), groups='sw1'),
        'f44': CSQuantiles(LAST('GrossIncomeRatio'), groups='sw1'),
        'f45': CSQuantiles(LAST('HBETA'), groups='sw1'),
        'f46': CSQuantiles(LAST('HBETA'), groups='sw1'),
        'f47': CSQuantiles(LAST('IntangibleAssetRatio'), groups='sw1'),
        'f48': CSQuantiles(LAST('InventoryTDays'), groups='sw1'),
        'f49': CSQuantiles(LAST('InventoryTRate'), groups='sw1'),
        'f50': CSQuantiles(LAST('InvestCashGrowRate'), groups='sw1'),
        'f51': CSQuantiles(LAST('LCAP'), groups='sw1'),
        'f52': CSQuantiles(LAST('LFLO'), groups='sw1'),
        'f53': CSQuantiles(LAST('LongDebtToAsset'), groups='sw1'),
        'f54': CSQuantiles(LAST('LongDebtToWorkingCapital'), groups='sw1'),
        'f55': CSQuantiles(LAST('LongTermDebtToAsset'), groups='sw1'),
        'f56': CSQuantiles(LAST('MA10'), groups='sw1'),
        'f57': CSQuantiles(LAST('MA120'), groups='sw1'),
        'f58': CSQuantiles(LAST('MA20'), groups='sw1'),
        'f59': CSQuantiles(LAST('MA5'), groups='sw1'),
        'f60': CSQuantiles(LAST('MA60'), groups='sw1'),
        'f61': CSQuantiles(LAST('MAWVAD'), groups='sw1'),
        'f62': CSQuantiles(LAST('MFI'), groups='sw1'),
        'f63': CSQuantiles(LAST('MLEV'), groups='sw1'),
        'f64': CSQuantiles(LAST('NetAssetGrowRate'), groups='sw1'),
        'f65': CSQuantiles(LAST('NetProfitGrowRate'), groups='sw1'),
        'f66': CSQuantiles(LAST('NetProfitRatio'), groups='sw1'),
        'f67': CSQuantiles(LAST('NetProfitRatio'), groups='sw1'),
        'f68': CSQuantiles(LAST('NonCurrentAssetsRatio'), groups='sw1'),
        'f69': CSQuantiles(LAST('NPParentCompanyGrowRate'), groups='sw1'),
        'f70': CSQuantiles(LAST('NPToTOR'), groups='sw1'),
        'f71': CSQuantiles(LAST('OperatingExpenseRate'), groups='sw1'),
        'f72': CSQuantiles(LAST('OperatingProfitGrowRate'), groups='sw1'),
        'f73': CSQuantiles(LAST('OperatingProfitRatio'), groups='sw1'),
        'f74': CSQuantiles(LAST('OperatingProfitToTOR'), groups='sw1'),
        'f75': CSQuantiles(LAST('OperatingRevenueGrowRate'), groups='sw1'),
        'f76': CSQuantiles(LAST('OperCashGrowRate'), groups='sw1'),
        'f77': CSQuantiles(LAST('OperCashInToCurrentLiability'), groups='sw1'),
        'f78': CSQuantiles(LAST('PB'), groups='sw1'),
        'f79': CSQuantiles(LAST('PCF'), groups='sw1'),
        'f80': CSQuantiles(LAST('PE'), groups='sw1'),
        'f81': CSQuantiles(LAST('PS'), groups='sw1'),
        'f82': CSQuantiles(LAST('PSY'), groups='sw1'),
        'f83': CSQuantiles(LAST('QuickRatio'), groups='sw1'),
        'f84': CSQuantiles(LAST('REVS10'), groups='sw1'),
        'f85': CSQuantiles(LAST('REVS20'), groups='sw1'),
        'f86': CSQuantiles(LAST('REVS5'), groups='sw1'),
        'f87': CSQuantiles(LAST('REVS5'), groups='sw1'),
        'f88': CSQuantiles(LAST('ROA5'), groups='sw1'),
        'f89': CSQuantiles(LAST('ROE'), groups='sw1'),
        'f90': CSQuantiles(LAST('ROE5'), groups='sw1'),
        'f91': CSQuantiles(LAST('RSI'), groups='sw1'),
        'f92': CSQuantiles(LAST('RSTR12'), groups='sw1'),
        'f93': CSQuantiles(LAST('RSTR24'), groups='sw1'),
        'f94': CSQuantiles(LAST('SalesCostRatio'), groups='sw1'),
        'f95': CSQuantiles(LAST('SaleServiceCashToOR'), groups='sw1'),
        'f96': CSQuantiles(LAST('SUE'), groups='sw1'),
        'f97': CSQuantiles(LAST('TaxRatio'), groups='sw1'),
        'f98': CSQuantiles(LAST('TOBT'), groups='sw1'),
        'f99': CSQuantiles(LAST('TotalAssetGrowRate'), groups='sw1'),
        'f100': CSQuantiles(LAST('TotalAssetsTRate'), groups='sw1')}

    features = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15',
                'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29',
                'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43',
                'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57',
                'f58', 'f59', 'f60', 'f61', 'f62', 'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 'f70', 'f71',
                'f72', 'f73', 'f74', 'f75', 'f76', 'f77', 'f78', 'f79', 'f80', 'f81', 'f82', 'f83', 'f84', 'f85',
                'f86', 'f87', 'f88', 'f89', 'f90', 'f91', 'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98', 'f99',
                'f100']

    label = ['dx']

    app.run(host='0.0.0.0', port='8000')

