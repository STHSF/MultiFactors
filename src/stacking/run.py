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
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
from PyFin.api import *
from alphamind.api import *
from src.conf.models import *
import numpy as np
from alphamind.execution.naiveexecutor import NaiveExecutor
from datetime import timedelta
from m1_xgb import *
from src.conf.configuration import regress_conf
import xgboost as xgb
import gc
from flask import Flask, request

app = Flask(__name__)


def create_scenario(train_data, features, label, ref_dates, freq, regress_conf, weight_gap, return_data, risk_total,
                    benchmark_total, industry_total, bounds, constraint_risk, total_risk_names, GPUs=True):
    executor = NaiveExecutor()
    trade_dates = []
    transact_cost = 0.003
    previous_pos = pd.DataFrame()
    tune_record = pd.DataFrame()
    current_pos = pd.DataFrame()
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

        tic = time.time()
        # training
        xgb_model = XGBooster(regress_conf)
        if GPUs:
            xgb_model.set_params(tree_method='gpu_hist', n_gpus=-1, max_depth=5)
        else:
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


@app.route('/backtest', methods=['POST'])
def backtest():
    data_source = 'postgresql+psycopg2://alpha:alpha@180.166.26.82:8889/alpha'
    engine = SqlEngine(data_source)

    start_date = request.form['start_date']
    end_date = request.form['end_date']
    freq = request.form['freq']
    max_round = request.form['max_round']
    GPUs = request.form['GPU']

    if start_date is None or start_date == '':
        start_date = '2019-01-01'
    if end_date is None or end_date == '':
        end_date = '2019-08-01'
    if freq is None or freq == '':
        freq = '2b'
    if max_round is None or max_round == '':
        max_round = 10

    # xgb_configuration
    regress_conf.xgb_config_r()
    regress_conf.cv_folds = None
    regress_conf.early_stop_round = 10
    regress_conf.max_round = int(max_round)

    ref_dates = makeSchedule(start_date, end_date, freq, 'china.sse')
    universe = Universe('zz500')
    benchmark_code = 905
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
        'f100': CSQuantiles(LAST('TotalAssetsTRate'), groups='sw1'),
        'f101': CSQuantiles(LAST('TotalProfitCostRatio'), groups='sw1'),
        'f102': CSQuantiles(LAST('TotalProfitGrowRate'), groups='sw1'),
        'f103': CSQuantiles(LAST('VOL10'), groups='sw1'),
        'f104': CSQuantiles(LAST('VOL120'), groups='sw1'),
        'f105': CSQuantiles(LAST('VOL20'), groups='sw1'),
        'f106': CSQuantiles(LAST('VOL240'), groups='sw1'),
        'f107': CSQuantiles(LAST('VOL5'), groups='sw1'),
        'f108': CSQuantiles(LAST('VOL60'), groups='sw1'),
        'f109': CSQuantiles(LAST('WVAD'), groups='sw1'),
        'f110': CSQuantiles(LAST('REC'), groups='sw1'),
        'f111': CSQuantiles(LAST('DAREC'), groups='sw1'),
        'f112': CSQuantiles(LAST('GREC'), groups='sw1'),
        'f113': CSQuantiles(LAST('FY12P'), groups='sw1'),
        'f114': CSQuantiles(LAST('DAREV'), groups='sw1'),
        'f115': CSQuantiles(LAST('GREV'), groups='sw1'),
        'f116': CSQuantiles(LAST('SFY12P'), groups='sw1'),
        'f117': CSQuantiles(LAST('DASREV'), groups='sw1'),
        'f118': CSQuantiles(LAST('GSREV'), groups='sw1'),
        'f119': CSQuantiles(LAST('FEARNG'), groups='sw1'),
        'f120': CSQuantiles(LAST('FSALESG'), groups='sw1'),
        'f121': CSQuantiles(LAST('TA2EV'), groups='sw1'),
        'f122': CSQuantiles(LAST('CFO2EV'), groups='sw1'),
        'f123': CSQuantiles(LAST('ACCA'), groups='sw1'),
        'f124': CSQuantiles(LAST('DEGM'), groups='sw1'),
        'f125': CSQuantiles(LAST('SUOI'), groups='sw1'),
        'f126': CSQuantiles(LAST('EARNMOM'), groups='sw1'),
        'f127': CSQuantiles(LAST('FiftyTwoWeekHigh'), groups='sw1'),
        'f128': CSQuantiles(LAST('Volatility'), groups='sw1'),
        'f129': CSQuantiles(LAST('Skewness'), groups='sw1'),
        'f130': CSQuantiles(LAST('ILLIQUIDITY'), groups='sw1'),
        'f131': CSQuantiles(LAST('BackwardADJ'), groups='sw1'),
        'f132': CSQuantiles(LAST('MACD'), groups='sw1'),
        'f133': CSQuantiles(LAST('ADTM'), groups='sw1'),
        'f134': CSQuantiles(LAST('ATR14'), groups='sw1'),
        'f135': CSQuantiles(LAST('ATR6'), groups='sw1'),
        'f136': CSQuantiles(LAST('BIAS10'), groups='sw1'),
        'f137': CSQuantiles(LAST('BIAS20'), groups='sw1'),
        'f138': CSQuantiles(LAST('BIAS5'), groups='sw1'),
        'f139': CSQuantiles(LAST('BIAS60'), groups='sw1'),
        'f140': CSQuantiles(LAST('BollDown'), groups='sw1'),
        'f141': CSQuantiles(LAST('BollUp'), groups='sw1'),
        'f142': CSQuantiles(LAST('CCI10'), groups='sw1'),
        'f143': CSQuantiles(LAST('CCI20'), groups='sw1'),
        'f144': CSQuantiles(LAST('CCI5'), groups='sw1'),
        'f145': CSQuantiles(LAST('CCI88'), groups='sw1'),
        'f146': CSQuantiles(LAST('KDJ_K'), groups='sw1'),
        'f147': CSQuantiles(LAST('KDJ_D'), groups='sw1'),
        'f148': CSQuantiles(LAST('KDJ_J'), groups='sw1'),
        'f149': CSQuantiles(LAST('ROC6'), groups='sw1'),
        'f150': CSQuantiles(LAST('ROC20'), groups='sw1'),
        'f151': CSQuantiles(LAST('SBM'), groups='sw1'),
        'f152': CSQuantiles(LAST('STM'), groups='sw1'),
        'f153': CSQuantiles(LAST('UpRVI'), groups='sw1'),
        'f154': CSQuantiles(LAST('DownRVI'), groups='sw1'),
        'f155': CSQuantiles(LAST('RVI'), groups='sw1'),
        'f156': CSQuantiles(LAST('SRMI'), groups='sw1'),
        'f157': CSQuantiles(LAST('ChandeSD'), groups='sw1'),
        'f158': CSQuantiles(LAST('ChandeSU'), groups='sw1'),
        'f159': CSQuantiles(LAST('CMO'), groups='sw1'),
        'f160': CSQuantiles(LAST('DBCD'), groups='sw1'),
        'f161': CSQuantiles(LAST('ARC'), groups='sw1'),
        'f162': CSQuantiles(LAST('OBV'), groups='sw1'),
        'f163': CSQuantiles(LAST('OBV6'), groups='sw1'),
        'f164': CSQuantiles(LAST('OBV20'), groups='sw1'),
        'f165': CSQuantiles(LAST('TVMA20'), groups='sw1'),
        'f166': CSQuantiles(LAST('TVMA6'), groups='sw1'),
        'f167': CSQuantiles(LAST('TVSTD20'), groups='sw1'),
        'f168': CSQuantiles(LAST('TVSTD6'), groups='sw1'),
        'f169': CSQuantiles(LAST('VDEA'), groups='sw1'),
        'f170': CSQuantiles(LAST('VDIFF'), groups='sw1'),
        'f171': CSQuantiles(LAST('VEMA10'), groups='sw1'),
        'f172': CSQuantiles(LAST('VEMA12'), groups='sw1'),
        'f173': CSQuantiles(LAST('VEMA26'), groups='sw1'),
        'f174': CSQuantiles(LAST('VEMA5'), groups='sw1'),
        'f175': CSQuantiles(LAST('VMACD'), groups='sw1'),
        'f176': CSQuantiles(LAST('VR'), groups='sw1'),
        'f177': CSQuantiles(LAST('VROC12'), groups='sw1'),
        'f178': CSQuantiles(LAST('VROC6'), groups='sw1'),
        'f179': CSQuantiles(LAST('VSTD10'), groups='sw1'),
        'f180': CSQuantiles(LAST('VSTD20'), groups='sw1'),
        'f181': CSQuantiles(LAST('KlingerOscillator'), groups='sw1'),
        'f182': CSQuantiles(LAST('MoneyFlow20'), groups='sw1'),
        'f183': CSQuantiles(LAST('AD'), groups='sw1'),
        'f184': CSQuantiles(LAST('AD20'), groups='sw1'),
        'f185': CSQuantiles(LAST('AD6'), groups='sw1'),
        'f186': CSQuantiles(LAST('CoppockCurve'), groups='sw1'),
        'f187': CSQuantiles(LAST('ASI'), groups='sw1'),
        'f188': CSQuantiles(LAST('ChaikinOscillator'), groups='sw1'),
        'f189': CSQuantiles(LAST('ChaikinVolatility'), groups='sw1'),
        'f190': CSQuantiles(LAST('EMV14'), groups='sw1'),
        'f191': CSQuantiles(LAST('EMV6'), groups='sw1'),
        'f192': CSQuantiles(LAST('plusDI'), groups='sw1'),
        'f193': CSQuantiles(LAST('minusDI'), groups='sw1'),
        'f194': CSQuantiles(LAST('ADX'), groups='sw1'),
        'f195': CSQuantiles(LAST('ADXR'), groups='sw1'),
        'f196': CSQuantiles(LAST('Aroon'), groups='sw1'),
        'f197': CSQuantiles(LAST('AroonDown'), groups='sw1'),
        'f198': CSQuantiles(LAST('AroonUp'), groups='sw1'),
        'f199': CSQuantiles(LAST('DEA'), groups='sw1'),
        'f200': CSQuantiles(LAST('DIFF'), groups='sw1'),
        'f201': CSQuantiles(LAST('DDI'), groups='sw1'),
        'f202': CSQuantiles(LAST('DIZ'), groups='sw1'),
        'f203': CSQuantiles(LAST('DIF'), groups='sw1'),
        'f204': CSQuantiles(LAST('MTM'), groups='sw1'),
        'f205': CSQuantiles(LAST('MTMMA'), groups='sw1'),
        'f206': CSQuantiles(LAST('PVT'), groups='sw1'),
        'f207': CSQuantiles(LAST('PVT6'), groups='sw1'),
        'f208': CSQuantiles(LAST('PVT12'), groups='sw1'),
        'f209': CSQuantiles(LAST('TRIX5'), groups='sw1'),
        'f210': CSQuantiles(LAST('TRIX10'), groups='sw1'),
        'f211': CSQuantiles(LAST('UOS'), groups='sw1'),
        'f212': CSQuantiles(LAST('MA10RegressCoeff12'), groups='sw1'),
        'f213': CSQuantiles(LAST('MA10RegressCoeff6'), groups='sw1'),
        'f214': CSQuantiles(LAST('PLRC6'), groups='sw1'),
        'f215': CSQuantiles(LAST('PLRC12'), groups='sw1'),
        'f216': CSQuantiles(LAST('SwingIndex'), groups='sw1'),
        'f217': CSQuantiles(LAST('Ulcer10'), groups='sw1'),
        'f218': CSQuantiles(LAST('Ulcer5'), groups='sw1'),
        'f219': CSQuantiles(LAST('Hurst'), groups='sw1'),
        'f220': CSQuantiles(LAST('ACD6'), groups='sw1'),
        'f221': CSQuantiles(LAST('ACD20'), groups='sw1'),
        'f222': CSQuantiles(LAST('EMA12'), groups='sw1'),
        'f223': CSQuantiles(LAST('EMA26'), groups='sw1'),
        'f224': CSQuantiles(LAST('APBMA'), groups='sw1'),
        'f225': CSQuantiles(LAST('APBMA'), groups='sw1'),
        'f226': CSQuantiles(LAST('BBIC'), groups='sw1'),
        'f227': CSQuantiles(LAST('TEMA10'), groups='sw1'),
        'f228': CSQuantiles(LAST('TEMA5'), groups='sw1'),
        'f229': CSQuantiles(LAST('MA10Close'), groups='sw1'),
        'f230': CSQuantiles(LAST('AR'), groups='sw1'),
        'f231': CSQuantiles(LAST('BR'), groups='sw1'),
        'f232': CSQuantiles(LAST('ARBR'), groups='sw1'),
        'f233': CSQuantiles(LAST('CR20'), groups='sw1'),
        'f234': CSQuantiles(LAST('MassIndex'), groups='sw1'),
        'f235': CSQuantiles(LAST('BearPower'), groups='sw1'),
        'f236': CSQuantiles(LAST('BullPower'), groups='sw1'),
        'f237': CSQuantiles(LAST('Elder'), groups='sw1'),
        'f238': CSQuantiles(LAST('NVI'), groups='sw1'),
        'f239': CSQuantiles(LAST('PVI'), groups='sw1'),
        'f240': CSQuantiles(LAST('RC12'), groups='sw1'),
        'f241': CSQuantiles(LAST('RC24'), groups='sw1'),
        'f242': CSQuantiles(LAST('JDQS20'), groups='sw1'),
        'f243': CSQuantiles(LAST('Variance20'), groups='sw1'),
        'f244': CSQuantiles(LAST('Variance60'), groups='sw1'),
        'f245': CSQuantiles(LAST('Variance120'), groups='sw1'),
        'f246': CSQuantiles(LAST('Kurtosis20'), groups='sw1'),
        'f247': CSQuantiles(LAST('Kurtosis60'), groups='sw1'),
        'f248': CSQuantiles(LAST('Kurtosis120'), groups='sw1'),
        'f249': CSQuantiles(LAST('Alpha20'), groups='sw1'),
        'f250': CSQuantiles(LAST('Alpha60'), groups='sw1'),
        'f251': CSQuantiles(LAST('Alpha120'), groups='sw1'),
        'f252': CSQuantiles(LAST('Beta20'), groups='sw1'),
        'f253': CSQuantiles(LAST('Beta60'), groups='sw1'),
        'f254': CSQuantiles(LAST('Beta60'), groups='sw1'),
        'f255': CSQuantiles(LAST('SharpeRatio20'), groups='sw1'),
        'f256': CSQuantiles(LAST('SharpeRatio60'), groups='sw1'),
        'f257': CSQuantiles(LAST('SharpeRatio120'), groups='sw1'),
        'f258': CSQuantiles(LAST('TreynorRatio20'), groups='sw1'),
        'f259': CSQuantiles(LAST('TreynorRatio60'), groups='sw1'),
        'f260': CSQuantiles(LAST('TreynorRatio120'), groups='sw1'),
        'f261': CSQuantiles(LAST('InformationRatio20'), groups='sw1'),
        'f262': CSQuantiles(LAST('InformationRatio60'), groups='sw1'),
        'f263': CSQuantiles(LAST('InformationRatio120'), groups='sw1'),
        'f264': CSQuantiles(LAST('GainVariance20'), groups='sw1'),
        'f265': CSQuantiles(LAST('GainVariance60'), groups='sw1'),
        'f266': CSQuantiles(LAST('GainVariance120'), groups='sw1'),
        'f267': CSQuantiles(LAST('LossVariance20'), groups='sw1'),
        'f268': CSQuantiles(LAST('LossVariance60'), groups='sw1'),
        'f269': CSQuantiles(LAST('LossVariance120'), groups='sw1'),
        'f270': CSQuantiles(LAST('GainLossVarianceRatio20'), groups='sw1'),
        'f271': CSQuantiles(LAST('GainLossVarianceRatio60'), groups='sw1'),
        'f272': CSQuantiles(LAST('GainLossVarianceRatio120'), groups='sw1'),
        'f273': CSQuantiles(LAST('RealizedVolatility'), groups='sw1'),
        'f274': CSQuantiles(LAST('REVS60'), groups='sw1'),
        'f275': CSQuantiles(LAST('REVS120'), groups='sw1'),
        'f276': CSQuantiles(LAST('REVS250'), groups='sw1'),
        'f277': CSQuantiles(LAST('REVS750'), groups='sw1'),
        'f278': CSQuantiles(LAST('REVS5m20'), groups='sw1'),
        'f279': CSQuantiles(LAST('REVS5m60'), groups='sw1'),
        'f280': CSQuantiles(LAST('REVS5Indu1'), groups='sw1'),
        'f281': CSQuantiles(LAST('REVS20Indu1'), groups='sw1'),
        'f282': CSQuantiles(LAST('Volumn1M'), groups='sw1'),
        'f283': CSQuantiles(LAST('Volumn3M'), groups='sw1'),
        'f284': CSQuantiles(LAST('Price1M'), groups='sw1'),
        'f285': CSQuantiles(LAST('Price3M'), groups='sw1'),
        'f286': CSQuantiles(LAST('Price1Y'), groups='sw1'),
        'f287': CSQuantiles(LAST('Rank1M'), groups='sw1'),
        'f288': CSQuantiles(LAST('CashDividendCover'), groups='sw1'),
        'f289': CSQuantiles(LAST('DividendCover'), groups='sw1'),
        'f290': CSQuantiles(LAST('DividendPaidRatio'), groups='sw1'),
        'f291': CSQuantiles(LAST('RetainedEarningRatio'), groups='sw1'),
        'f292': CSQuantiles(LAST('CashEquivalentPS'), groups='sw1'),
        'f293': CSQuantiles(LAST('DividendPS'), groups='sw1'),
        'f294': CSQuantiles(LAST('EPSTTM'), groups='sw1'),
        'f295': CSQuantiles(LAST('NetAssetPS'), groups='sw1'),
        'f296': CSQuantiles(LAST('TORPS'), groups='sw1'),
        'f297': CSQuantiles(LAST('TORPSLatest'), groups='sw1'),
        'f298': CSQuantiles(LAST('OperatingRevenuePS'), groups='sw1'),
        'f299': CSQuantiles(LAST('OperatingRevenuePSLatest'), groups='sw1'),
        'f300': CSQuantiles(LAST('OperatingProfitPS'), groups='sw1'),
        'f301': CSQuantiles(LAST('OperatingProfitPSLatest'), groups='sw1'),
        'f302': CSQuantiles(LAST('CapitalSurplusFundPS'), groups='sw1'),
        'f303': CSQuantiles(LAST('SurplusReserveFundPS'), groups='sw1'),
        'f304': CSQuantiles(LAST('UndividedProfitPS'), groups='sw1'),
        'f305': CSQuantiles(LAST('RetainedEarningsPS'), groups='sw1'),
        'f306': CSQuantiles(LAST('OperCashFlowPS'), groups='sw1'),
        'f307': CSQuantiles(LAST('CashFlowPS'), groups='sw1'),
        'f308': CSQuantiles(LAST('NetNonOIToTP'), groups='sw1'),
        'f309': CSQuantiles(LAST('NetNonOIToTPLatest'), groups='sw1'),
        'f310': CSQuantiles(LAST('PeriodCostsRate'), groups='sw1'),
        'f311': CSQuantiles(LAST('InterestCover'), groups='sw1'),
        'f312': CSQuantiles(LAST('NetProfitGrowRate3Y'), groups='sw1'),
        'f313': CSQuantiles(LAST('NetProfitGrowRate5Y'), groups='sw1'),
        'f314': CSQuantiles(LAST('OperatingRevenueGrowRate3Y'), groups='sw1'),
        'f315': CSQuantiles(LAST('OperatingRevenueGrowRate5Y'), groups='sw1'),
        'f316': CSQuantiles(LAST('NetCashFlowGrowRate'), groups='sw1'),
        'f317': CSQuantiles(LAST('NetProfitCashCover'), groups='sw1'),
        'f318': CSQuantiles(LAST('OperCashInToAsset'), groups='sw1'),
        'f319': CSQuantiles(LAST('CashConversionCycle'), groups='sw1'),
        'f320': CSQuantiles(LAST('OperatingCycle'), groups='sw1'),
        'f321': CSQuantiles(LAST('PEG3Y'), groups='sw1'),
        'f322': CSQuantiles(LAST('PEG5Y'), groups='sw1'),
        'f323': CSQuantiles(LAST('PEIndu'), groups='sw1'),
        'f324': CSQuantiles(LAST('PBIndu'), groups='sw1'),
        'f325': CSQuantiles(LAST('PSIndu'), groups='sw1'),
        'f326': CSQuantiles(LAST('PCFIndu'), groups='sw1'),
        'f327': CSQuantiles(LAST('PEHist20'), groups='sw1'),
        'f328': CSQuantiles(LAST('PEHist60'), groups='sw1'),
        'f329': CSQuantiles(LAST('PEHist120'), groups='sw1'),
        'f330': CSQuantiles(LAST('PEHist250'), groups='sw1'),
        'f331': CSQuantiles(LAST('StaticPE'), groups='sw1'),
        'f332': CSQuantiles(LAST('ForwardPE'), groups='sw1'),
        'f333': CSQuantiles(LAST('EnterpriseFCFPS'), groups='sw1'),
        'f334': CSQuantiles(LAST('ShareholderFCFPS'), groups='sw1'),
        'f335': CSQuantiles(LAST('ROEDiluted'), groups='sw1'),
        'f336': CSQuantiles(LAST('ROEAvg'), groups='sw1'),
        'f337': CSQuantiles(LAST('ROEWeighted'), groups='sw1'),
        'f338': CSQuantiles(LAST('ROECut'), groups='sw1'),
        'f339': CSQuantiles(LAST('ROECutWeighted'), groups='sw1'),
        'f340': CSQuantiles(LAST('ROIC'), groups='sw1'),
        'f341': CSQuantiles(LAST('ROAEBIT'), groups='sw1'),
        'f342': CSQuantiles(LAST('ROAEBITTTM'), groups='sw1'),
        'f343': CSQuantiles(LAST('OperatingNIToTP'), groups='sw1'),
        'f344': CSQuantiles(LAST('OperatingNIToTPLatest'), groups='sw1'),
        'f345': CSQuantiles(LAST('InvestRAssociatesToTP'), groups='sw1'),
        'f346': CSQuantiles(LAST('InvestRAssociatesToTPLatest'), groups='sw1'),
        'f347': CSQuantiles(LAST('NPCutToNP'), groups='sw1'),
        'f348': CSQuantiles(LAST('SuperQuickRatio'), groups='sw1'),
        'f349': CSQuantiles(LAST('TSEPToInterestBearDebt'), groups='sw1'),
        'f350': CSQuantiles(LAST('DebtTangibleEquityRatio'), groups='sw1'),
        'f351': CSQuantiles(LAST('TangibleAToInteBearDebt'), groups='sw1'),
        'f352': CSQuantiles(LAST('TangibleAToNetDebt'), groups='sw1'),
        'f353': CSQuantiles(LAST('NOCFToTLiability'), groups='sw1'),
        'f354': CSQuantiles(LAST('NOCFToInterestBearDebt'), groups='sw1'),
        'f355': CSQuantiles(LAST('NOCFToNetDebt'), groups='sw1'),
        'f356': CSQuantiles(LAST('TSEPToTotalCapital'), groups='sw1'),
        'f357': CSQuantiles(LAST('InteBearDebtToTotalCapital'), groups='sw1'),
        'f358': CSQuantiles(LAST('NPParentCompanyCutYOY'), groups='sw1'),
        'f359': CSQuantiles(LAST('SalesServiceCashToORLatest'), groups='sw1'),
        'f360': CSQuantiles(LAST('CashRateOfSalesLatest'), groups='sw1'),
        'f361': CSQuantiles(LAST('NOCFToOperatingNILatest'), groups='sw1'),
        'f362': CSQuantiles(LAST('TotalAssets'), groups='sw1'),
        'f363': CSQuantiles(LAST('MktValue'), groups='sw1'),
        'f364': CSQuantiles(LAST('NegMktValue'), groups='sw1'),
        'f365': CSQuantiles(LAST('TEAP'), groups='sw1'),
        'f366': CSQuantiles(LAST('NIAP'), groups='sw1'),
        'f367': CSQuantiles(LAST('TotalFixedAssets'), groups='sw1'),
        'f368': CSQuantiles(LAST('IntFreeCL'), groups='sw1'),
        'f369': CSQuantiles(LAST('IntFreeNCL'), groups='sw1'),
        'f370': CSQuantiles(LAST('IntCL'), groups='sw1'),
        'f371': CSQuantiles(LAST('IntDebt'), groups='sw1'),
        'f372': CSQuantiles(LAST('NetDebt'), groups='sw1'),
        'f373': CSQuantiles(LAST('NetTangibleAssets'), groups='sw1'),
        'f374': CSQuantiles(LAST('WorkingCapital'), groups='sw1'),
        'f375': CSQuantiles(LAST('WorkingCapital'), groups='sw1'),
        'f376': CSQuantiles(LAST('TotalPaidinCapital'), groups='sw1'),
        'f377': CSQuantiles(LAST('RetainedEarnings'), groups='sw1'),
        'f378': CSQuantiles(LAST('OperateNetIncome'), groups='sw1'),
        'f379': CSQuantiles(LAST('ValueChgProfit'), groups='sw1'),
        'f380': CSQuantiles(LAST('NetIntExpense'), groups='sw1'),
        'f381': CSQuantiles(LAST('EBIT'), groups='sw1'),
        'f382': CSQuantiles(LAST('EBITDA'), groups='sw1'),
        'f383': CSQuantiles(LAST('EBIAT'), groups='sw1'),
        'f384': CSQuantiles(LAST('NRProfitLoss'), groups='sw1'),
        'f385': CSQuantiles(LAST('NIAPCut'), groups='sw1'),
        'f386': CSQuantiles(LAST('FCFF'), groups='sw1'),
        'f387': CSQuantiles(LAST('FCFE'), groups='sw1'),
        'f388': CSQuantiles(LAST('DA'), groups='sw1'),
        'f389': CSQuantiles(LAST('TRevenueTTM'), groups='sw1'),
        'f390': CSQuantiles(LAST('TCostTTM'), groups='sw1'),
        'f391': CSQuantiles(LAST('RevenueTTM'), groups='sw1'),
        'f392': CSQuantiles(LAST('CostTTM'), groups='sw1'),
        'f393': CSQuantiles(LAST('GrossProfitTTM'), groups='sw1'),
        'f394': CSQuantiles(LAST('SalesExpenseTTM'), groups='sw1'),
        'f395': CSQuantiles(LAST('AdminExpenseTTM'), groups='sw1'),
        'f396': CSQuantiles(LAST('FinanExpenseTTM'), groups='sw1'),
        'f397': CSQuantiles(LAST('AssetImpairLossTTM'), groups='sw1'),
        'f398': CSQuantiles(LAST('NPFromOperatingTTM'), groups='sw1'),
        'f399': CSQuantiles(LAST('NPFromValueChgTTM'), groups='sw1'),
        'f400': CSQuantiles(LAST('OperateProfitTTM'), groups='sw1'),
        'f401': CSQuantiles(LAST('NonOperatingNPTTM'), groups='sw1'),
        'f402': CSQuantiles(LAST('TProfitTTM'), groups='sw1'),
        'f403': CSQuantiles(LAST('NetProfitTTM'), groups='sw1'),
        'f404': CSQuantiles(LAST('NetProfitAPTTM'), groups='sw1'),
        'f405': CSQuantiles(LAST('SaleServiceRenderCashTTM'), groups='sw1'),
        'f406': CSQuantiles(LAST('NetOperateCFTTM'), groups='sw1'),
        'f407': CSQuantiles(LAST('NetInvestCFTTM'), groups='sw1'),
        'f408': CSQuantiles(LAST('NetFinanceCFTTM'), groups='sw1'),
        'f409': CSQuantiles(LAST('GrossProfit'), groups='sw1'),
        'f410': CSQuantiles(LAST('Beta252'), groups='sw1'),
        'f411': CSQuantiles(LAST('RSTR504'), groups='sw1'),
        'f412': CSQuantiles(LAST('EPIBS'), groups='sw1'),
        'f413': CSQuantiles(LAST('CETOP'), groups='sw1'),
        'f414': CSQuantiles(LAST('DASTD'), groups='sw1'),
        'f415': CSQuantiles(LAST('CmraCNE5'), groups='sw1'),
        'f416': CSQuantiles(LAST('HsigmaCNE5'), groups='sw1'),
        'f417': CSQuantiles(LAST('HsigmaCNE5'), groups='sw1'),
        'f418': CSQuantiles(LAST('EgibsLong'), groups='sw1'),
        'f419': CSQuantiles(LAST('STOM'), groups='sw1'),
        'f420': CSQuantiles(LAST('STOQ'), groups='sw1'),
        'f421': CSQuantiles(LAST('STOA'), groups='sw1'),
        'f422': CSQuantiles(LAST('NLSIZE'), groups='sw1')}

    # features = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15',
    #             'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29',
    #             'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43',
    #             'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57',
    #             'f58', 'f59', 'f60', 'f61', 'f62', 'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 'f70', 'f71',
    #             'f72', 'f73', 'f74', 'f75', 'f76', 'f77', 'f78', 'f79', 'f80', 'f81', 'f82', 'f83', 'f84', 'f85',
    #             'f86', 'f87', 'f88', 'f89', 'f90', 'f91', 'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98', 'f99',
    #             'f100']

    features = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15',
                'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29',
                'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43',
                'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57',
                'f58', 'f59', 'f60', 'f61', 'f62', 'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 'f70', 'f71',
                'f72', 'f73', 'f74', 'f75', 'f76', 'f77', 'f78', 'f79', 'f80', 'f81', 'f82', 'f83', 'f84', 'f85',
                'f86', 'f87', 'f88', 'f89', 'f90', 'f91', 'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98', 'f99',
                'f100', 'f101', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107', 'f108', 'f109', 'f110', 'f111', 'f112',
                'f113', 'f114', 'f115', 'f116', 'f117', 'f118', 'f119', 'f120', 'f121', 'f122', 'f123', 'f124', 'f125',
                'f126', 'f127', 'f128', 'f129', 'f130', 'f131', 'f132', 'f133', 'f134', 'f135', 'f136', 'f137', 'f138',
                'f139', 'f140', 'f141', 'f142', 'f143', 'f144', 'f145', 'f146', 'f147', 'f148', 'f149', 'f150', 'f151',
                'f152', 'f153', 'f154', 'f155', 'f156', 'f157', 'f158', 'f159', 'f160', 'f161', 'f162', 'f163', 'f164',
                'f165', 'f166', 'f167', 'f168', 'f169', 'f170', 'f171', 'f172', 'f173', 'f174', 'f175', 'f176', 'f177',
                'f178', 'f179', 'f180', 'f181', 'f182', 'f183', 'f184', 'f185', 'f186', 'f187', 'f188', 'f189', 'f190',
                'f191', 'f192', 'f193', 'f194', 'f195', 'f196', 'f197', 'f198', 'f199', 'f200', 'f201', 'f202', 'f203',
                'f204', 'f205', 'f206', 'f207', 'f208', 'f209', 'f210', 'f211', 'f212', 'f213', 'f214', 'f215', 'f216',
                'f217', 'f218', 'f219', 'f220', 'f221', 'f222', 'f223', 'f224', 'f225', 'f226', 'f227', 'f228', 'f229',
                'f230', 'f231', 'f232', 'f233', 'f234', 'f235', 'f236', 'f237', 'f238', 'f239', 'f240', 'f241', 'f242',
                'f243', 'f244', 'f245', 'f246', 'f247', 'f248', 'f249', 'f250', 'f251', 'f252', 'f253', 'f254', 'f255',
                'f256', 'f257', 'f258', 'f259', 'f260', 'f261', 'f262', 'f263', 'f264', 'f265', 'f266', 'f267', 'f268',
                'f269', 'f270', 'f271', 'f272', 'f273', 'f274', 'f275', 'f276', 'f277', 'f278', 'f279', 'f280', 'f281',
                'f282', 'f283', 'f284', 'f285', 'f286', 'f287', 'f288', 'f289', 'f290', 'f291', 'f292', 'f293', 'f294',
                'f295', 'f296', 'f297', 'f298', 'f299', 'f300', 'f301', 'f302', 'f303', 'f304', 'f305', 'f306', 'f307',
                'f308', 'f309', 'f310', 'f311', 'f312', 'f313', 'f314', 'f315', 'f316', 'f317', 'f318', 'f319', 'f320',
                'f321', 'f322', 'f323', 'f324', 'f325', 'f326', 'f327', 'f328', 'f329', 'f330', 'f331', 'f332', 'f333',
                'f334', 'f335', 'f336', 'f337', 'f338', 'f339', 'f340', 'f341', 'f342', 'f343', 'f344', 'f345', 'f346',
                'f347', 'f348', 'f349', 'f350', 'f351', 'f352', 'f353', 'f354', 'f355', 'f356', 'f357', 'f358', 'f359',
                'f360', 'f361', 'f362', 'f363', 'f364', 'f365', 'f366', 'f367', 'f368', 'f369', 'f370', 'f371', 'f372',
                'f373', 'f374', 'f375', 'f376', 'f377', 'f378', 'f379', 'f380', 'f381', 'f382', 'f383', 'f384', 'f385',
                'f386', 'f387', 'f388', 'f389', 'f390', 'f391', 'f392', 'f393', 'f394', 'f395', 'f396', 'f397', 'f398',
                'f399', 'f400', 'f401', 'f402', 'f403', 'f404', 'f405', 'f406', 'f407', 'f408', 'f409', 'f410', 'f411',
                'f412', 'f413', 'f414', 'f415', 'f416', 'f417', 'f418', 'f419', 'f420', 'f421', 'f422']

    label = ['dx']
    print('>>>>>>>>>>>>>>>>> Backtesting >>>>>>>>>>>>>>>>>')
    print('>>>>>>>>> Loading Factor Data >>>>>>>>>>')
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
    print('>>>>>>>>> Loading Benchmark >>>>>>>>>>')
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

    bounds = create_box_bounds(total_risk_names, b_type, l_val, u_val)  # Constraintes settings

    train_data = pd.merge(factor_data, return_data, on=['trade_date', 'code']).dropna()
    print('>>>>>>>>>>>> Data Load Success >>>>>>>>>>>>')
    ret_df, tune_record = create_scenario(train_data, features, label, ref_dates, freq, regress_conf, weight_gap,
                                          return_data, risk_total, benchmark_total, industry_total,
                                          bounds, constraint_risk, total_risk_names, GPUs)

    result_dic = {'ret_df': ret_df.to_json(), 'tune_record': tune_record.reset_index().to_json()}
    return result_dic


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8987')
