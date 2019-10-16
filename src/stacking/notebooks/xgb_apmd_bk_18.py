#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import logging
import pandas as pd
from PyFin.api import *
from alphamind.api import *
from src.conf.models import *
import numpy as np
from alphamind.execution.naiveexecutor import NaiveExecutor
from matplotlib import pyplot as plt

data_source = 'postgresql+psycopg2://alpha:alpha@180.166.26.82:8889/alpha'
engine = SqlEngine(data_source)

universe = Universe('zz500')
freq = '5b'
benchmark_code = 905
start_date = '2018-01-01'  # 训练集的起始时间
back_start_date = '2018-01-01'  # 模型回测的起始时间
end_date = '2019-10-01'
ref_dates = makeSchedule(start_date, end_date, freq, 'china.sse')
back_ref_dates = makeSchedule(back_start_date, end_date, freq, 'china.sse')
horizon = map_freq(freq)
industry_name = 'sw'
industry_level = 1

basic_factor_store = {
    'AccountsPayablesTDays': CSQuantiles(LAST('AccountsPayablesTDays'), groups='sw1'),
    'AccountsPayablesTRate': CSQuantiles(LAST('AccountsPayablesTRate'), groups='sw1'),
    'AdminiExpenseRate': CSQuantiles(LAST('AdminiExpenseRate'), groups='sw1'),
    'ARTDays': CSQuantiles(LAST('ARTDays'), groups='sw1'),
    'ARTRate': CSQuantiles(LAST('ARTRate'), groups='sw1'),
    'ASSI': CSQuantiles(LAST('ASSI'), groups='sw1'),
    'BLEV': CSQuantiles(LAST('BLEV'), groups='sw1'),
    'BondsPayableToAsset': CSQuantiles(LAST('BondsPayableToAsset'), groups='sw1'),
    'CashRateOfSales': CSQuantiles(LAST('CashRateOfSales'), groups='sw1'),
    'CashToCurrentLiability': CSQuantiles(LAST('CashToCurrentLiability'), groups='sw1'),
    'CMRA': CSQuantiles(LAST('CMRA'), groups='sw1'),
    'CTOP': CSQuantiles(LAST('CTOP'), groups='sw1'),
    'CTP5': CSQuantiles(LAST('CTP5'), groups='sw1'),
    'CurrentAssetsRatio': CSQuantiles(LAST('CurrentAssetsRatio'), groups='sw1'),
    'CurrentAssetsTRate': CSQuantiles(LAST('CurrentAssetsTRate'), groups='sw1'),
    'CurrentRatio': CSQuantiles(LAST('CurrentRatio'), groups='sw1'),
    'DAVOL10': CSQuantiles(LAST('DAVOL10'), groups='sw1'),
    'DAVOL20': CSQuantiles(LAST('DAVOL20'), groups='sw1'),
    'DAVOL5': CSQuantiles(LAST('DAVOL5'), groups='sw1'),
    'DDNBT': CSQuantiles(LAST('DDNBT'), groups='sw1'),
    'DDNCR': CSQuantiles(LAST('DDNCR'), groups='sw1'),
    'DDNSR': CSQuantiles(LAST('DDNSR'), groups='sw1'),
    'DebtEquityRatio': CSQuantiles(LAST('DebtEquityRatio'), groups='sw1'),
    'DebtsAssetRatio': CSQuantiles(LAST('DebtsAssetRatio'), groups='sw1'),
    'DHILO': CSQuantiles(LAST('DHILO'), groups='sw1'),
    'DilutedEPS': CSQuantiles(LAST('DilutedEPS'), groups='sw1'),
    'DVRAT': CSQuantiles(LAST('DVRAT'), groups='sw1'),
    'EBITToTOR': CSQuantiles(LAST('EBITToTOR'), groups='sw1'),
    'EGRO': CSQuantiles(LAST('EGRO'), groups='sw1'),
    'EMA10': CSQuantiles(LAST('EMA10'), groups='sw1'),
    'EMA120': CSQuantiles(LAST('EMA120'), groups='sw1'),
    'EMA20': CSQuantiles(LAST('EMA20'), groups='sw1'),
    'EMA5': CSQuantiles(LAST('EMA5'), groups='sw1'),
    'EMA60': CSQuantiles(LAST('EMA60'), groups='sw1'),
    'EPS': CSQuantiles(LAST('EPS'), groups='sw1'),
    'EquityFixedAssetRatio': CSQuantiles(LAST('EquityFixedAssetRatio'), groups='sw1'),
    'EquityToAsset': CSQuantiles(LAST('EquityToAsset'), groups='sw1'),
    'EquityTRate': CSQuantiles(LAST('EquityTRate'), groups='sw1'),
    'ETOP': CSQuantiles(LAST('ETOP'), groups='sw1'),
    'ETP5': CSQuantiles(LAST('ETP5'), groups='sw1'),
    'FinancialExpenseRate': CSQuantiles(LAST('FinancialExpenseRate'), groups='sw1'),
    'FinancingCashGrowRate': CSQuantiles(LAST('FinancingCashGrowRate'), groups='sw1'),
    'FixAssetRatio': CSQuantiles(LAST('FixAssetRatio'), groups='sw1'),
    'FixedAssetsTRate': CSQuantiles(LAST('FixedAssetsTRate'), groups='sw1'),
    'GrossIncomeRatio': CSQuantiles(LAST('GrossIncomeRatio'), groups='sw1'),
    'HBETA': CSQuantiles(LAST('HBETA'), groups='sw1'),
    'HSIGMA': CSQuantiles(LAST('HBETA'), groups='sw1'),
    'IntangibleAssetRatio': CSQuantiles(LAST('IntangibleAssetRatio'), groups='sw1'),
    'InventoryTDays': CSQuantiles(LAST('InventoryTDays'), groups='sw1'),
    'InventoryTRate': CSQuantiles(LAST('InventoryTRate'), groups='sw1'),
    'InvestCashGrowRate': CSQuantiles(LAST('InvestCashGrowRate'), groups='sw1'),
    'LCAP': CSQuantiles(LAST('LCAP'), groups='sw1'),
    'LFLO': CSQuantiles(LAST('LFLO'), groups='sw1'),
    'LongDebtToAsset': CSQuantiles(LAST('LongDebtToAsset'), groups='sw1'),
    'LongDebtToWorkingCapital': CSQuantiles(LAST('LongDebtToWorkingCapital'), groups='sw1'),
    'LongTermDebtToAsset': CSQuantiles(LAST('LongTermDebtToAsset'), groups='sw1'),
    'MA10': CSQuantiles(LAST('MA10'), groups='sw1'),
    'MA120': CSQuantiles(LAST('MA120'), groups='sw1'),
    'MA20': CSQuantiles(LAST('MA20'), groups='sw1'),
    'MA5': CSQuantiles(LAST('MA5'), groups='sw1'),
    'MA60': CSQuantiles(LAST('MA60'), groups='sw1'),
    'MAWVAD': CSQuantiles(LAST('MAWVAD'), groups='sw1'),
    'MFI': CSQuantiles(LAST('MFI'), groups='sw1'),
    'MLEV': CSQuantiles(LAST('MLEV'), groups='sw1'),
    'NetAssetGrowRate': CSQuantiles(LAST('NetAssetGrowRate'), groups='sw1'),
    'NetProfitGrowRate': CSQuantiles(LAST('NetProfitGrowRate'), groups='sw1'),
    'NetProfitRatio': CSQuantiles(LAST('NetProfitRatio'), groups='sw1'),
    'NOCFToOperatingNI': CSQuantiles(LAST('NetProfitRatio'), groups='sw1'),
    'NonCurrentAssetsRatio': CSQuantiles(LAST('NonCurrentAssetsRatio'), groups='sw1'),
    'NPParentCompanyGrowRate': CSQuantiles(LAST('NPParentCompanyGrowRate'), groups='sw1'),
    'NPToTOR': CSQuantiles(LAST('NPToTOR'), groups='sw1'),
    'OperatingExpenseRate': CSQuantiles(LAST('OperatingExpenseRate'), groups='sw1'),
    'OperatingProfitGrowRate': CSQuantiles(LAST('OperatingProfitGrowRate'), groups='sw1'),
    'OperatingProfitRatio': CSQuantiles(LAST('OperatingProfitRatio'), groups='sw1'),
    'OperatingProfitToTOR': CSQuantiles(LAST('OperatingProfitToTOR'), groups='sw1'),
    'OperatingRevenueGrowRate': CSQuantiles(LAST('OperatingRevenueGrowRate'), groups='sw1'),
    'OperCashGrowRate': CSQuantiles(LAST('OperCashGrowRate'), groups='sw1'),
    'OperCashInToCurrentLiability': CSQuantiles(LAST('OperCashInToCurrentLiability'), groups='sw1'),
    'PB': CSQuantiles(LAST('PB'), groups='sw1'),
    'PCF': CSQuantiles(LAST('PCF'), groups='sw1'),
    'PE': CSQuantiles(LAST('PE'), groups='sw1'),
    'PS': CSQuantiles(LAST('PS'), groups='sw1'),
    'PSY': CSQuantiles(LAST('PSY'), groups='sw1'),
    'QuickRatio': CSQuantiles(LAST('QuickRatio'), groups='sw1'),
    'REVS10': CSQuantiles(LAST('REVS10'), groups='sw1'),
    'REVS20': CSQuantiles(LAST('REVS20'), groups='sw1'),
    'REVS5': CSQuantiles(LAST('REVS5'), groups='sw1'),
    'ROA': CSQuantiles(LAST('REVS5'), groups='sw1'),
    'ROA5': CSQuantiles(LAST('ROA5'), groups='sw1'),
    'ROE': CSQuantiles(LAST('ROE'), groups='sw1'),
    'ROE5': CSQuantiles(LAST('ROE5'), groups='sw1'),
    'RSI': CSQuantiles(LAST('RSI'), groups='sw1'),
    'RSTR12': CSQuantiles(LAST('RSTR12'), groups='sw1'),
    'RSTR24': CSQuantiles(LAST('RSTR24'), groups='sw1'),
    'SalesCostRatio': CSQuantiles(LAST('SalesCostRatio'), groups='sw1'),
    'SaleServiceCashToOR': CSQuantiles(LAST('SaleServiceCashToOR'), groups='sw1'),
    'SUE': CSQuantiles(LAST('SUE'), groups='sw1'),
    'TaxRatio': CSQuantiles(LAST('TaxRatio'), groups='sw1'),
    'TOBT': CSQuantiles(LAST('TOBT'), groups='sw1'),
    'TotalAssetGrowRate': CSQuantiles(LAST('TotalAssetGrowRate'), groups='sw1'),
    'TotalAssetsTRate': CSQuantiles(LAST('TotalAssetsTRate'), groups='sw1'),
    'TotalProfitCostRatio': CSQuantiles(LAST('TotalProfitCostRatio'), groups='sw1'),
    'TotalProfitGrowRate': CSQuantiles(LAST('TotalProfitGrowRate'), groups='sw1'),
    'VOL10': CSQuantiles(LAST('VOL10'), groups='sw1'),
    'VOL120': CSQuantiles(LAST('VOL120'), groups='sw1'),
    'VOL20': CSQuantiles(LAST('VOL20'), groups='sw1'),
    'VOL240': CSQuantiles(LAST('VOL240'), groups='sw1'),
    'VOL5': CSQuantiles(LAST('VOL5'), groups='sw1'),
    'VOL60': CSQuantiles(LAST('VOL60'), groups='sw1'),
    'WVAD': CSQuantiles(LAST('WVAD'), groups='sw1'),
    'REC': CSQuantiles(LAST('REC'), groups='sw1'),
    'DAREC': CSQuantiles(LAST('DAREC'), groups='sw1'),
    'GREC': CSQuantiles(LAST('GREC'), groups='sw1'),
    'FY12P': CSQuantiles(LAST('FY12P'), groups='sw1'),
    'DAREV': CSQuantiles(LAST('DAREV'), groups='sw1'),
    'GREV': CSQuantiles(LAST('GREV'), groups='sw1'),
    'SFY12P': CSQuantiles(LAST('SFY12P'), groups='sw1'),
    'DASREV': CSQuantiles(LAST('DASREV'), groups='sw1'),
    'GSREV': CSQuantiles(LAST('GSREV'), groups='sw1'),
    'FEARNG': CSQuantiles(LAST('FEARNG'), groups='sw1'),
    'FSALESG': CSQuantiles(LAST('FSALESG'), groups='sw1'),
    'TA2EV': CSQuantiles(LAST('TA2EV'), groups='sw1'),
    'CFO2EV': CSQuantiles(LAST('CFO2EV'), groups='sw1'),
    'ACCA': CSQuantiles(LAST('ACCA'), groups='sw1'),
    'DEGM': CSQuantiles(LAST('DEGM'), groups='sw1'),
    'SUOI': CSQuantiles(LAST('SUOI'), groups='sw1'),
    'EARNMOM': CSQuantiles(LAST('EARNMOM'), groups='sw1'),
    'FiftyTwoWeekHigh': CSQuantiles(LAST('FiftyTwoWeekHigh'), groups='sw1'),
    'Volatility': CSQuantiles(LAST('Volatility'), groups='sw1'),
    'Skewness': CSQuantiles(LAST('Skewness'), groups='sw1'),
    'ILLIQUIDITY': CSQuantiles(LAST('ILLIQUIDITY'), groups='sw1'),
    'BackwardADJ': CSQuantiles(LAST('BackwardADJ'), groups='sw1'),
    'MACD': CSQuantiles(LAST('MACD'), groups='sw1'),
    'ADTM': CSQuantiles(LAST('ADTM'), groups='sw1'),
    'ATR14': CSQuantiles(LAST('ATR14'), groups='sw1'),
    'ATR6': CSQuantiles(LAST('ATR6'), groups='sw1'),
    'BIAS10': CSQuantiles(LAST('BIAS10'), groups='sw1'),
    'BIAS20': CSQuantiles(LAST('BIAS20'), groups='sw1'),
    'BIAS5': CSQuantiles(LAST('BIAS5'), groups='sw1'),
    'BIAS60': CSQuantiles(LAST('BIAS60'), groups='sw1'),
    'BollDown': CSQuantiles(LAST('BollDown'), groups='sw1'),
    'BollUp': CSQuantiles(LAST('BollUp'), groups='sw1'),
    'CCI10': CSQuantiles(LAST('CCI10'), groups='sw1'),
    'CCI20': CSQuantiles(LAST('CCI20'), groups='sw1'),
    'CCI5': CSQuantiles(LAST('CCI5'), groups='sw1'),
    'CCI88': CSQuantiles(LAST('CCI88'), groups='sw1'),
    'KDJ_K': CSQuantiles(LAST('KDJ_K'), groups='sw1'),
    'KDJ_D': CSQuantiles(LAST('KDJ_D'), groups='sw1'),
    'KDJ_J': CSQuantiles(LAST('KDJ_J'), groups='sw1'),
    'ROC6': CSQuantiles(LAST('ROC6'), groups='sw1'),
    'ROC20': CSQuantiles(LAST('ROC20'), groups='sw1'),
    'SBM': CSQuantiles(LAST('SBM'), groups='sw1'),
    'STM': CSQuantiles(LAST('STM'), groups='sw1'),
    'UpRVI': CSQuantiles(LAST('UpRVI'), groups='sw1'),
    'DownRVI': CSQuantiles(LAST('DownRVI'), groups='sw1'),
    'RVI': CSQuantiles(LAST('RVI'), groups='sw1'),
    'SRMI': CSQuantiles(LAST('SRMI'), groups='sw1'),
    'ChandeSD': CSQuantiles(LAST('ChandeSD'), groups='sw1'),
    'ChandeSU': CSQuantiles(LAST('ChandeSU'), groups='sw1'),
    'CMO': CSQuantiles(LAST('CMO'), groups='sw1'),
    'DBCD': CSQuantiles(LAST('DBCD'), groups='sw1'),
    'ARC': CSQuantiles(LAST('ARC'), groups='sw1'),
    'OBV': CSQuantiles(LAST('OBV'), groups='sw1'),
    'OBV6': CSQuantiles(LAST('OBV6'), groups='sw1'),
    'OBV20': CSQuantiles(LAST('OBV20'), groups='sw1'),
    'TVMA20': CSQuantiles(LAST('TVMA20'), groups='sw1'),
    'TVMA6': CSQuantiles(LAST('TVMA6'), groups='sw1'),
    'TVSTD20': CSQuantiles(LAST('TVSTD20'), groups='sw1'),
    'TVSTD6': CSQuantiles(LAST('TVSTD6'), groups='sw1'),
    'VDEA': CSQuantiles(LAST('VDEA'), groups='sw1'),
    'VDIFF': CSQuantiles(LAST('VDIFF'), groups='sw1'),
    'VEMA10': CSQuantiles(LAST('VEMA10'), groups='sw1'),
    'VEMA12': CSQuantiles(LAST('VEMA12'), groups='sw1'),
    'VEMA26': CSQuantiles(LAST('VEMA26'), groups='sw1'),
    'VEMA5': CSQuantiles(LAST('VEMA5'), groups='sw1'),
    'VMACD': CSQuantiles(LAST('VMACD'), groups='sw1'),
    'VR': CSQuantiles(LAST('VR'), groups='sw1'),
    'VROC12': CSQuantiles(LAST('VROC12'), groups='sw1'),
    'VROC6': CSQuantiles(LAST('VROC6'), groups='sw1'),
    'VSTD10': CSQuantiles(LAST('VSTD10'), groups='sw1'),
    'VSTD20': CSQuantiles(LAST('VSTD20'), groups='sw1'),
    'KlingerOscillator': CSQuantiles(LAST('KlingerOscillator'), groups='sw1'),
    'MoneyFlow20': CSQuantiles(LAST('MoneyFlow20'), groups='sw1'),
    'AD': CSQuantiles(LAST('AD'), groups='sw1'),
    'AD20': CSQuantiles(LAST('AD20'), groups='sw1'),
    'AD6': CSQuantiles(LAST('AD6'), groups='sw1'),
    'CoppockCurve': CSQuantiles(LAST('CoppockCurve'), groups='sw1'),
    'ASI': CSQuantiles(LAST('ASI'), groups='sw1'),
    'ChaikinOscillator': CSQuantiles(LAST('ChaikinOscillator'), groups='sw1'),
    'ChaikinVolatility': CSQuantiles(LAST('ChaikinVolatility'), groups='sw1'),
    'EMV14': CSQuantiles(LAST('EMV14'), groups='sw1'),
    'EMV6': CSQuantiles(LAST('EMV6'), groups='sw1'),
    'plusDI': CSQuantiles(LAST('plusDI'), groups='sw1'),
    'minusDI': CSQuantiles(LAST('minusDI'), groups='sw1'),
    'ADX': CSQuantiles(LAST('ADX'), groups='sw1'),
    'ADXR': CSQuantiles(LAST('ADXR'), groups='sw1'),
    'Aroon': CSQuantiles(LAST('Aroon'), groups='sw1'),
    'AroonDown': CSQuantiles(LAST('AroonDown'), groups='sw1'),
    'AroonUp': CSQuantiles(LAST('AroonUp'), groups='sw1'),
    'DEA': CSQuantiles(LAST('DEA'), groups='sw1'),
    'DIFF': CSQuantiles(LAST('DIFF'), groups='sw1'),
    'DDI': CSQuantiles(LAST('DDI'), groups='sw1'),
    'DIZ': CSQuantiles(LAST('DIZ'), groups='sw1'),
    'DIF': CSQuantiles(LAST('DIF'), groups='sw1'),
    'MTM': CSQuantiles(LAST('MTM'), groups='sw1'),
    'MTMMA': CSQuantiles(LAST('MTMMA'), groups='sw1'),
    'PVT': CSQuantiles(LAST('PVT'), groups='sw1'),
    'PVT6': CSQuantiles(LAST('PVT6'), groups='sw1'),
    'PVT12': CSQuantiles(LAST('PVT12'), groups='sw1'),
    'TRIX5': CSQuantiles(LAST('TRIX5'), groups='sw1'),
    'TRIX10': CSQuantiles(LAST('TRIX10'), groups='sw1'),
    'UOS': CSQuantiles(LAST('UOS'), groups='sw1'),
    'MA10RegressCoeff12': CSQuantiles(LAST('MA10RegressCoeff12'), groups='sw1'),
    'MA10RegressCoeff6': CSQuantiles(LAST('MA10RegressCoeff6'), groups='sw1'),
    'PLRC6': CSQuantiles(LAST('PLRC6'), groups='sw1'),
    'PLRC12': CSQuantiles(LAST('PLRC12'), groups='sw1'),
    'SwingIndex': CSQuantiles(LAST('SwingIndex'), groups='sw1'),
    'Ulcer10': CSQuantiles(LAST('Ulcer10'), groups='sw1'),
    'Ulcer5': CSQuantiles(LAST('Ulcer5'), groups='sw1'),
    'Hurst': CSQuantiles(LAST('Hurst'), groups='sw1'),
    'ACD6': CSQuantiles(LAST('ACD6'), groups='sw1'),
    'ACD20': CSQuantiles(LAST('ACD20'), groups='sw1'),
    'EMA12': CSQuantiles(LAST('EMA12'), groups='sw1'),
    'EMA26': CSQuantiles(LAST('EMA26'), groups='sw1'),
    'APBMA': CSQuantiles(LAST('APBMA'), groups='sw1'),
    'BBI': CSQuantiles(LAST('APBMA'), groups='sw1'),
    'BBIC': CSQuantiles(LAST('BBIC'), groups='sw1'),
    'TEMA10': CSQuantiles(LAST('TEMA10'), groups='sw1'),
    'TEMA5': CSQuantiles(LAST('TEMA5'), groups='sw1'),
    'MA10Close': CSQuantiles(LAST('MA10Close'), groups='sw1'),
    'AR': CSQuantiles(LAST('AR'), groups='sw1'),
    'BR': CSQuantiles(LAST('BR'), groups='sw1'),
    'ARBR': CSQuantiles(LAST('ARBR'), groups='sw1'),
    'CR20': CSQuantiles(LAST('CR20'), groups='sw1'),
    'MassIndex': CSQuantiles(LAST('MassIndex'), groups='sw1'),
    'BearPower': CSQuantiles(LAST('BearPower'), groups='sw1'),
    'BullPower': CSQuantiles(LAST('BullPower'), groups='sw1'),
    'Elder': CSQuantiles(LAST('Elder'), groups='sw1'),
    'NVI': CSQuantiles(LAST('NVI'), groups='sw1'),
    'PVI': CSQuantiles(LAST('PVI'), groups='sw1'),
    'RC12': CSQuantiles(LAST('RC12'), groups='sw1'),
    'RC24': CSQuantiles(LAST('RC24'), groups='sw1'),
    'JDQS20': CSQuantiles(LAST('JDQS20'), groups='sw1'),
    'Variance20': CSQuantiles(LAST('Variance20'), groups='sw1'),
    'Variance60': CSQuantiles(LAST('Variance60'), groups='sw1'),
    'Variance120': CSQuantiles(LAST('Variance120'), groups='sw1'),
    'Kurtosis20': CSQuantiles(LAST('Kurtosis20'), groups='sw1'),
    'Kurtosis60': CSQuantiles(LAST('Kurtosis60'), groups='sw1'),
    'Kurtosis120': CSQuantiles(LAST('Kurtosis120'), groups='sw1'),
    'Alpha20': CSQuantiles(LAST('Alpha20'), groups='sw1'),
    'Alpha60': CSQuantiles(LAST('Alpha60'), groups='sw1'),
    'Alpha120': CSQuantiles(LAST('Alpha120'), groups='sw1'),
    'Beta20': CSQuantiles(LAST('Beta20'), groups='sw1'),
    'Beta60': CSQuantiles(LAST('Beta60'), groups='sw1'),
    'Beta120': CSQuantiles(LAST('Beta60'), groups='sw1'),
    'SharpeRatio20': CSQuantiles(LAST('SharpeRatio20'), groups='sw1'),
    'SharpeRatio60': CSQuantiles(LAST('SharpeRatio60'), groups='sw1'),
    'SharpeRatio120': CSQuantiles(LAST('SharpeRatio120'), groups='sw1'),
    'TreynorRatio20': CSQuantiles(LAST('TreynorRatio20'), groups='sw1'),
    'TreynorRatio60': CSQuantiles(LAST('TreynorRatio60'), groups='sw1'),
    'TreynorRatio120': CSQuantiles(LAST('TreynorRatio120'), groups='sw1'),
    'InformationRatio20': CSQuantiles(LAST('InformationRatio20'), groups='sw1'),
    'InformationRatio60': CSQuantiles(LAST('InformationRatio60'), groups='sw1'),
    'InformationRatio120': CSQuantiles(LAST('InformationRatio120'), groups='sw1'),
    'GainVariance20': CSQuantiles(LAST('GainVariance20'), groups='sw1'),
    'GainVariance60': CSQuantiles(LAST('GainVariance60'), groups='sw1'),
    'GainVariance120': CSQuantiles(LAST('GainVariance120'), groups='sw1'),
    'LossVariance20': CSQuantiles(LAST('LossVariance20'), groups='sw1'),
    'LossVariance60': CSQuantiles(LAST('LossVariance60'), groups='sw1'),
    'LossVariance120': CSQuantiles(LAST('LossVariance120'), groups='sw1'),
    'GainLossVarianceRatio20': CSQuantiles(LAST('GainLossVarianceRatio20'), groups='sw1'),
    'GainLossVarianceRatio60': CSQuantiles(LAST('GainLossVarianceRatio60'), groups='sw1'),
    'GainLossVarianceRatio120': CSQuantiles(LAST('GainLossVarianceRatio120'), groups='sw1'),
    'RealizedVolatility': CSQuantiles(LAST('RealizedVolatility'), groups='sw1'),
    'REVS60': CSQuantiles(LAST('REVS60'), groups='sw1'),
    'REVS120': CSQuantiles(LAST('REVS120'), groups='sw1'),
    'REVS250': CSQuantiles(LAST('REVS250'), groups='sw1'),
    'REVS750': CSQuantiles(LAST('REVS750'), groups='sw1'),
    'REVS5m20': CSQuantiles(LAST('REVS5m20'), groups='sw1'),
    'REVS5m60': CSQuantiles(LAST('REVS5m60'), groups='sw1'),
    'REVS5Indu1': CSQuantiles(LAST('REVS5Indu1'), groups='sw1'),
    'REVS20Indu1': CSQuantiles(LAST('REVS20Indu1'), groups='sw1'),
    'Volumn1M': CSQuantiles(LAST('Volumn1M'), groups='sw1'),
    'Volumn3M': CSQuantiles(LAST('Volumn3M'), groups='sw1'),
    'Price1M': CSQuantiles(LAST('Price1M'), groups='sw1'),
    'Price3M': CSQuantiles(LAST('Price3M'), groups='sw1'),
    'Price1Y': CSQuantiles(LAST('Price1Y'), groups='sw1'),
    'Rank1M': CSQuantiles(LAST('Rank1M'), groups='sw1'),
    'CashDividendCover': CSQuantiles(LAST('CashDividendCover'), groups='sw1'),
    'DividendCover': CSQuantiles(LAST('DividendCover'), groups='sw1'),
    'DividendPaidRatio': CSQuantiles(LAST('DividendPaidRatio'), groups='sw1'),
    'RetainedEarningRatio': CSQuantiles(LAST('RetainedEarningRatio'), groups='sw1'),
    'CashEquivalentPS': CSQuantiles(LAST('CashEquivalentPS'), groups='sw1'),
    'DividendPS': CSQuantiles(LAST('DividendPS'), groups='sw1'),
    'EPSTTM': CSQuantiles(LAST('EPSTTM'), groups='sw1'),
    'NetAssetPS': CSQuantiles(LAST('NetAssetPS'), groups='sw1'),
    'TORPS': CSQuantiles(LAST('TORPS'), groups='sw1'),
    'TORPSLatest': CSQuantiles(LAST('TORPSLatest'), groups='sw1'),
    'OperatingRevenuePS': CSQuantiles(LAST('OperatingRevenuePS'), groups='sw1'),
    'OperatingRevenuePSLatest': CSQuantiles(LAST('OperatingRevenuePSLatest'), groups='sw1'),
    'OperatingProfitPS': CSQuantiles(LAST('OperatingProfitPS'), groups='sw1'),
    'OperatingProfitPSLatest': CSQuantiles(LAST('OperatingProfitPSLatest'), groups='sw1'),
    'CapitalSurplusFundPS': CSQuantiles(LAST('CapitalSurplusFundPS'), groups='sw1'),
    'SurplusReserveFundPS': CSQuantiles(LAST('SurplusReserveFundPS'), groups='sw1'),
    'UndividedProfitPS': CSQuantiles(LAST('UndividedProfitPS'), groups='sw1'),
    'RetainedEarningsPS': CSQuantiles(LAST('RetainedEarningsPS'), groups='sw1'),
    'OperCashFlowPS': CSQuantiles(LAST('OperCashFlowPS'), groups='sw1'),
    'CashFlowPS': CSQuantiles(LAST('CashFlowPS'), groups='sw1'),
    'NetNonOIToTP': CSQuantiles(LAST('NetNonOIToTP'), groups='sw1'),
    'NetNonOIToTPLatest': CSQuantiles(LAST('NetNonOIToTPLatest'), groups='sw1'),
    'PeriodCostsRate': CSQuantiles(LAST('PeriodCostsRate'), groups='sw1'),
    'InterestCover': CSQuantiles(LAST('InterestCover'), groups='sw1'),
    'NetProfitGrowRate3Y': CSQuantiles(LAST('NetProfitGrowRate3Y'), groups='sw1'),
    'NetProfitGrowRate5Y': CSQuantiles(LAST('NetProfitGrowRate5Y'), groups='sw1'),
    'OperatingRevenueGrowRate3Y': CSQuantiles(LAST('OperatingRevenueGrowRate3Y'), groups='sw1'),
    'OperatingRevenueGrowRate5Y': CSQuantiles(LAST('OperatingRevenueGrowRate5Y'), groups='sw1'),
    'NetCashFlowGrowRate': CSQuantiles(LAST('NetCashFlowGrowRate'), groups='sw1'),
    'NetProfitCashCover': CSQuantiles(LAST('NetProfitCashCover'), groups='sw1'),
    'OperCashInToAsset': CSQuantiles(LAST('OperCashInToAsset'), groups='sw1'),
    'CashConversionCycle': CSQuantiles(LAST('CashConversionCycle'), groups='sw1'),
    'OperatingCycle': CSQuantiles(LAST('OperatingCycle'), groups='sw1'),
    'PEG3Y': CSQuantiles(LAST('PEG3Y'), groups='sw1'),
    'PEG5Y': CSQuantiles(LAST('PEG5Y'), groups='sw1'),
    'PEIndu': CSQuantiles(LAST('PEIndu'), groups='sw1'),
    'PBIndu': CSQuantiles(LAST('PBIndu'), groups='sw1'),
    'PSIndu': CSQuantiles(LAST('PSIndu'), groups='sw1'),
    'PCFIndu': CSQuantiles(LAST('PCFIndu'), groups='sw1'),
    'PEHist20': CSQuantiles(LAST('PEHist20'), groups='sw1'),
    'PEHist60': CSQuantiles(LAST('PEHist60'), groups='sw1'),
    'PEHist120': CSQuantiles(LAST('PEHist120'), groups='sw1'),
    'PEHist250': CSQuantiles(LAST('PEHist250'), groups='sw1'),
    'StaticPE': CSQuantiles(LAST('StaticPE'), groups='sw1'),
    'ForwardPE': CSQuantiles(LAST('ForwardPE'), groups='sw1'),
    'EnterpriseFCFPS': CSQuantiles(LAST('EnterpriseFCFPS'), groups='sw1'),
    'ShareholderFCFPS': CSQuantiles(LAST('ShareholderFCFPS'), groups='sw1'),
    'ROEDiluted': CSQuantiles(LAST('ROEDiluted'), groups='sw1'),
    'ROEAvg': CSQuantiles(LAST('ROEAvg'), groups='sw1'),
    'ROEWeighted': CSQuantiles(LAST('ROEWeighted'), groups='sw1'),
    'ROECut': CSQuantiles(LAST('ROECut'), groups='sw1'),
    'ROECutWeighted': CSQuantiles(LAST('ROECutWeighted'), groups='sw1'),
    'ROIC': CSQuantiles(LAST('ROIC'), groups='sw1'),
    'ROAEBIT': CSQuantiles(LAST('ROAEBIT'), groups='sw1'),
    'ROAEBITTTM': CSQuantiles(LAST('ROAEBITTTM'), groups='sw1'),
    'OperatingNIToTP': CSQuantiles(LAST('OperatingNIToTP'), groups='sw1'),
    'OperatingNIToTPLatest': CSQuantiles(LAST('OperatingNIToTPLatest'), groups='sw1'),
    'InvestRAssociatesToTP': CSQuantiles(LAST('InvestRAssociatesToTP'), groups='sw1'),
    'InvestRAssociatesToTPLatest': CSQuantiles(LAST('InvestRAssociatesToTPLatest'), groups='sw1'),
    'NPCutToNP': CSQuantiles(LAST('NPCutToNP'), groups='sw1'),
    'SuperQuickRatio': CSQuantiles(LAST('SuperQuickRatio'), groups='sw1'),
    'TSEPToInterestBearDebt': CSQuantiles(LAST('TSEPToInterestBearDebt'), groups='sw1'),
    'DebtTangibleEquityRatio': CSQuantiles(LAST('DebtTangibleEquityRatio'), groups='sw1'),
    'TangibleAToInteBearDebt': CSQuantiles(LAST('TangibleAToInteBearDebt'), groups='sw1'),
    'TangibleAToNetDebt': CSQuantiles(LAST('TangibleAToNetDebt'), groups='sw1'),
    'NOCFToTLiability': CSQuantiles(LAST('NOCFToTLiability'), groups='sw1'),
    'NOCFToInterestBearDebt': CSQuantiles(LAST('NOCFToInterestBearDebt'), groups='sw1'),
    'NOCFToNetDebt': CSQuantiles(LAST('NOCFToNetDebt'), groups='sw1'),
    'TSEPToTotalCapital': CSQuantiles(LAST('TSEPToTotalCapital'), groups='sw1'),
    'InteBearDebtToTotalCapital': CSQuantiles(LAST('InteBearDebtToTotalCapital'), groups='sw1'),
    'NPParentCompanyCutYOY': CSQuantiles(LAST('NPParentCompanyCutYOY'), groups='sw1'),
    'SalesServiceCashToORLatest': CSQuantiles(LAST('SalesServiceCashToORLatest'), groups='sw1'),
    'CashRateOfSalesLatest': CSQuantiles(LAST('CashRateOfSalesLatest'), groups='sw1'),
    'NOCFToOperatingNILatest': CSQuantiles(LAST('NOCFToOperatingNILatest'), groups='sw1'),
    'TotalAssets': CSQuantiles(LAST('TotalAssets'), groups='sw1'),
    'MktValue': CSQuantiles(LAST('MktValue'), groups='sw1'),
    'NegMktValue': CSQuantiles(LAST('NegMktValue'), groups='sw1'),
    'TEAP': CSQuantiles(LAST('TEAP'), groups='sw1'),
    'NIAP': CSQuantiles(LAST('NIAP'), groups='sw1'),
    'TotalFixedAssets': CSQuantiles(LAST('TotalFixedAssets'), groups='sw1'),
    'IntFreeCL': CSQuantiles(LAST('IntFreeCL'), groups='sw1'),
    'IntFreeNCL': CSQuantiles(LAST('IntFreeNCL'), groups='sw1'),
    'IntCL': CSQuantiles(LAST('IntCL'), groups='sw1'),
    'IntDebt': CSQuantiles(LAST('IntDebt'), groups='sw1'),
    'NetDebt': CSQuantiles(LAST('NetDebt'), groups='sw1'),
    'NetTangibleAssets': CSQuantiles(LAST('NetTangibleAssets'), groups='sw1'),
    'WorkingCapital': CSQuantiles(LAST('WorkingCapital'), groups='sw1'),
    'NetWorkingCapital': CSQuantiles(LAST('WorkingCapital'), groups='sw1'),
    'TotalPaidinCapital': CSQuantiles(LAST('TotalPaidinCapital'), groups='sw1'),
    'RetainedEarnings': CSQuantiles(LAST('RetainedEarnings'), groups='sw1'),
    'OperateNetIncome': CSQuantiles(LAST('OperateNetIncome'), groups='sw1'),
    'ValueChgProfit': CSQuantiles(LAST('ValueChgProfit'), groups='sw1'),
    'NetIntExpense': CSQuantiles(LAST('NetIntExpense'), groups='sw1'),
    'EBIT': CSQuantiles(LAST('EBIT'), groups='sw1'),
    'EBITDA': CSQuantiles(LAST('EBITDA'), groups='sw1'),
    'EBIAT': CSQuantiles(LAST('EBIAT'), groups='sw1'),
    'NRProfitLoss': CSQuantiles(LAST('NRProfitLoss'), groups='sw1'),
    'NIAPCut': CSQuantiles(LAST('NIAPCut'), groups='sw1'),
    'FCFF': CSQuantiles(LAST('FCFF'), groups='sw1'),
    'FCFE': CSQuantiles(LAST('FCFE'), groups='sw1'),
    'DA': CSQuantiles(LAST('DA'), groups='sw1'),
    'TRevenueTTM': CSQuantiles(LAST('TRevenueTTM'), groups='sw1'),
    'TCostTTM': CSQuantiles(LAST('TCostTTM'), groups='sw1'),
    'RevenueTTM': CSQuantiles(LAST('RevenueTTM'), groups='sw1'),
    'CostTTM': CSQuantiles(LAST('CostTTM'), groups='sw1'),
    'GrossProfitTTM': CSQuantiles(LAST('GrossProfitTTM'), groups='sw1'),
    'SalesExpenseTTM': CSQuantiles(LAST('SalesExpenseTTM'), groups='sw1'),
    'AdminExpenseTTM': CSQuantiles(LAST('AdminExpenseTTM'), groups='sw1'),
    'FinanExpenseTTM': CSQuantiles(LAST('FinanExpenseTTM'), groups='sw1'),
    'AssetImpairLossTTM': CSQuantiles(LAST('AssetImpairLossTTM'), groups='sw1'),
    'NPFromOperatingTTM': CSQuantiles(LAST('NPFromOperatingTTM'), groups='sw1'),
    'NPFromValueChgTTM': CSQuantiles(LAST('NPFromValueChgTTM'), groups='sw1'),
    'OperateProfitTTM': CSQuantiles(LAST('OperateProfitTTM'), groups='sw1'),
    'NonOperatingNPTTM': CSQuantiles(LAST('NonOperatingNPTTM'), groups='sw1'),
    'TProfitTTM': CSQuantiles(LAST('TProfitTTM'), groups='sw1'),
    'NetProfitTTM': CSQuantiles(LAST('NetProfitTTM'), groups='sw1'),
    'NetProfitAPTTM': CSQuantiles(LAST('NetProfitAPTTM'), groups='sw1'),
    'SaleServiceRenderCashTTM': CSQuantiles(LAST('SaleServiceRenderCashTTM'), groups='sw1'),
    'NetOperateCFTTM': CSQuantiles(LAST('NetOperateCFTTM'), groups='sw1'),
    'NetInvestCFTTM': CSQuantiles(LAST('NetInvestCFTTM'), groups='sw1'),
    'NetFinanceCFTTM': CSQuantiles(LAST('NetFinanceCFTTM'), groups='sw1'),
    'GrossProfit': CSQuantiles(LAST('GrossProfit'), groups='sw1'),
    'Beta252': CSQuantiles(LAST('Beta252'), groups='sw1'),
    'RSTR504': CSQuantiles(LAST('RSTR504'), groups='sw1'),
    'EPIBS': CSQuantiles(LAST('EPIBS'), groups='sw1'),
    'CETOP': CSQuantiles(LAST('CETOP'), groups='sw1'),
    'DASTD': CSQuantiles(LAST('DASTD'), groups='sw1'),
    'CmraCNE5': CSQuantiles(LAST('CmraCNE5'), groups='sw1'),
    'HsigmaCNE5': CSQuantiles(LAST('HsigmaCNE5'), groups='sw1'),
    'SGRO': CSQuantiles(LAST('SGRO'), groups='sw1'),
    'EgibsLong': CSQuantiles(LAST('EgibsLong'), groups='sw1'),
    'STOM': CSQuantiles(LAST('STOM'), groups='sw1'),
    'STOQ': CSQuantiles(LAST('STOQ'), groups='sw1'),
    'STOA': CSQuantiles(LAST('STOA'), groups='sw1'),
    'NLSIZE': CSQuantiles(LAST('NLSIZE'), groups='sw1')}

alpha_factor_store = {
    'alpha_1': LAST('alpha_1'), 'alpha_2': LAST('alpha_2'), 'alpha_3': LAST('alpha_3'),
    'alpha_4': LAST('alpha_4'), 'alpha_5': LAST('alpha_5'), 'alpha_6': LAST('alpha_6'),
    'alpha_7': LAST('alpha_7'), 'alpha_8': LAST('alpha_8'), 'alpha_9': LAST('alpha_9'),
    'alpha_10': LAST('alpha_10'), 'alpha_11': LAST('alpha_11'), 'alpha_12': LAST('alpha_12'),
    'alpha_13': LAST('alpha_13'), 'alpha_14': LAST('alpha_14'), 'alpha_15': LAST('alpha_15'),
    'alpha_16': LAST('alpha_16'), 'alpha_17': LAST('alpha_17'), 'alpha_18': LAST('alpha_18'),
    'alpha_19': LAST('alpha_19'), 'alpha_20': LAST('alpha_20'), 'alpha_21': LAST('alpha_21'),
    'alpha_22': LAST('alpha_22'), 'alpha_23': LAST('alpha_23'), 'alpha_24': LAST('alpha_24'),
    'alpha_25': LAST('alpha_25'), 'alpha_26': LAST('alpha_26'), 'alpha_27': LAST('alpha_27'),
    'alpha_28': LAST('alpha_28'), 'alpha_29': LAST('alpha_29'), 'alpha_30': LAST('alpha_30'),
    'alpha_31': LAST('alpha_31'), 'alpha_32': LAST('alpha_32'), 'alpha_33': LAST('alpha_33'),
    'alpha_34': LAST('alpha_34'), 'alpha_35': LAST('alpha_35'), 'alpha_36': LAST('alpha_36'),
    'alpha_37': LAST('alpha_37'), 'alpha_38': LAST('alpha_38'), 'alpha_39': LAST('alpha_39'),
    'alpha_40': LAST('alpha_40'), 'alpha_41': LAST('alpha_41'), 'alpha_42': LAST('alpha_42'),
    'alpha_43': LAST('alpha_43'), 'alpha_44': LAST('alpha_44'), 'alpha_45': LAST('alpha_45'),
    'alpha_46': LAST('alpha_46'), 'alpha_47': LAST('alpha_47'), 'alpha_48': LAST('alpha_48'),
    'alpha_49': LAST('alpha_49'), 'alpha_50': LAST('alpha_50'), 'alpha_51': LAST('alpha_51'),
    'alpha_52': LAST('alpha_52'), 'alpha_53': LAST('alpha_53'), 'alpha_54': LAST('alpha_54'),
    'alpha_55': LAST('alpha_55'), 'alpha_56': LAST('alpha_56'), 'alpha_57': LAST('alpha_57'),
    'alpha_58': LAST('alpha_58'), 'alpha_59': LAST('alpha_59'), 'alpha_60': LAST('alpha_60'),
    'alpha_61': LAST('alpha_61'), 'alpha_62': LAST('alpha_62'), 'alpha_63': LAST('alpha_63'),
    'alpha_64': LAST('alpha_64'), 'alpha_65': LAST('alpha_65'), 'alpha_66': LAST('alpha_66'),
    'alpha_67': LAST('alpha_67'), 'alpha_68': LAST('alpha_68'), 'alpha_69': LAST('alpha_69'),
    'alpha_70': LAST('alpha_70'), 'alpha_71': LAST('alpha_71'), 'alpha_72': LAST('alpha_72'),
    'alpha_73': LAST('alpha_73'), 'alpha_74': LAST('alpha_74'), 'alpha_75': LAST('alpha_75'),
    'alpha_76': LAST('alpha_76'), 'alpha_77': LAST('alpha_77'), 'alpha_78': LAST('alpha_78'),
    'alpha_79': LAST('alpha_79'), 'alpha_80': LAST('alpha_80'), 'alpha_81': LAST('alpha_81'),
    'alpha_82': LAST('alpha_82'), 'alpha_83': LAST('alpha_83'), 'alpha_84': LAST('alpha_84'),
    'alpha_85': LAST('alpha_85'), 'alpha_86': LAST('alpha_86'), 'alpha_87': LAST('alpha_87'),
    'alpha_88': LAST('alpha_88'), 'alpha_89': LAST('alpha_89'), 'alpha_90': LAST('alpha_90'),
    'alpha_91': LAST('alpha_91'), 'alpha_92': LAST('alpha_92'), 'alpha_93': LAST('alpha_93'),
    'alpha_94': LAST('alpha_94'), 'alpha_95': LAST('alpha_95'), 'alpha_96': LAST('alpha_96'),
    'alpha_97': LAST('alpha_97'), 'alpha_98': LAST('alpha_98'), 'alpha_99': LAST('alpha_99'),
    'alpha_100': LAST('alpha_100'), 'alpha_101': LAST('alpha_101'), 'alpha_102': LAST('alpha_102'),
    'alpha_103': LAST('alpha_103'), 'alpha_104': LAST('alpha_104'), 'alpha_105': LAST('alpha_105'),
    'alpha_106': LAST('alpha_106'), 'alpha_107': LAST('alpha_107'), 'alpha_108': LAST('alpha_108'),
    'alpha_109': LAST('alpha_109'), 'alpha_110': LAST('alpha_110'), 'alpha_111': LAST('alpha_111'),
    'alpha_112': LAST('alpha_113'), 'alpha_113': LAST('alpha_113'), 'alpha_114': LAST('alpha_114'),
    'alpha_115': LAST('alpha_116'), 'alpha_116': LAST('alpha_116'), 'alpha_117': LAST('alpha_117'),
    'alpha_118': LAST('alpha_118'), 'alpha_119': LAST('alpha_119'), 'alpha_120': LAST('alpha_120'),
    'alpha_121': LAST('alpha_121'), 'alpha_122': LAST('alpha_122'), 'alpha_123': LAST('alpha_123'),
    'alpha_124': LAST('alpha_124'), 'alpha_125': LAST('alpha_125'), 'alpha_126': LAST('alpha_126'),
    'alpha_127': LAST('alpha_127'), 'alpha_128': LAST('alpha_128'), 'alpha_129': LAST('alpha_129'),
    'alpha_130': LAST('alpha_130'), 'alpha_131': LAST('alpha_131'), 'alpha_132': LAST('alpha_132'),
    'alpha_133': LAST('alpha_133'), 'alpha_134': LAST('alpha_134'), 'alpha_135': LAST('alpha_135'),
    'alpha_136': LAST('alpha_136'), 'alpha_137': LAST('alpha_137'), 'alpha_138': LAST('alpha_138'),
    'alpha_139': LAST('alpha_139'), 'alpha_140': LAST('alpha_140'), 'alpha_141': LAST('alpha_141'),
    'alpha_142': LAST('alpha_142'), 'alpha_143': LAST('alpha_143'), 'alpha_144': LAST('alpha_144'),
    'alpha_145': LAST('alpha_145'), 'alpha_146': LAST('alpha_146'), 'alpha_147': LAST('alpha_147'),
    'alpha_148': LAST('alpha_148'), 'alpha_149': LAST('alpha_149'), 'alpha_150': LAST('alpha_150'),
    'alpha_151': LAST('alpha_151'), 'alpha_152': LAST('alpha_152'), 'alpha_153': LAST('alpha_153'),
    'alpha_154': LAST('alpha_154'), 'alpha_155': LAST('alpha_155'), 'alpha_156': LAST('alpha_156'),
    'alpha_157': LAST('alpha_157'), 'alpha_158': LAST('alpha_158'), 'alpha_159': LAST('alpha_159'),
    'alpha_160': LAST('alpha_160'), 'alpha_161': LAST('alpha_161'), 'alpha_162': LAST('alpha_162'),
    'alpha_163': LAST('alpha_163'), 'alpha_164': LAST('alpha_164'), 'alpha_165': LAST('alpha_165'),
    'alpha_166': LAST('alpha_166'), 'alpha_167': LAST('alpha_167'), 'alpha_168': LAST('alpha_168'),
    'alpha_169': LAST('alpha_169'), 'alpha_170': LAST('alpha_170'), 'alpha_171': LAST('alpha_171'),
    'alpha_172': LAST('alpha_172'), 'alpha_173': LAST('alpha_173'), 'alpha_174': LAST('alpha_174'),
    'alpha_175': LAST('alpha_175'), 'alpha_176': LAST('alpha_176'), 'alpha_177': LAST('alpha_177'),
    'alpha_178': LAST('alpha_178'), 'alpha_179': LAST('alpha_179'), 'alpha_180': LAST('alpha_180'),
    'alpha_181': LAST('alpha_181'), 'alpha_182': LAST('alpha_182'), 'alpha_183': LAST('alpha_183'),
    'alpha_184': LAST('alpha_184'), 'alpha_185': LAST('alpha_185'), 'alpha_186': LAST('alpha_186'),
    'alpha_187': LAST('alpha_187'), 'alpha_188': LAST('alpha_188'), 'alpha_189': LAST('alpha_189'),
    'alpha_190': LAST('alpha_190'), 'alpha_191': LAST('alpha_191')
}

# 提取Uqer因子
basic_factor_org = engine.fetch_factor_range(universe, basic_factor_store, dates=ref_dates)
logging.info('basic_factor_org loading success')
# basic_factor_orgl = basic_factor_org.set_index(['trade_date', 'code'])
# 提取alpha191因子
alpha191_factor_org = engine.fetch_factor_range(universe, alpha_factor_store, dates=ref_dates, used_factor_tables=[Alpha191])
logging.info('alpha191_factor_org loading success')

# alpha191_factor_orgl = alpha191_factor_org.set_index(['trade_date', 'code'])

# 合并所有的因子
factor_data_org = pd.merge(basic_factor_org, alpha191_factor_org, on=['trade_date', 'code'], how='inner')

# 获取
industry = engine.fetch_industry_range(universe, dates=ref_dates)
factor_data = pd.merge(factor_data_org, industry, on=['trade_date', 'code']).fillna(0.)
risk_total = engine.fetch_risk_model_range(universe, dates=ref_dates)[1]

return_data = engine.fetch_dx_return_range(universe, dates=ref_dates, horizon=horizon, offset=0,
                                           benchmark=benchmark_code)

benchmark_total = engine.fetch_benchmark_range(dates=ref_dates, benchmark=benchmark_code)
industry_total = engine.fetch_industry_matrix_range(universe, dates=ref_dates, category=industry_name,
                                                    level=industry_level)
logging.info('industry_total loading success')

train_data = pd.merge(factor_data, return_data, on=['trade_date', 'code']).dropna()

# Constraintes settings

industry_names = industry_list(industry_name, industry_level)
constraint_risk = ['EARNYILD', 'LIQUIDTY', 'GROWTH', 'SIZE', 'SIZENL', 'BETA', 'MOMENTUM'] + industry_names
# constraint_risk = ['EARNYILD', 'LIQUIDTY', 'GROWTH', 'SIZE', 'BETA', 'MOMENTUM'] + industry_names

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
# for name in total_risk_names:
#     if name == 'benchmark':
#         b_type.append(BoundaryType.RELATIVE)
#         l_val.append(0.0)
#         u_val.append(1.0)
#     elif name == 'total':
#         b_type.append(BoundaryType.ABSOLUTE)
#         l_val.append(.0)
#         u_val.append(.0)
#     else:
#         b_type.append(BoundaryType.ABSOLUTE)
#         l_val.append(-1.005)
#         u_val.append(1.005)

bounds = create_box_bounds(total_risk_names, b_type, l_val, u_val)

features = [
    'AccountsPayablesTDays', 'AccountsPayablesTRate', 'AdminiExpenseRate', 'ARTDays',
    'ARTRate', 'ASSI', 'BLEV', 'BondsPayableToAsset', 'CashRateOfSales', 'CashToCurrentLiability',
    'CMRA', 'CTOP', 'CTP5', 'CurrentAssetsRatio', 'CurrentAssetsTRate', 'CurrentRatio', 'DAVOL10',
    'DAVOL20', 'DAVOL5', 'DDNBT', 'DDNCR', 'DDNSR', 'DebtEquityRatio', 'DebtsAssetRatio', 'DHILO',
    'DilutedEPS', 'DVRAT', 'EBITToTOR', 'EGRO', 'EMA10', 'EMA120', 'EMA20', 'EMA5', 'EMA60', 'EPS',
    'EquityFixedAssetRatio', 'EquityToAsset', 'EquityTRate', 'ETOP', 'ETP5', 'FinancialExpenseRate',
    'FinancingCashGrowRate', 'FixAssetRatio', 'FixedAssetsTRate', 'GrossIncomeRatio', 'HBETA',
    'HSIGMA', 'IntangibleAssetRatio', 'InventoryTDays', 'InventoryTRate', 'InvestCashGrowRate',
    'LCAP', 'LFLO', 'LongDebtToAsset', 'LongDebtToWorkingCapital', 'LongTermDebtToAsset',
    'MA10', 'MA120', 'MA20', 'MA5', 'MA60', 'MAWVAD', 'MFI', 'MLEV', 'NetAssetGrowRate',
    'NetProfitGrowRate', 'NetProfitRatio', 'NOCFToOperatingNI', 'NonCurrentAssetsRatio',
    'NPParentCompanyGrowRate', 'NPToTOR', 'OperatingExpenseRate', 'OperatingProfitGrowRate',
    'OperatingProfitRatio', 'OperatingProfitToTOR', 'OperatingRevenueGrowRate', 'OperCashGrowRate',
    'OperCashInToCurrentLiability', 'PB', 'PCF', 'PE', 'PS', 'PSY', 'QuickRatio', 'REVS10',
    'REVS20', 'REVS5', 'ROA', 'ROA5', 'ROE', 'ROE5', 'RSI', 'RSTR12', 'RSTR24', 'SalesCostRatio',
    'SaleServiceCashToOR', 'SUE', 'TaxRatio', 'TOBT', 'TotalAssetGrowRate', 'TotalAssetsTRate',
    'TotalProfitCostRatio', 'TotalProfitGrowRate', 'VOL10', 'VOL120', 'VOL20', 'VOL240', 'VOL5',
    'VOL60', 'WVAD', 'REC', 'DAREC', 'GREC', 'FY12P', 'DAREV', 'GREV', 'SFY12P', 'DASREV', 'GSREV',
    'FEARNG', 'FSALESG', 'TA2EV', 'CFO2EV', 'ACCA', 'DEGM', 'SUOI', 'EARNMOM', 'FiftyTwoWeekHigh',
    'Volatility', 'Skewness', 'ILLIQUIDITY', 'BackwardADJ', 'MACD', 'ADTM', 'ATR14', 'ATR6', 'BIAS10',
    'BIAS20', 'BIAS5', 'BIAS60', 'BollDown', 'BollUp', 'CCI10', 'CCI20', 'CCI5', 'CCI88', 'KDJ_K', 'KDJ_D',
    'KDJ_J', 'ROC6', 'ROC20', 'SBM', 'STM', 'UpRVI', 'DownRVI', 'RVI', 'SRMI', 'ChandeSD', 'ChandeSU',
    'CMO', 'DBCD', 'ARC', 'OBV', 'OBV6', 'OBV20', 'TVMA20', 'TVMA6', 'TVSTD20', 'TVSTD6', 'VDEA', 'VDIFF',
    'VEMA10', 'VEMA12', 'VEMA26', 'VEMA5', 'VMACD', 'VR', 'VROC12', 'VROC6', 'VSTD10', 'VSTD20', 'KlingerOscillator',
    'MoneyFlow20', 'AD', 'AD20', 'AD6', 'CoppockCurve', 'ASI', 'ChaikinOscillator', 'ChaikinVolatility',
    'EMV14', 'EMV6', 'plusDI', 'minusDI', 'ADX', 'ADXR', 'Aroon', 'AroonDown', 'AroonUp', 'DEA', 'DIFF', 'DDI', 'DIZ',
    'DIF', 'MTM', 'MTMMA', 'PVT', 'PVT6', 'PVT12', 'TRIX5', 'TRIX10', 'UOS', 'MA10RegressCoeff12', 'MA10RegressCoeff6',
    'PLRC6', 'PLRC12', 'SwingIndex', 'Ulcer10', 'Ulcer5', 'Hurst', 'ACD6', 'ACD20', 'EMA12', 'EMA26', 'APBMA',
    'BBI', 'BBIC', 'TEMA10', 'TEMA5', 'MA10Close', 'AR', 'BR', 'ARBR', 'CR20', 'MassIndex', 'BearPower', 'BullPower',
    'Elder', 'NVI', 'PVI', 'RC12', 'RC24', 'JDQS20', 'Variance20', 'Variance60', 'Variance120', 'Kurtosis20',
    'Kurtosis60', 'Kurtosis120', 'Alpha20', 'Alpha60', 'Alpha120', 'Beta20', 'Beta60', 'Beta120', 'SharpeRatio20',
    'SharpeRatio60', 'SharpeRatio120', 'TreynorRatio20', 'TreynorRatio60', 'TreynorRatio120', 'InformationRatio20',
    'InformationRatio60', 'InformationRatio120', 'GainVariance20', 'GainVariance60', 'GainVariance120',
    'LossVariance20',
    'LossVariance60', 'LossVariance120', 'GainLossVarianceRatio20', 'GainLossVarianceRatio60',
    'GainLossVarianceRatio120',
    'RealizedVolatility', 'REVS60', 'REVS120', 'REVS250', 'REVS750', 'REVS5m20', 'REVS5m60', 'REVS5Indu1',
    'REVS20Indu1',
    'Volumn1M', 'Volumn3M', 'Price1M', 'Price3M', 'Price1Y', 'Rank1M', 'CashDividendCover', 'DividendCover',
    'DividendPaidRatio', 'RetainedEarningRatio', 'CashEquivalentPS', 'DividendPS', 'EPSTTM', 'NetAssetPS', 'TORPS',
    'TORPSLatest', 'OperatingRevenuePS', 'OperatingRevenuePSLatest', 'OperatingProfitPS', 'OperatingProfitPSLatest',
    'CapitalSurplusFundPS', 'SurplusReserveFundPS', 'UndividedProfitPS', 'RetainedEarningsPS', 'OperCashFlowPS',
    'CashFlowPS', 'NetNonOIToTP', 'NetNonOIToTPLatest', 'PeriodCostsRate', 'InterestCover', 'NetProfitGrowRate3Y',
    'NetProfitGrowRate5Y', 'OperatingRevenueGrowRate3Y', 'OperatingRevenueGrowRate5Y', 'NetCashFlowGrowRate',
    'NetProfitCashCover', 'OperCashInToAsset', 'CashConversionCycle', 'OperatingCycle', 'PEG3Y', 'PEG5Y', 'PEIndu',
    'PBIndu', 'PSIndu', 'PCFIndu', 'PEHist20', 'PEHist60', 'PEHist120', 'PEHist250', 'StaticPE', 'ForwardPE',
    'EnterpriseFCFPS', 'ShareholderFCFPS', 'ROEDiluted', 'ROEAvg', 'ROEWeighted', 'ROECut', 'ROECutWeighted',
    'ROIC', 'ROAEBIT', 'ROAEBITTTM', 'OperatingNIToTP', 'OperatingNIToTPLatest', 'InvestRAssociatesToTP',
    'InvestRAssociatesToTPLatest',
    'NPCutToNP', 'SuperQuickRatio', 'TSEPToInterestBearDebt', 'DebtTangibleEquityRatio', 'TangibleAToInteBearDebt',
    'TangibleAToNetDebt', 'NOCFToTLiability', 'NOCFToInterestBearDebt', 'NOCFToNetDebt', 'TSEPToTotalCapital',
    'InteBearDebtToTotalCapital', 'NPParentCompanyCutYOY', 'SalesServiceCashToORLatest', 'CashRateOfSalesLatest',
    'NOCFToOperatingNILatest', 'TotalAssets', 'MktValue', 'NegMktValue', 'TEAP', 'NIAP', 'TotalFixedAssets',
    'IntFreeCL', 'IntFreeNCL', 'IntCL', 'IntDebt', 'NetDebt', 'NetTangibleAssets', 'WorkingCapital',
    'NetWorkingCapital',
    'TotalPaidinCapital', 'RetainedEarnings', 'OperateNetIncome', 'ValueChgProfit', 'NetIntExpense', 'EBIT',
    'EBITDA', 'EBIAT', 'NRProfitLoss', 'NIAPCut', 'FCFF', 'FCFE', 'DA', 'TRevenueTTM', 'TCostTTM', 'RevenueTTM',
    'CostTTM', 'GrossProfitTTM', 'SalesExpenseTTM', 'AdminExpenseTTM', 'FinanExpenseTTM', 'AssetImpairLossTTM',
    'NPFromOperatingTTM', 'NPFromValueChgTTM', 'OperateProfitTTM', 'NonOperatingNPTTM', 'TProfitTTM', 'NetProfitTTM',
    'NetProfitAPTTM', 'SaleServiceRenderCashTTM', 'NetOperateCFTTM', 'NetInvestCFTTM', 'NetFinanceCFTTM', 'GrossProfit',
    'Beta252', 'RSTR504', 'EPIBS', 'CETOP', 'DASTD', 'CmraCNE5', 'HsigmaCNE5', 'SGRO', 'EgibsLong', 'STOM', 'STOQ',
    'STOA', 'NLSIZE']

alpha_features = [
    'alpha_1', 'alpha_2', 'alpha_3', 'alpha_4', 'alpha_5', 'alpha_6', 'alpha_7', 'alpha_8', 'alpha_9', 'alpha_10',
    'alpha_11', 'alpha_12', 'alpha_13', 'alpha_14', 'alpha_15', 'alpha_16', 'alpha_17', 'alpha_18', 'alpha_19',
    'alpha_20',
    'alpha_21', 'alpha_22', 'alpha_23', 'alpha_24', 'alpha_25', 'alpha_26', 'alpha_27', 'alpha_28', 'alpha_29',
    'alpha_30',
    'alpha_31', 'alpha_32', 'alpha_33', 'alpha_34', 'alpha_35', 'alpha_36', 'alpha_37', 'alpha_38', 'alpha_39',
    'alpha_40',
    'alpha_41', 'alpha_42', 'alpha_43', 'alpha_44', 'alpha_45', 'alpha_46', 'alpha_47', 'alpha_48', 'alpha_49',
    'alpha_50',
    'alpha_51', 'alpha_52', 'alpha_53', 'alpha_54', 'alpha_55', 'alpha_56', 'alpha_57', 'alpha_58', 'alpha_59',
    'alpha_60',
    'alpha_61', 'alpha_62', 'alpha_63', 'alpha_64', 'alpha_65', 'alpha_66', 'alpha_67', 'alpha_68', 'alpha_69',
    'alpha_70',
    'alpha_71', 'alpha_72', 'alpha_73', 'alpha_74', 'alpha_75', 'alpha_76', 'alpha_77', 'alpha_78', 'alpha_79',
    'alpha_80',
    'alpha_81', 'alpha_82', 'alpha_83', 'alpha_84', 'alpha_85', 'alpha_86', 'alpha_87', 'alpha_88', 'alpha_89',
    'alpha_90',
    'alpha_91', 'alpha_92', 'alpha_93', 'alpha_94', 'alpha_95', 'alpha_96', 'alpha_97', 'alpha_98', 'alpha_99',
    'alpha_100',
    'alpha_101', 'alpha_102', 'alpha_103', 'alpha_104', 'alpha_105', 'alpha_106', 'alpha_107', 'alpha_108', 'alpha_109',
    'alpha_110',
    'alpha_111', 'alpha_112', 'alpha_113', 'alpha_114', 'alpha_115', 'alpha_116', 'alpha_117', 'alpha_118', 'alpha_119',
    'alpha_120',
    'alpha_121', 'alpha_122', 'alpha_123', 'alpha_124', 'alpha_125', 'alpha_126', 'alpha_127', 'alpha_128', 'alpha_129',
    'alpha_130',
    'alpha_131', 'alpha_132', 'alpha_133', 'alpha_134', 'alpha_135', 'alpha_136', 'alpha_137', 'alpha_138', 'alpha_139',
    'alpha_140',
    'alpha_141', 'alpha_142', 'alpha_143', 'alpha_144', 'alpha_145', 'alpha_146', 'alpha_147', 'alpha_148', 'alpha_149',
    'alpha_150',
    'alpha_151', 'alpha_152', 'alpha_153', 'alpha_154', 'alpha_155', 'alpha_156', 'alpha_157', 'alpha_158', 'alpha_159',
    'alpha_160',
    'alpha_171', 'alpha_172', 'alpha_173', 'alpha_174', 'alpha_175', 'alpha_176', 'alpha_177', 'alpha_178', 'alpha_179',
    'alpha_180',
    'alpha_181', 'alpha_182', 'alpha_183', 'alpha_184', 'alpha_185', 'alpha_186', 'alpha_187', 'alpha_188', 'alpha_189',
    'alpha_190',
    'alpha_191'
]

features.extend(alpha_features)

label = ['dx']

from datetime import datetime, timedelta
from m1_xgb import *
from src.conf.configuration import regress_conf
import xgboost as xgb
import gc


def create_scenario():
    weight_gap = 1
    transact_cost = 0.003
    GPU_device = True

    executor = NaiveExecutor()
    leverags = []
    trade_dates = []
    current_pos = pd.DataFrame()
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
        print(xgb_model.get_params)

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

        # 股票过滤, 组合优化之前过滤掉
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
        executor.set_current(current_pos)
        net_rets.append(np.log(1. + ret - transact_cost * turn_over))
        alpha_logger.info('len_net_rets: {}, len_trade_dates: {}'.format(len(net_rets), len(trade_dates)))

        alpha_logger.info('{} is finished'.format(ref_date))

    # ret_df = pd.DataFrame({'xgb_regress': rets}, index=trade_dates)
    ret_df = pd.DataFrame({'xgb_regress': rets, 'net_xgb_regress': net_rets}, index=trade_dates)
    ret_df.loc[advanceDateByCalendar('china.sse', ref_dates[-1], freq).strftime('%Y-%m-%d')] = 0.
    ret_df = ret_df.shift(1)
    ret_df.iloc[0] = 0.
    return ret_df, tune_record, rets, net_rets


ret_df, tune_record, rets, net_rets = create_scenario()

# 调仓记录保存
import sqlite3

con = sqlite3.connect('./tune_record.db')
tune_record.to_sql('tune_record', con=con, if_exists='append', index=False)
