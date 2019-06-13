#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: datasp.py
@time: 2019-06-10 17:30
"""
import pickle
import pysnooper
import pandas as pd
import dask.dataframe as ddf

from src.utils import data_source

engine_postgre = data_source.GetDataEngine("ALPHA_FACTOR")


@pysnooper.snoop()
def daily_price_read(sheet_name):
    """
    读取股票名称和股票代码
    :param sheet_name:
    :return:
    """
    sql = "SELECT * FROM public.%s limit 50000" % sheet_name
    resultdf = pd.read_sql(sql, engine_postgre)
    resultdf['trade_date'] = resultdf['trade_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    resultdf['code'] = resultdf[['code', 'exchangeCD']].apply(lambda x: str(x[0]).zfill(6) + '.'+x[1], axis=1)
    return resultdf


@pysnooper.snoop()
def factor_read(sheet_name):
    sql = "SELECT count(*) FROM public.%s" % sheet_name
    resultdf = pd.read_sql(sql, engine_postgre)
    return resultdf


def merged(zz500_df, stock_df, trade_date):
    """
    投资域成分股筛选， 按照投资域内的成分股，筛选出对应股票的基础行情
    :param zz500_df: 投资域成分股列表
    :param stock_df: 股票行情数据
    :param trade_date:
    :return:
    """
    tmp1 = zz500_df[zz500_df['trade_date'] == trade_date]
    tmp2 = stock_df[stock_df['trade_date'] == trade_date]
    tmp3 = pd.merge(tmp1, tmp2.drop(['trade_date'], axis=1), on='code')
    return tmp3


if __name__ == '__main__':
    # 读取中证500成分股
    zz500_df = pd.read_csv('../dataset/zz500_all.csv')
    zz500_df = zz500_df[['code', 'trade_date']]
    # trade_date_list = list(set(zz500_df['trade_date'].values))
    # print(trade_date_list)

    # 读取全市场行情数据
    stock_df = daily_price_read('market')
    stock_df = stock_df[['trade_date', 'code', 'secShortName', 'closePrice']]

    trade_date_list = list(set(stock_df['trade_date'].values))
    print(trade_date_list)

    result = pd.DataFrame()
    for date in trade_date_list:
        print(date)
        tt = merged(zz500_df, stock_df, date)
        print(len(tt))
        result = result.append(tt)

    # print(result)
    stock_grouped = result.groupby('code')

    result_df = pd.DataFrame()
    for i in stock_grouped:
        code = i[0]
        data_df = i[1]
        print(code)
        tmp_df = data_df.sort_values(by=['trade_date'])
        tmp_df['return_2'] = tmp_df['closePrice'].div(tmp_df['closePrice'].shift(2)) - 1
        tmp_df['return_3'] = tmp_df['closePrice'].div(tmp_df['closePrice'].shift(3)) - 1
        result_df = result_df.append(tmp_df[['trade_date', 'code', 'closePrice', 'return_2', 'return_3']])

    print(result_df)

