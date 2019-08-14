#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: read_data.py
@time: 2019-06-11 16:35
"""
import dask.dataframe as ddf
import pandas as pd
import pysnooper
from src.utils import data_source
from sqlalchemy import select


engine_postgre = data_source.GetDataEngine("ALPHA_FACTOR")

@pysnooper.snoop()
def factor_read(sheet_name):
    sql = "SELECT * FROM public.%s" % sheet_name
    resultdf = pd.read_sql(sql, engine_postgre)
    return resultdf


if __name__ == '__main__':
    pd.set_option('display.max_rows', None, 'display.max_columns', None, "display.max_colwidth", 1000, 'display.width',
                  1000)

    # zz500_df = pd.read_csv('../dataset/return_label.csv')
    # zz500_df = zz500_df.drop(['Unnamed: 0'], axis=1)
    #
    # print(len(zz500_df))
    # print(zz500_df[zz500_df['code'] == '000006.XSHG'])
    # factor_df = factor_read('alpha191')
    factor_df = pd.read_csv('../dataset/training_sample.csv')
    print(factor_df.head())




