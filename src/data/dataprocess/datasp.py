#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: datasp.py
@time: 2019-06-10 17:30
"""

import pandas as pd


def daily_price():


    pass


if __name__ == '__main__':
    zz500_df = pd.read_csv('zz500.csv')
    print(zz500_df[['code', 'trade_date']])