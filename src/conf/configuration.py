#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: configuration.py
@time: 2019-03-04 10:13
"""


class RegressionConfig(object):
    def __init__(self):
        self.params = {}
        self.max_round = None
        self.cv_folds = None
        self.early_stop_round = None
        self.seed = None
        self.save_model_path = None

    def xgb_config_r(self):
        # 回归
        self.params = {
            'booster': 'dart',
            'learning_rate': 0.01,
            'max_depth': 27,
            'eta': 1,
            'silent': 1,
            'objective': 'reg:linear',
            'eval_metric': 'rmse'}
        self.max_round = 500
        self.cv_folds = 10
        self.early_stop_round = 500
        self.seed = 3
        self.save_model_path = '../model/xgb/'


regress_conf = RegressionConfig()

