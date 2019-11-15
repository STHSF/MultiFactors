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
        self.ts_cv_folds = None

    def xgb_config_r(self):
        # 回归
        self.params = {
            'booster': 'dart',
            'learning_rate': 0.01,
            'max_depth': 5,
            'eta': 1,
            'silent': 1,
            'objective': 'reg:linear',
            'eval_metric': 'rmse'}
        self.max_round = 500
        self.cv_folds = 10
        self.early_stop_round = 500
        self.seed = 3
        self.save_model_path = '../model/xgb/'

    def lgb_config_r(self):
        # 将参数写成字典下形式
        self.params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'regression',  # 目标函数
            'metric': {'l2', 'auc'},  # 评估函数
            'num_leaves': 31,  # 叶子节点数
            'learning_rate': 0.05,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.8,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }


class ClassificationConfig(object):
    def __init__(self):
        self.params = {}
        self.max_round = None
        self.cv_folds = None
        self.early_stop_round = None
        self.seed = None
        self.save_model_path = None
        self.ts_cv_folds = None

    def lgb_config_c(self):
        self.params = {'task': 'train',
                       'boosting_type': 'gbdt',
                       'objective': 'multiclass',
                       'num_class': 3,
                       'metric': ['multi_error', 'multi_logloss'],
                       'metric_freq': 1,
                       # 'max_bin': 255,
                       'num_leaves': 31,
                       'max_depth': 20,
                       'learning_rate': 0.05,
                       'feature_fraction': 0.9,
                       'bagging_fraction': 0.95,
                       'bagging_freq': 5}

        self.max_round = 100
        self.cv_folds = 4
        self.early_stop_round = 30
        self.seed = 3
        self.save_model_path = 'model/lgb/lgb.txt'


regress_conf = RegressionConfig()
classify_conf = ClassificationConfig()

