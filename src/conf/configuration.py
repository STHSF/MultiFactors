#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
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
            'rate_drop': 0.1,
            'objective': 'reg:linear',
            'eval_metric': ['rmse', 'logloss']}
        self.max_round = 800
        self.cv_folds = None
        self.early_stop_round = 100
        self.seed = 3
        self.save_model_path = '../bst_model/xgb/'

    def lgb_config_r(self):
        self.params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'regression',  # 目标函数
            'metric': {'l2', 'auc'},  # 评估函数
            'num_leaves': 31,  # 叶子节点数
            'learning_rate': 0.05,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例 # 样本列采样
            'bagging_fraction': 0.8,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'lambda_l1': 0.90,  # L1 正则化
            'lambda_l2': 0.95,  # L2 正则化
            'bagging_seed': 100,  # 随机种子,light中默认为100
            'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
        self.max_round = 500
        self.cv_folds = None
        self.early_stop_round = 10
        self.seed = 3


class ClassificationConfig(object):
    def __init__(self):
        self.params = {}
        self.max_round = None
        self.cv_folds = None
        self.early_stop_round = None
        self.seed = None
        self.save_model_path = None

    def lgb_config_c(self):
        self.params = {'task': 'train',
                       'boosting': 'gbdt',
                       'objective': 'multiclass',
                       'num_class': 3,
                       'metric': ['multi_error', 'multi_logloss'],
                       'max_bin': 63,  # 表示 feature 将存入的 bin 的最大数量
                       'metric_freq': 1,
                       'num_leaves': 31,  # 由于lightGBM是leaves_wise生长，官方说法是要num_leaves<=2^max_depth,超过此值容易过拟合
                       'max_depth': 6,  # 树的最大层数为7 ,可以选择一个适中的值，其实4-10都可以。但要注意它越大越容易出现过拟合
                       'learning_rate': 0.05,
                       'feature_fraction': 0.9,  # bagging_fraction相当于样本特征采样，使bagging运行更快的同时可以降拟合
                       'bagging_fraction': 0.95,  # 用来进行特征的子抽样，可以用来防止过拟合并提高训练速度[0.5, 0.6, 0.7,0.8,0.9]
                       'bagging_freq': 5,
                       'lambda_l1': 0.9,
                       'lambda_l2': 0.95,  # L2正则化系数
                       # 'device': 'gpu',  # 默认使用集显
                       # 'gpu_platform_id': 1,  # 确定是使用集成显卡还是独立显卡，0代表独显，1代表独显
                       # 'gpu_device_id': 0  # id为0的独显
                       }

        self.max_round = 10000
        self.cv_folds = None
        self.early_stop_round = 30
        self.seed = 3
        self.save_model_path = 'bst_model/lgb/lgb.txt'

regress_conf = RegressionConfig()
classify_conf = ClassificationConfig()
