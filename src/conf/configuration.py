#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: configuration.py
@time: 2019-03-04 10:13
"""


class XGBConfig:
    def __init__(self):
        self.params = {}
        self.max_round = None
        self.early_stop_round = None
        self.ts_cv_folds = None
        self.cv_folds = None
        self.cv_seed = None
        self.save_model_path = None

    def xgb_config_r(self):
        # 回归
        self.params = {'booster': 'gbtree',
                       'objective': 'reg:linear',
                       'eval_metric': ['rmse', 'logloss'],
                       'nthread': 4,  # 运行的线程数，-1所有线程
                       'silent': 1,
                       'learning_rate': 0.01,
                       'max_depth': 5,
                       'eta': 0.03,
                       'alpha': 0,  # L1正则，树的深度过大时，可以适大该参数
                       'lambda': 0,  # L2正则
                       'subsample': 0.7,  # 随机采样的比率，通俗理解就是选多少样本做为训练集，选择小于1的比例可以减少方差，即防止过拟合
                       'colsample_bytree': 0.5,  # 这里是选择多少列作为训练集，具体的理解就是选择多少特征
                       'min_child_weight': 3,  # 决定最小叶子节点样本权重和。当它的值较大时，可以避免模型学习到局部的特殊样本。但如果这个值过高，会导致欠拟合
                       'seed': 2019,  # 这个随机指定一个常数，防止每次结果不一致
                       }
        self.max_round = 1000
        self.early_stop_round = 100
        self.cv_folds = None
        self.cv_seed = 2019
        self.save_model_path = '../bst_model/xgb/'

    def xgb_config_c(self):
        self.params = {'objective': 'multi:softmax',  # 目标函数
                       'num_class': 3,  # 当是objective为'multi:softmax'时需要指定类别数量，eg:'num_class':33
                       'nthread': 4,  # 运行的线程数，-1所有线程
                       'silent': 0,
                       'learning_rate': 0.01,
                       'eta': 0.03,
                       'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子
                       "eval_metric": ["mlogloss", "merror"],  # 评价函数，如果该参数没有指定，缺省值是通过目标函数来做匹配，
                       'max_depth': 5,  #  树的深度，对结果影响较大，越深越容易过拟合
                       'alpha': 0,  # L1正则，树的深度过大时，可以适大该参数
                       'lambda': 0,  # L2正则
                       'subsample': 0.7,  # 随机采样的比率，通俗理解就是选多少样本做为训练集，选择小于1的比例可以减少方差，即防止过拟合
                       'colsample_bytree': 0.5,  # 这里是选择多少列作为训练集，具体的理解就是选择多少特征
                       'min_child_weight': 3,  # 决定最小叶子节点样本权重和。当它的值较大时，可以避免模型学习到局部的特殊样本。但如果这个值过高，会导致欠拟合
                       'seed': 2019,  # 这个随机指定一个常数，防止每次结果不一致
                       }

        self.max_round = 1000
        self.cv_folds = None
        self.early_stop_round = 100
        self.cv_seed = 2019


class LGBConfig:
    def __init__(self):
        self.params = {}
        self.max_round = None
        self.early_stop_round = None
        self.ts_cv_folds = None
        self.cv_folds = None
        self.cv_seed = None
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
                       'verbosity': -1,
                       # 'device': 'gpu',  # 默认使用集显
                       # 'gpu_platform_id': 1,  # 确定是使用集成显卡还是独立显卡，0代表独显，1代表独显
                       # 'gpu_device_id': 0  # id为0的独显
                       }

        self.max_round = 1000
        self.cv_folds = None
        self.early_stop_round = 100
        self.cv_seed = 2019
        self.save_model_path = 'bst_model/lgb/lgb.txt'

    def lgb_config_r(self):
        self.params = {
            'task': 'train',
            'boosting': 'gbdt',  # 设置提升类型
            'objective': 'regression',  # 目标函数
            'metric': {'l2', 'mean_squared_error'},  # 评估函数
            'num_leaves': 31,  # 叶子节点数
            'learning_rate': 0.05,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例 # 样本列采样
            'bagging_fraction': 0.8,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'lambda_l1': 0.90,  # L1 正则化
            'lambda_l2': 0.95,  # L2 正则化
            'bagging_seed': 100,  # 随机种子,light中默认为100
            'verbosity': -1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
        self.max_round = 1000
        self.early_stop_round = 100
        self.cv_folds = None
        self.cv_seed = 2019


xgb_conf = XGBConfig()
lgb_conf = LGBConfig()
