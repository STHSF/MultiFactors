#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: bayes_opt.py
@time: 2019/11/26 4:23 下午
"""


import gc
import warnings
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
from bayes_opt import BayesianOptimization
from src.utils import log_util


log = log_util.Logger('BayesOptimizationXGBoost', level='info')


class BayesOptimizationXGB(object):
    """
    基于贝叶斯优化的XGBoost参数寻优过程
    注意不同的eval_metric使用的best_score不一样，需要自己调整。
    """

    def __init__(self, X_train, y_train, X_test=None, y_test=None, kfolds=None):
        """
        init
        :param X_train: train target
        :param y_train: train label
        :param X_test: test target
        :param y_test: test label
        :param kfolds: cv folds
        """
        self.BestScore = 1.  # best_score保存的时候需要注意选用的eval_metric，一般都是指标越小越好， 如果是auc，则是越大越好
        self.BestIter = 0.
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.folds = kfolds

    def xgb_cv(self, max_depth, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree):
        """
        XGBoost model with NonCrossValidation
        :param max_depth:
        :param gamma:
        :param min_child_weight:
        :param max_delta_step:
        :param subsample:
        :param colsample_bytree:
        :return:
        """

        data_train = xgb.DMatrix(self.X_train, label=self.y_train)
        paramt = {'eta': 0.1,
                  'objective': 'multi:softmax',
                  'num_class': 3,
                  'nthread': 4,
                  'silent': 0,
                  "eval_metric": ["mlogloss", "merror"],
                  'max_depth': int(max_depth),
                  'gamma': int(gamma),
                  'subsample': max(min(subsample, 1), 0),
                  'colsample_bytree': max(min(colsample_bytree, 1), 0),
                  'min_child_weight': int(min_child_weight),
                  'max_delta_step': int(max_delta_step),
                  'seed': 1001}

        xgbc = xgb.cv(
            paramt,
            data_train,
            num_boost_round=20000,
            stratified=True,
            nfold=self.folds,
            early_stopping_rounds=100,
            verbose_eval=True,
            show_stdv=True
        )
        val_score = xgbc['test-merror-mean'].iloc[-1]
        train_score = xgbc['train-merror-mean'].iloc[-1]
        log.logger.info(
            'Stopped after %d iterations with train-score = %f val-score = %f ( diff = %f ) train_-gini = %f '
            'val-gini = %f' % (len(xgbc),
                               train_score,
                               val_score,
                               (train_score - val_score),
                               (train_score * 2 - 1),
                               (val_score * 2 - 1)))

        if val_score < self.BestScore:
            # merror指标越小越好，使用AUC则是指标越大越好
            self.BestScore = val_score
            self.BestIter = len(xgbc)
        return (val_score * 2) - 1

    def xgb_no(self, max_depth, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree):
        """
        XGBoost model with NonCrossValidation
        :param max_depth:
        :param gamma:
        :param min_child_weight:
        :param max_delta_step:
        :param subsample:
        :param colsample_bytree:
        :return:
        """

        data_train = xgb.DMatrix(self.X_train, label=self.y_train)
        data_test = xgb.DMatrix(self.X_test, label=self.y_test)
        watchlist = [(data_test, 'eval'), (data_train, 'train')]
        params = {'objective': 'multi:softmax',
                  'num_class': 3,
                  'nthread': 4,
                  'silent': 0,
                  'eta': 0.1,
                  "eval_metric": ["mlogloss", "merror"],
                  'max_depth': int(max_depth),
                  'gamma': int(gamma),
                  'subsample': max(min(subsample, 1), 0),
                  'colsample_bytree': max(min(colsample_bytree, 1), 0),
                  'min_child_weight': int(min_child_weight),
                  'max_delta_step': int(max_delta_step),
                  'seed': 1001}

        best_model = xgb.train(params=params,
                               dtrain=data_train,
                               num_boost_round=20000,
                               evals=watchlist,
                               early_stopping_rounds=100)
        best_round = best_model.best_iteration
        best_score = best_model.best_score
        log.logger.info(' Stopped after %d iterations with train_-score = %f train_-gini = %f' %
                        (best_round, best_score, (best_score * 2 - 1)))
        if best_score < self.BestScore:
            # merror指标越小越好，使用AUC则是指标越大越好
            self.BestScore = best_score
            self.BestIter = best_round
        return (best_score * 2) - 1

    def bayesian_optimization(self, opt_parameters, gp_params=None):
        """
        BayesianOptimization
        :param opt_parameters: 待优化的模型参数
        :param gp_params: 贝叶斯优化模型的参数
        :return:
        """
        if gp_params is None:
            # gp_params = {"init_points": 10, "n_iter": 50, "acq": 'ucb', "xi": 0.0, "alpha": 1e-4, "kappa": 10}
            # gp_params = {"init_points": 2, "n_iter": 50, "acq": 'ei', "xi": 0.0, "alpha": 1e-4}
            gp_params = {"init_points": 2, "n_iter": 2, "acq": 'ei', "xi": 0.0, "alpha": 1e-4}

        if self.folds is None and self.X_test is not None and self.y_test is not None:
            BO = BayesianOptimization(self.xgb_no, opt_parameters)
        else:
            BO = BayesianOptimization(self.xgb_cv, opt_parameters)
        BO.maximize(**gp_params)
        return BO.max

    def train_opt(self, parameters, gp_params=None):
        """
        Train Optimization model
        :param parameters: 待优化的模型参数
        :param gp_params: 贝叶斯优化模型的参数
        :return:
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            best_solution = self.bayesian_optimization(parameters, gp_params)
        params_opt = best_solution["params"]
        log.logger.info('Best XGBOOST opt_parameters: %s' % params_opt)
        return params_opt


if __name__ == '__main__':

    # Classify Parameter Optimization Test
    log.logger.info('Classify Parameter Optimization Test')
    iris = load_iris()
    data = iris.data
    target = iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
    log.logger.info(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))

    # XGB_BO = BayesianOptimization(xgb_cv,
    #                               {'max_depth': (2, 12),
    #                                'gamma': (0.001, 10.0),
    #                                'min_child_weight': (0, 20),
    #                                'max_delta_step': (0, 10),
    #                                'subsample': (0.4, 1.0),
    #                                'colsample_bytree': (0.4, 1.0)
    #                               })
    #
    # XGB_BO = BayesianOptimization(xgb_no,
    #                               {'max_depth': (2, 12),
    #                                'gamma': (0.001, 10.0),
    #                                'min_child_weight': (0, 20),
    #                                'max_delta_step': (0, 10),
    #                                'subsample': (0.4, 1.0),
    #                                'colsample_bytree': (0.4, 1.0)
    #                               })
    #
    # with warnings.catch_warnings():
    #     warnings.filterwarnings('ignore')
    #     # XGB_BO.maximize(init_points=2, n_iter=5, acq='ei', xi=0.0)
    #     XGB_BO.maximize(init_points=10, n_iter=50, acq='ei', xi=0.0)
    #     # XGB_BO.maximize(init_points=10, n_iter=50, acq='ei', xi=0.01)
    #     # XGB_BO.maximize(init_points=10, n_iter=50, acq='ucb', kappa=10)
    #     # XGB_BO.maximize(init_points=10, n_iter=50, acq='ucb', kappa=1)
    #
    # params = XGB_BO.max["params"]
    # print(params)
    # print('Maximum XGBOOST value: %f' % XGB_BO.res)
    # print('Best XGBOOST opt_parameters: ', XGB_BO.max)

    opti_parameters = {'max_depth': (2, 12),
                       'gamma': (0.001, 10.0),
                       'min_child_weight': (0, 20),
                       'max_delta_step': (0, 10),
                       'subsample': (0.4, 1.0),
                       'colsample_bytree': (0.4, 1.0)
                       }
    gp_params = {"init_points": 2, "n_iter": 2, "acq": 'ei', "xi": 0.0, "alpha": 1e-4}

    opt_xgb = BayesOptimizationXGB(X_train, y_train, X_test, y_test)
    params_op = opt_xgb.train_opt(opti_parameters, gp_params)
    log.logger.info('BestScore: {}, BestIter: {}'.format(opt_xgb.BestScore, opt_xgb.BestIter))
