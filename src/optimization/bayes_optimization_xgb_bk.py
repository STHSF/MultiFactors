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

BestScore = -1.
BestIter = 0


def xgb_cv(max_depth, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree):
    global BestScore
    global BestIter

    data_train = xgb.DMatrix(X_train, label=y_train)
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

    folds = 2
    xgbc = xgb.cv(
        paramt,
        data_train,
        num_boost_round=20000,
        stratified=True,
        nfold=folds,
        early_stopping_rounds=100,
        verbose_eval=True,
        show_stdv=True
    )

    val_score = xgbc['test-merror-mean'].iloc[-1]
    train_score = xgbc['train_opt-merror-mean'].iloc[-1]
    print(' Stopped after %d iterations with train_opt-score = %f val-score = %f ( diff = %f ) train_opt-gini = %f val-gini = %f' % (
        len(xgbc), train_score, val_score, (train_score - val_score), (train_score * 2 - 1), (val_score * 2 - 1)))

    if val_score > BestScore:
        BestScore = val_score
        BestIter = len(xgbc)
    return (val_score * 2) - 1


def xgb_no(max_depth, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree):
    global BestScore
    global BestIter

    data_train = xgb.DMatrix(X_train, label=y_train)
    data_test = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(data_test, 'eval'), (data_train, 'train_opt')]
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
    print(' Stopped after %d iterations with train_opt-score = %f train_opt-gini = %f' %
          (best_round, best_score, (best_score * 2 - 1)))

    if best_score > BestScore:
        BestScore = best_score
        BestIter = best_round

    return (best_score * 2) - 1


def bayesian_optimization(function, parameters):
    # gp_params = {"init_points": 10, "n_iter": 50, "acq": 'ucb', "xi": 0.0, "alpha": 1e-4, "kappa": 10}
    # gp_params = {"init_points": 2, "n_iter": 50, "acq": 'ei', "xi": 0.0, "alpha": 1e-4}
    gp_params = {"init_points": 2, "n_iter": 2, "acq": 'ei', "xi": 0.0, "alpha": 1e-4}

    BO = BayesianOptimization(function, parameters)
    BO.maximize(**gp_params)

    return BO.max


# Train model
def train_opt(function, parameters):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        best_solution = bayesian_optimization(function, parameters)
    params_opt = best_solution["params"]
    print('Best XGBOOST opt_parameters: %s' % params_opt)
    return params_opt


if __name__ == '__main__':

    # Classify Parameter Optimization Test
    iris = load_iris()
    data = iris.data
    target = iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
    print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))

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

    parameters = {'max_depth': (2, 12),
                  'gamma': (0.001, 10.0),
                  'min_child_weight': (0, 20),
                  'max_delta_step': (0, 10),
                  'subsample': (0.4, 1.0),
                  'colsample_bytree': (0.4, 1.0)
                  }
    # None CV
    params_op = train_opt(xgb_no, parameters)
