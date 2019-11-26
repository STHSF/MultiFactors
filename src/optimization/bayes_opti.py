#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: bayes_opti.py
@time: 2019/11/26 4:23 下午
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb

import gc
import warnings

from bayes_opt import BayesianOptimization

#Import libraries
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# Bayesian optimization
def bayesian_optimization(dataset, function, parameters):
   X_train, y_train, X_test, y_test = dataset
   n_iterations = 5
   gp_params = {"alpha": 1e-4}

   BO = BayesianOptimization(function, parameters)
   BO.maximize(n_iter=n_iterations, **gp_params)

   return BO.max


def rfc_optimization(cv_splits):
    def function(n_estimators, max_depth, min_samples_split):
        return cross_val_score(
            RandomForestClassifier(
                n_estimators=int(max(n_estimators, 0)),
                max_depth=int(max(max_depth, 1)),
                min_samples_split=int(max(min_samples_split, 2)),
                n_jobs=-1,
                random_state=42,
                class_weight="balanced"),
            X=X_train,
            y=y_train,
            cv=cv_splits,
            scoring="roc_auc",
            n_jobs=-1).mean()

    parameters = {"n_estimators": (10, 1000),
                  "max_depth": (1, 150),
                  "min_samples_split": (2, 10)}

    return function, parameters


def xgb_optimization(cv_splits, eval_set):
    def function(eta, gamma, max_depth):
        return cross_val_score(
            xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=3,
                learning_rate=max(eta, 0),
                gamma=max(gamma, 0),
                max_depth=int(max_depth),
                seed=42,
                nthread=-1,
                scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            ),
            X=X_train,
            y=y_train,
            cv=cv_splits,
            scoring="roc_auc",
            fit_params={
                "early_stopping_rounds": 10,
                "eval_metric": "auc",
                "eval_set": eval_set},
            n_jobs=-1).mean()

    parameters = {"eta": (0.001, 0.4),
                  "gamma": (0, 20),
                  "max_depth": (1, 2000)}

    return function, parameters


# Train model
def train(X_train, y_train, X_test, y_test, function, parameters):
    dataset = (X_train, y_train, X_test, y_test)
    cv_splits = 4

    best_solution = bayesian_optimization(dataset, function, parameters)
    params = best_solution["params"]


    # model = xgb.XGBClassifier(eta=int(max(params["eta"], 0)),
    #                           gamma=int(max(params["gamma"], 0)),
    #                           max_depth=int(max(params["max_depth"], 0)),
    #                           seed=42,
    #                           nthread=-1,
    #                           scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    #                           )
    #
    # model.fit(X_train, y_train)

    return model


if __name__ == '__main__':

    # CLASSIFY TEST
    iris = load_iris()
    data = iris.data
    target = iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    data_train = xgb.DMatrix(X_train, label=y_train)
    data_test = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(data_test, 'eval'), (data_train, 'train_opt')]
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3, 'metrics':'auc'}

    bst = xgb.train(param, data_train, num_boost_round=10, evals=watchlist)
    y_hat = bst.predict(data_test)
    result = y_test.reshape(1, -1) == y_hat
    print('the accuracy:\t', float(np.sum(result)) / len(y_hat))

    function, parameters = xgb_optimization(5, watchlist)
    function(eta=1, gamma=10, max_depth=5)
    # train_opt(X_train, y_train, X_test, y_test, function, parameters)