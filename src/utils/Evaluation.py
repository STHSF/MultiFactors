#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: Evaluation.py
@time: 2019/11/18 5:07 下午
"""
import numpy as np
from sklearn import metrics


class RegressionEvaluate(object):

    @staticmethod
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

    @staticmethod
    def smape(y_true, y_pred):
        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

    @staticmethod
    def mse(y_true, y_pred):
        return metrics.mean_squared_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true, y_pred):
        return metrics.mean_absolute_error(y_true, y_pred)

    @staticmethod
    def r_square_error(y_true, y_pred):
        r_square_error_ = metrics.r2_score(y_true, y_pred)
        _r_square_error = 1 - metrics.mean_squared_error(y_true, y_pred) / np.var(y_true)
        return _r_square_error


class ClassifyEvaluate(object):

    @staticmethod
    def auc(y_true, y_pred):
        auc_bool = y_true.reshape(1, -1) == y_pred
        return float(np.sum(auc_bool)) / len(y_pred)


cls_eva = ClassifyEvaluate()
reg_eva = RegressionEvaluate()

if __name__ == '__main__':
    y_true = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
    y_pred = np.array([1.0, 4.5, 3.5, 5.0, 8.0, 4.5, 1.0])

    eva = RegressionEvaluate()
    # MSE
    print(eva.mse(y_true, y_pred))  # 8.107142857142858
    # RMSE
    print(eva.rmse(y_true, y_pred))  # 2.847304489713536
    # MAE
    print(eva.mae(y_true, y_pred))  # 1.9285714285714286
    # MAPE
    print(eva.mape(y_true, y_pred))  # 76.07142857142858
    # SMAPE
    print(eva.smape(y_true, y_pred))  # 57.76942355889724

