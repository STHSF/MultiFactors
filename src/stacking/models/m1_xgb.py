#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: m1_xgb.py
@time: 2019-03-04 09:36
"""

import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import time
import argparse
import numpy as np
from math import *
import xgboost as xgb
from src.conf.configuration import regress_conf
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None, 'display.max_columns', None, "display.max_colwidth", 1000, 'display.width', 1000)


class XGBooster(object):
    def __init__(self, args):
        self.xgb_params = args.params
        self.num_boost_round = args.max_round
        self.cv_folds = args.cv_folds
        self.avg_score = None
        self.ts_cv_folds = args.ts_cv_folds
        self.early_stop_round = args.early_stop_round
        self.seed = args.seed
        self.save_model_path = args.save_model_path

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        xgb_start = time.time()
        if self.cv_folds is not None:
            print('cross_validation。。。。')
            d_train = xgb.DMatrix(x_train, label=y_train)
            cv_result = self._kfold(d_train)
            print('cv_result %s' % cv_result)
            print('type_cv_result %s' % type(cv_result))
            min_rmse = cv_result['test-rmse-mean'].min()
            self.best_round = cv_result[cv_result['test-rmse-mean'].isin([min_rmse])].index[0]
            self.best_score = min_rmse
            self.best_model = xgb.train(self.xgb_params, d_train, self.best_round)

        elif self.ts_cv_folds is not None:
            # 时间序列k_fold
            best_score = 0
            details = []
            tscv = TimeSeriesSplit(n_splits=self.ts_cv_folds)
            for n_fold, (tr_idx, val_idx) in enumerate(tscv.split(x_train)):
                print(f'the {n_fold} training start ...')
                tr_x, tr_y, val_x, val_y = x_train.iloc[tr_idx], y_train[tr_idx], x_train.iloc[val_idx], y_train[val_idx]
                d_train = xgb.DMatrix(tr_x, label=tr_y)
                d_valid = xgb.DMatrix(val_x, label=val_y)
                watchlist = [(d_train, "train"), (d_valid, "valid")]
                xgb_model = xgb.train(params=self.xgb_params,
                                      dtrain=d_train,
                                      num_boost_round=self.num_boost_round,
                                      evals=watchlist,
                                      early_stopping_rounds=self.early_stop_round)
                details.append((xgb_model.best_score, xgb_model.best_iteration, xgb_model))

                if xgb_model.best_score > best_score:
                    best_score = xgb_model.best_score
                    self.best_score = xgb_model.best_score
                    self.best_model = xgb_model
                    self.best_round = xgb_model.best_iteration
                else:
                    self.best_score = xgb_model.best_score
                    self.best_model = xgb_model
                    self.best_round = xgb_model.best_iteration
                scores.append(xgb_model.best_score)
            self.avg_score = np.mean(scores)

        else:
            print('non_cross_validation。。。。')
            if x_val is None and y_val is None:
                # 注意这里的shift
                x_train, x_valid, y_train, y_valid = train_test_sp(x_train, y_train, test_size=0.2, shift=0)
            else:
                x_valid, y_valid = x_val, y_val
            d_train = xgb.DMatrix(x_train, label=y_train)
            d_valid = xgb.DMatrix(x_valid, label=y_valid)
            watchlist = [(d_train, "train"), (d_valid, "valid")]
            self.best_model = xgb.train(params=self.xgb_params,
                                        dtrain=d_train,
                                        num_boost_round=self.num_boost_round,
                                        evals=watchlist,
                                        early_stopping_rounds=self.early_stop_round)
            self.best_round = self.best_model.best_iteration
            self.best_score = self.best_model.best_score

        # print('spend time :' + str((time.time() - xgb_start)) + '(s)')
        return self.best_score, self.best_round, self.best_model

    def predict(self, bst_model, x_pred):
        dpred = xgb.DMatrix(x_pred)
        return bst_model.predict(dpred)

    def _kfold(self, dtrain):
        cv_result = xgb.cv(self.xgb_params, dtrain,
                           num_boost_round=self.num_boost_round,
                           nfold=self.cv_folds,
                           seed=self.seed,
                           verbose_eval=True,
                           early_stopping_rounds=self.early_stop_round,
                           show_stdv=False,
                           shuffle=True)
        return cv_result

    def plot_feature_importances(self):
        feat_imp = pd.Series(self.best_model.get_fscore()).sort_values(ascending=False)
        feat_imp.plot(title='Feature Importances')
        plt.ylabel('Feature Importance Score')

    def get_params(self, deep=True):
        return self.xgb_params

    def set_params(self, **params):
        self.xgb_params.update(params)

    def save_model(self, model_path=None):
        # now = time.strftime('%Y-%m-%d %H:%M')
        model_name = 'xgboost_{}.bat'.format(now)
        if model_path:
            joblib.dump(self.best_model, model_path + model_name)
        else:
            joblib.dump(self.best_model, self.save_model_path + model_name)

    def load_model(self, model_path=None):
        if model_path is None and self.save_model_path is None:
            print('model load error')
            exit()
        if model_path:
            bst_model = joblib.load(model_path)
        else:
            bst_model = joblib.load(self.save_model_path)
        return bst_model


def cal_acc(test_data, pre_data):
    pass


def plot_figure(y_pred, y_test, fig_name):
    fig1 = plt.figure(num=fig_name, figsize=(10, 3), dpi=75, facecolor='#FFFFFF', edgecolor='#0000FF')
    plt.plot(y_pred)
    plt.plot(y_test)
    plt.title(u"REGRESSION")
    plt.legend((u'Predict', u'Test'), loc='best')  # sets our legend for our graph.
    plt.show()
    plt.close()


def ic_cal(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    return np.corrcoef(y_pred, y_test)[0, 1]


def xgb_predict(model, x_test, y_test=None, save_result_path=None):
    d_test = xgb.DMatrix(x_test)
    # 输出
    y_pred = model.predict(d_test)
    if y_test is None:
        return y_pred
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('rmse_of_pred: %s' % rmse)
    r_square_ = r2_score(y_test, y_pred)
    r_square = 1 - mean_squared_error(y_test, y_pred)/ np.var(y_test)
    print('r_square_of_pred: %s' % r_square)
    print('r_square_of_pred: %s' % r_square_)
    # print(y_pred)
    # print(y_test)
    plot_figure(y_pred, y_test, 'xgb_predict')

    if save_result_path:
        df_reult = pd.DataFrame(x_test)
        df_reult['y_test'] = y_test
        df_reult['result'] = y_pred
        df_reult.to_csv(save_result_path, index=False)


def run_cv(x_train, x_test, y_train, y_test):
    x_train = x_train
    regress_conf.xgb_config_r()
    tic = time.time()
    data_message = 'X_train.shape={}, X_test.shape = {}'.format(np.shape(x_train), np.shape(x_test))
    print(data_message)
    xgb = XGBooster(regress_conf)
    best_score, best_round, best_model = xgb.fit(x_train, y_train)
    print('Training time cost {}s'.format(time.time() - tic))
    # xgb.save_model()
    result_message = 'best_score = {}, best_round = {}'.format(best_score, best_round)
    print(result_message)

    # now = time.strftime('%Y-%m-%d %H:%M')
    result_saved_path = '../result/result_{}-{:.4f}.csv'.format(now, best_score)
    # xgb_predict(best_model, x_test, y_test, save_result_path=result_saved_path)
    xgb_predict(best_model, x_test, y_test, save_result_path=None)


def train_test_sp(train_dataset_df, label_dataset_df, test_size=0.02, shift=100, random=None):
    """
    # 训练集和测试集划分
    :param train_dataset_df: 训练集
    :param label_dataset_df: 标签集
    :param test_size: 测试数据所占比例
    :param shift: 测试数据后移, 针对时间序列的数据, 为了train和test不重合
    :param random: 随机划分还是分段切分
    :return:
    """
    if random:
        # 随机划分
        x_train, x_test, y_train, y_test = train_test_split(train_dataset_df, label_dataset_df, test_size=0.02, random_state=10000, shuffle=True)
    else:
        # 按时间序列前后划分
        len_data = len(train_dataset_df)
        a1 = ceil(len_data * (1 - test_size))
        x_train, x_test = train_dataset_df[:a1], train_dataset_df[a1+shift:]
        y_train, y_test = label_dataset_df[:a1], label_dataset_df[a1+shift:]

    return x_train, x_test, y_train, y_test


now = time.strftime('%Y-%m-%d %H:%M')

if __name__ == '__main__':
    # 输入数据为dataframe格式
    train_sample_df = pd.read_csv('../data/dataset/training_sample.csv')
    print(train_sample_df.head())
    train_dataset_df = train_sample_df[['alpha_1', 'alpha_2', 'alpha_3', 'alpha_4', 'alpha_5',
                                        'alpha_6', 'alpha_7', 'alpha_8', 'alpha_9', 'alpha_10']]
    label_dataset_df = train_sample_df[['dx_2']]

    x_train, x_test, y_train, y_test = train_test_sp(train_dataset_df[:30000], label_dataset_df[:30000])
    print('x_train_pre: %s' % x_train.head())
    print('y_train_pre: %s' % y_train.head())
    print('x_test_pre: %s' % x_test.head())
    print('y_test_pre: %s' % y_test.head())

    # 数据统计用
    # x_test.to_csv('../result/x_test_{}.csv'.format(now), index=0)
    # y_test.to_csv('../result/y_test_{}.csv'.format(now), index=0)

    # 样本预处理

    # 模型训练
    run_cv(x_train.values, x_test.values, y_train.values, y_test.values)

