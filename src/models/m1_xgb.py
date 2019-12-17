#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: m1_xgb.py
@time: 2019-03-04 09:36
"""

import os
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import time
import numpy as np
from math import *
import xgboost as xgb
from utils import log_util
from src.conf.configuration import lgb_conf
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from utils.Evaluation import cls_eva, reg_eva
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None, 'display.max_columns', None, "display.max_colwidth", 1000, 'display.width', 1000)

log = log_util.Logger('m1_xgboost', level='info')


class XGBooster(object):
    def __init__(self, args):
        self.xgb_params = args.params
        self.num_boost_round = args.max_round
        self.cv_folds = args.cv_folds
        self.ts_cv_folds = args.ts_cv_folds
        self.early_stop_round = args.early_stop_round
        self.cv_seed = args.cv_seed
        self.save_model_path = args.save_model_path

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        # xgb_start = time.time()
        best_model = None
        best_round = None
        best_score = {}
        if self.cv_folds is not None:
            log.logger.info('CrossValidation。。。。')
            d_train = xgb.DMatrix(x_train, label=y_train)
            cv_result = self._kfold(d_train)
            print('cv_result %s' % cv_result)
            print('type_cv_result %s' % type(cv_result))
            # min_rmse = pd.Series(cv_result['test-rmse-mean']).min()
            # self.best_score['min_test-rmse-mean'] = min_rmse
            # self.best_round = cv_result[cv_result['test-rmse-mean'].isin([min_rmse])].index[0]
            best_score['min_test-rmse-mean'] = pd.Series(cv_result['test-rmse-mean']).min()
            best_round = pd.Series(cv_result['test-rmse-mean']).idxmin()
            best_model = xgb.train(self.xgb_params, d_train, best_round)

        elif self.ts_cv_folds is not None:
            log.logger.info('TimeSeriesCrossValidation。。。。')
            # 时间序列k_fold
            bst_score = 0
            details = []
            scores = []
            tscv = TimeSeriesSplit(n_splits=self.ts_cv_folds)
            if self.xgb_params['objective'] is not 'reg:linear':
                log.logger.error('Objective ERROR........')
                exit()

            for n_fold, (tr_idx, val_idx) in enumerate(tscv.split(x_train)):
                print(f'the {n_fold} training start ...')
                # tr_x, tr_y, val_x, val_y = x_train.iloc[tr_idx], y_train[tr_idx], x_train.iloc[val_idx], y_train[val_idx]
                d_train = xgb.DMatrix(tr_x, label=tr_y)
                d_valid = xgb.DMatrix(val_x, label=val_y)
                watchlist = [(d_train, "train"), (d_valid, "valid")]
                xgb_model = xgb.train(params=self.xgb_params,
                                      dtrain=d_train,
                                      num_boost_round=self.num_boost_round,
                                      evals=watchlist,
                                      early_stopping_rounds=self.early_stop_round)
                details.append((xgb_model.best_score, xgb_model.best_iteration, xgb_model))

                if xgb_model.best_score > bst_score:
                    bst_score = xgb_model.best_score
                    best_model = xgb_model
                    best_round = xgb_model.best_iteration
                else:
                    best_model = xgb_model
                    best_round = xgb_model.best_iteration
                scores.append(xgb_model.best_score)
            best_score['avg_score'] = np.mean(scores)

        else:
            log.logger.info('NonCrossValidation。。。。')
            if x_valid is None and y_valid is None:
                # 注意这里的shift
                # x_train, x_valid, y_train, y_valid = train_test_sp(x_train, y_train, test_size=0.2, shift=0)
                d_train = xgb.DMatrix(x_train, label=y_train)
                watchlist = [(d_train, "train")]
            else:
                d_train = xgb.DMatrix(x_train, label=y_train)
                d_valid = xgb.DMatrix(x_valid, label=y_valid)
                watchlist = [(d_train, "train"), (d_valid, "valid")]

            best_model = xgb.train(params=self.xgb_params,
                                   dtrain=d_train,
                                   num_boost_round=self.num_boost_round,
                                   evals=watchlist,
                                   verbose_eval=5,
                                   early_stopping_rounds=self.early_stop_round)
            best_round = best_model.best_iteration
            best_score['best_score'] = best_model.best_score
        # print('spend time :' + str((time.time() - xgb_start)) + '(s)')
        return best_score, best_round, best_model

    def predict(self, bst_model, x_pred):
        dpred = xgb.DMatrix(x_pred)
        return bst_model.predict(dpred)

    def _kfold(self, dtrain):
        cv_result = xgb.cv(self.xgb_params,
                           dtrain,
                           num_boost_round=self.num_boost_round,
                           nfold=self.cv_folds,
                           seed=self.cv_seed,
                           verbose_eval=True,
                           early_stopping_rounds=self.early_stop_round,
                           show_stdv=False,
                           shuffle=True)
        return cv_result

    @staticmethod
    def plot_feature_importance(best_model, top_n=20):
        print(80 * '*')
        print(31 * '*' + 'Feature Importance' + 31 * '*')
        print(80 * '.')
        print("\n".join((".%50s => %9.5f" % x) for x in
                        sorted(zip(best_model.get_fscore().keys(), best_model.get_fscore().values()),
                               key=lambda x: x[1],
                               reverse=True)))
        print(80 * '.')

        # plot
        feature_importance_df = pd.DataFrame()
        feature_importance_df["Feature"] = best_model.get_fscore().keys()
        feature_importance_df["importance"] = best_model.get_fscore().values()
        # # plot feature importance of top_n
        best_features = feature_importance_df[["Feature", "importance"]].sort_values(by="importance", ascending=True)
        best_features['importance'] = best_features['importance'] / best_features['importance'].sum()
        # best_features = feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=True)[:top_n].reset_index()
        plt.figure(figsize=(16, 10))
        best_features[:top_n].plot(kind='barh', x='Feature', y='importance', legend=False, figsize=(16, 10))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative Features')
        plt.ylabel('Feature Importance Score')
        # save picture
        # plt.gcf().savefig('feature_importance_xgb.png')
        plt.show()

    def get_params(self, deep=True):
        return self.xgb_params

    def set_params(self, **params):
        self.xgb_params.update(params)

    def save_model(self, best_model,model_path=None):
        # now = time.strftime('%Y-%m-%d %H:%M')
        model_name = 'xgboost_{}.bat'.format(now)
        if model_path:
            joblib.dump(best_model, model_path + model_name)
        else:
            joblib.dump(best_model, self.save_model_path + model_name)

    def load_model(self, model_path=None):
        if model_path is None and self.save_model_path is None:
            print('bst_model load error')
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


def xgb_predict(model, conf, x_test, y_test=None, result_save_path=None):

    d_test = xgb.DMatrix(x_test)
    if conf.params['objective'] == 'multi:softmax':
        y_pred = model.predict(d_test)
        if y_test is None:
            log.logger.info(y_test)
        else:
            log.logger.info(y_test)
            log.logger.info(y_pred)
            log.logger.info('The Accuracy:\t{}'.format(cls_eva.auc(y_test, y_pred)))
    elif conf.params['objective'] == "reg:linear":
        y_pred = model.predict(d_test)
        if y_test is None:
            log.logger.info('y_pre: \n{}'.format(y_pred))
        else:
            log.logger.info('y_pre: \n{}'.format(y_pred))
            log.logger.info('y_test: \n{}'.format(y_test))

            rmse = reg_eva.rmse(y_test, y_pred)
            log.logger.info('rmse: {}'.format(rmse))
            r2_sc = reg_eva.r_square_error(y_test, y_pred)
            log.logger.info('r_square_error: {}'.format(r2_sc))
            # PLOT
            plot_figure(y_pred, y_test, 'xgb_regression')
    else:
        log.logger.error('CAN NOT FIND OBJECTIVE PARAMS')
        y_pred = None

    if result_save_path:
        if not isinstance(x_test, pd.DataFrame):
            df_result = pd.DataFrame(x_test)
        else:
            df_result = x_test
        df_result['y_test'] = y_test
        df_result['result'] = y_pred
        df_result.to_csv(result_save_path, index=False)


def run_cv(x_train, x_test, y_train, y_test, regress_conf):
    x_train = x_train
    tic = time.time()
    data_message = 'X_train.shape={}, X_test.shape = {}'.format(np.shape(x_train), np.shape(x_test))
    log.logger.info(data_message)
    xgb = XGBooster(regress_conf)
    best_score, best_round, best_model = xgb.fit(x_train, y_train)
    log.logger.info('Training time cost {}s'.format(time.time() - tic))
    # xgb.save_model()
    result_message = 'best_score = {}, best_round = {}'.format(best_score, best_round)
    log.logger.info(result_message)

    # now = time.strftime('%Y-%m-%d %H:%M')
    result_saved_path = '../result/result_{}-{:.4f}.csv'.format(now, best_round)
    # xgb_predict(best_model, x_test, y_test, result_save_path=result_saved_path)
    xgb_predict(best_model, x_test, y_test, result_save_path=None)


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
    # # 输入数据为dataframe格式
    # train_sample_df = pd.read_csv('../../data/dataset/training_sample.csv')
    # print(train_sample_df.head())
    # train_dataset_df = train_sample_df[['alpha_1', 'alpha_2', 'alpha_3', 'alpha_4', 'alpha_5',
    #                                     'alpha_6', 'alpha_7', 'alpha_8', 'alpha_9', 'alpha_10']]
    # label_dataset_df = train_sample_df[['dx_2']]
    #
    # x_train, x_test, y_train, y_test = train_test_sp(train_dataset_df[:30000], label_dataset_df[:30000])
    # print('x_train_pre: \n%s' % x_train.head())
    # # print('y_train_pre: %s' % y_train.head())
    # print('x_test_pre: %s' % x_test.head())
    # # print('y_test_pre: %s' % y_test.head())
    #
    # # 数据统计用
    # # x_test.to_csv('../result/x_test_{}.csv'.format(now), index=0)
    # # y_test.to_csv('../result/y_test_{}.csv'.format(now), index=0)
    #
    # # 样本预处理(标准化等)
    # x_train_mean = x_train.mean()
    # x_train_std = x_train.std()
    # x_train = (x_train - x_train_mean) / x_train_std
    # x_test = (x_test - x_train_mean) / x_train_std
    # print(x_train.head())
    # print(x_test.head())
    #
    # # # 超参数
    # xgb_conf.xgb_config_r()
    # log.logger.info("params before: {}".format(xgb_conf.params))
    # # 超参数寻优
    # from src.optimization.bayes_optimization_xgb import *
    # opt_parameters = {'max_depth': (2, 12),
    #                   'gamma': (0.001, 10.0),
    #                   'min_child_weight': (0, 20),
    #                   'max_delta_step': (0, 10),
    #                   'subsample': (0.01, 0.99),
    #                   'colsample_bytree': (0.01, 0.99)
    #                   }
    # gp_params = {"init_points": 2, "n_iter": 2, "acq": 'ei', "xi": 0.0, "alpha": 1e-4}
    # bayes_opt_xgb = BayesOptimizationXGB(x_train.values, y_train.values, x_test.values, y_test.values)
    # params_op = bayes_opt_xgb.train_opt(opt_parameters, gp_params)
    #
    # xgb_conf.params.update(params_op)
    # log.logger.info("params after: {}".format(xgb_conf.params))
    # # 模型训练
    # run_cv(x_train.values, x_test.values, y_train.values, y_test.values, xgb_conf)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris, load_boston
    from sklearn.model_selection import train_test_split
    from src.conf.configuration import xgb_conf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", dest="test_type", default="classification", type=str, help="输入回归or分类")
    args = parser.parse_args()
    _type = args.test_type

    def classify_test():
        # ===========================classify Test start==========================================
        iris = load_iris()
        data = iris.data
        target = iris.target
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
        log.logger.info('{},{},{},{}'.format(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test)))
        xgb_conf.xgb_config_c()

        # # train model
        xgbc = XGBooster(xgb_conf)
        best_score, best_round, best_model = xgbc.fit(X_train, y_train)
        # # predict
        xgb_predict(best_model, xgb_conf, X_test, y_test)
        xgbc.plot_feature_importance(best_model)
        # ===========================classify Test end==========================================

    def regression_test():
        # ===========================REGRESSION TEST START==========================================
        boston = load_boston()
        data = boston.data
        target = boston.target
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
        log.logger.info('{},{},{},{}'.format(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test)))
        xgb_conf.xgb_config_r()

        # train model
        xgbc = XGBooster(xgb_conf)
        best_score, best_round, best_model = xgbc.fit(X_train, y_train)
        # eval
        xgb_predict(best_model, xgb_conf, X_test, y_test)
        xgbc.plot_feature_importance(best_model)
        # ===========================REGRESSION TEST END==========================================

    if _type == 'regression':
        regression_test()
    elif _type == 'classification':
        classify_test()
