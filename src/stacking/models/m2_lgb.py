#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: m2_lgb.py
@time: 2019-03-04 11:03
"""

import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import time
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.conf.configuration import classify_conf, regress_conf
from src.utils import log_util
from src.utils.Evaluation import cls_eva, reg_eva
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


log = log_util.Logger('m2_lightgbm', level='info')


class LightGBM(object):
    def __init__(self, args):
        self.params = args.params
        self.max_round = args.max_round
        self.cv_folds = args.cv_folds
        self.early_stop_round = args.early_stop_round
        self.seed = args.seed
        self.save_model_path = args.save_model_path

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        best_model, best_round, best_score = None, None, None
        if self.cv_folds is None:
            log.logger.info('NonCrossValidation。。。。')
            if x_valid is None and y_valid is None:
                x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
            else:
                x_valid, y_valid = x_valid, y_valid
            d_train = lgb.Dataset(x_train, label=y_train)
            d_valid = lgb.Dataset(x_valid, label=y_valid)
            watchlist = [d_train, d_valid]
            best_model = lgb.train(self.params,
                                   d_train,
                                   num_boost_round=self.max_round,
                                   valid_sets=watchlist,
                                   early_stopping_rounds=self.early_stop_round)
            best_round = best_model.best_iteration
            best_score = best_model.best_score

        else:
            log.logger.info('CrossValidation ........')
            d_train = lgb.Dataset(x_train, label=y_train)
            if self.params['objective'] is 'multiclass':
                cv_result = lgb.cv(self.params,
                                   d_train,
                                   num_boost_round=self.max_round,
                                   nfold=self.cv_folds,
                                   seed=self.seed,
                                   verbose_eval=True,
                                   metrics=['multi_error', 'multi_logloss'],
                                   early_stopping_rounds=self.early_stop_round,
                                   show_stdv=False)
                log.logger.info('cv_result %s' % cv_result)
                log.logger.info('type_cv_result %s' % type(cv_result))
                best_round = len(cv_result['multi_error-mean'])
                best_score = cv_result['multi_error-mean'][-1]
                best_model = lgb.train(self.params, d_train, best_round)

            elif self.params['objective'] is 'regression':
                cv_result = lgb.cv(self.params,
                                   d_train,
                                   num_boost_round=self.max_round,
                                   nfold=self.cv_folds,
                                   seed=self.seed,
                                   verbose_eval=True,
                                   metrics=['l2', 'auc'],
                                   early_stopping_rounds=self.early_stop_round,
                                   show_stdv=False)
                log.logger.info('cv_result %s' % cv_result)
                log.logger.info('type_cv_result %s' % type(cv_result))
                if 'l2' in self.params['metric']:
                    min_error = min(cv_result['l2-mean'])
                    best_round = cv_result[cv_result['l2-mean'].isin([min_error])].index[0]
                elif 'rmse' in self.params['metric']:
                    min_error = min(cv_result['test-rmse-mean'])
                    best_round = cv_result[cv_result['test-rmse-mean'].isin([min_error])].index[0]
                else:
                    min_error = None
                best_score = min_error
                best_model = lgb.train(self.params, d_train, best_round)
            else:
                print('ERROR: LightGBM OBJECTIVE IS NOT CLASSIFY OR REGRESSION')
                exit()
        return best_model, best_score, best_round

    def predict(self, bst_model, x_test, save_result_path=None):
        if conf.params['objective'] == "multiclass":
            pre_data = bst_model.predit(x_test).argmax(axis=1)
        else:
            pre_data = bst_model.predit(x_test)

        if save_result_path:
            df_reult = pd.DataFrame()
            df_reult['result'] = y_pred
            df_reult.to_csv(save_result_path, index=False)

        return pre_data

    def _kfold(self):
        pass

    def set_params(self, **params):
        self.params.update(params)

    def get_params(self):
        return self.params

    def save_model(self, best_model):
        if self.save_model_path:
            best_model.save_model(self.save_model_path)

    def load_model(self, model_path=None):
        if model_path is None and self.save_model_path is None:
            log.logger.error('Model Load Error, {} or {} is not exit'.format(model_path, self.save_model_path))
            log.logger.error('Exit.........')
            exit()
        if model_path:
            bst_model = joblib.load(model_path)
        else:
            bst_model = joblib.load(self.save_model_path)
        return bst_model


def run_feat_search(X_train, X_test, y_train, feature_names):
    pass


def lgb_predict(model, x_test, y_test, save_result_path=None):
    # x_test = x_test.flatten()
    classify_conf.lgb_config_c()
    if classify_conf.params['objective'] == "multiclass":
        y_pred = model.predict(x_test).argmax(axis=1)
        log.logger.info(y_pred)
        log.logger.info(y_test)
        if y_test is not None:
            # AUC计算
            log.logger.info('The Accuracy:\t{}'.format(cls_eva.auc(y_test, y_pred)))

    elif regress_conf.params['objective'] == "regression":
        y_pred = model.predict(x_test)
        log.logger.info('y_pre: {}'.format(y_pred))
    else:
        y_pred = None

    if save_result_path:
        df_reult = pd.DataFrame(y_pred, columns='result')
        df_reult.to_csv(save_result_path, index=False)


def run_cv(x_train, x_test, y_test, y_train):
    regress_conf.lgb_config_r()
    regress_conf.cv_folds = 5
    tic = time.time()
    data_message = 'x_train.shape={}, x_test.shape={}'.format(x_train.shape, x_test.shape)
    log.logger.info(data_message)

    lgb = LightGBM(regress_conf)
    lgb_model, best_score, best_round = lgb.fit(x_train, y_train)
    log.logger.info('Time cost {}s'.format(time.time() - tic))
    result_message = 'best_round={}, best_score={}'.format(best_round, best_score)
    log.logger.info(result_message)

    # predict
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_lgb_{}-{:.4f}.csv'.format(now, best_score)
    # check_path(result_path)
    lgb_predict(lgb_model, x_test, y_test, save_result_path=None)


if __name__ == '__main__':
    import json
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    iris = load_iris()
    data = iris.data
    target = iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    # regress_conf.lgb_config_r()
    classify_conf.lgb_config_c()
    log.logger.info('Model Params:\n{}'.format(classify_conf.params))

    # lgbm = LightGBM(classify_conf)
    # best_model, best_score, best_round = lgbm.fit(X_train, y_train)
    # lgb_predict(best_model, X_test, y_test)

    run_cv(X_train, X_test, y_test, y_train)

    # # 创建成lgb特征的数据集格式
    # lgb_train = lgb.Dataset(X_train, y_train)
    # lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # print('Start training...')
    # # 训练 cv and train
    # gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
    #
    # print('Save model...')
    # # 保存模型到文件
    # gbm.save_model('model.txt')
    #
    # print('Start predicting...')
    # # 预测数据集
    # y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # # 评估模型
    # print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)