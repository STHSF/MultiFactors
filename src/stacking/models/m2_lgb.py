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
from src.conf.configuration import classify_conf
from sklearn.model_selection import train_test_split


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
            print('Non Cross Validation....')
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
            cv_result = None

        else:
            print('Cross Validation')
            d_train = lgb.Dataset(x_train, label=y_train)
            cv_result = lgb.cv(self.params,
                               d_train,
                               num_boost_round=self.max_round,
                               nfold=self.cv_folds,
                               seed=self.seed,
                               verbose_eval=True,
                               metrics=['multi_error', 'multi_logloss'],
                               early_stopping_rounds=self.early_stop_round,
                               show_stdv=False)
            print('cv_result %s' % cv_result)
            print('type_cv_result %s' % type(cv_result))
            if self.params['objective'] is 'multiclass':
                best_round = len(cv_result['multi_error-mean'])
                best_score = cv_result['multi_error-mean'][-1]
                best_model = lgb.train(self.params, d_train, best_round)

            elif self.params['objective'] is 'reg:linear':
                min_error = cv_result['test-rmse-mean'].min()
                best_round = cv_result[cv_result['test-rmse-mean'].isin([min_error])].index[0]
                best_score = min_error
                best_model = lgb.train(self.params, d_train, best_round)
            else:
                print('ERROR: LightGBM OBJECTIVE IS NOT CLASSIFY OR REGRESSION')
                exit()

        return best_model, best_score, best_round, cv_result

    @staticmethod
    def predict(bst_model, x_test, y_test=None, result_path=None):
        if conf.params['objective'] == "multiclass":
            y_pred = bst_model.predict(x_test).argmax(axis=1)
            print(y_pred)
            print(y_test)
            if y_test is not None:
                auc_bool = y_test.reshape(1, -1) == y_pred
                print('the accuracy:\t', float(np.sum(auc_bool)) / len(y_pred))
        else:
            # 输出概率
            y_pred_prob = bst_model.predict(x_test)
            y_pred = y_pred_prob
        if result_path:
            df_reult = pd.DataFrame()
            df_reult['result'] = y_pred
            df_reult.to_csv(save_result_path, index=False)

    def _kfold(self):
        pass

    def set_params(self):
        pass

    def get_params(self):
        pass

    def save_model(self):
        pass


    def load_model(self):
        pass




def lgb_fit(config, x_train, y_train):
    params = config.params
    print('params %s' % params)
    max_round = config.max_round
    cv_folds = config.cv_folds
    early_stop_round = config.early_stop_round
    seed = config.seed
    save_model_path = config.save_model_path

    if cv_folds is not None:
        print('cross_validation')
        d_train = lgb.Dataset(x_train, label=y_train)
        cv_result = lgb.cv(params,
                           d_train,
                           max_round,
                           nfold=cv_folds,
                           seed=seed,
                           verbose_eval=True,
                           metrics=['multi_error', 'multi_logloss'],
                           early_stopping_rounds=early_stop_round,
                           show_stdv=False)
        print('cv_result %s' % cv_result)
        print('type_cv_result %s' % type(cv_result))
        best_round = len(cv_result['multi_error-mean'])
        best_auc = cv_result['multi_error-mean'][-1]
        best_model = lgb.train(params, d_train, best_round)

    else:
        print('non_cross_validation')
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=100)
        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid)
        watchlist = [d_train, d_valid]
        best_model = lgb.train(params, d_train, valid_sets=watchlist, early_stopping_rounds=early_stop_round)
        best_round = best_model.best_iteration
        best_auc = best_model.best_score
        cv_result = None

    if save_model_path:
        pass
        # check_path(save_model_path)
        # best_model.save_model(save_model_path)
    return best_model, best_auc, best_round, cv_result


def lgb_predict(model, x_test, y_test, save_result_path=None):
    if conf.params['objective'] == "multiclass":
        y_pred = model.predict(x_test).argmax(axis=1)
        print(y_pred)
        print(y_test)
        result = y_test.reshape(1, -1) == y_pred
        print('the accuracy:\t', float(np.sum(result)) / len(y_pred))
    else:
        # 输出概率
        y_pred_prob = model.predict(x_test)
        y_pred = y_pred_prob
    if save_result_path:
        df_reult = pd.DataFrame()
        df_reult['result'] = y_pred
        df_reult.to_csv(save_result_path, index=False)


def run_feat_search(X_train, X_test, y_train, feature_names):
    pass


def run_cv(x_train, x_test, y_test, y_train):
    conf.lgb_config_c()
    tic = time.time()
    data_message = 'x_train.shape={}, x_test.shape={}'.format(x_train.shape, x_test.shape)
    print(data_message)
    # logger.info(data_message)
    lgb_model, best_score, best_round, cv_result = lgb_fit(conf, x_train, y_train)
    print('Time cost {}s'.format(time.time() - tic))
    result_message = 'best_round={}, best_score={}'.format(best_round, best_score)
    # logger.info(result_message)
    print(result_message)

    # predict
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_lgb_{}-{:.4f}.csv'.format(now, best_score)
    # check_path(result_path)
    lgb_predict(lgb_model, x_test, y_test, save_result_path=None)


if __name__ == '__main__':
    import lightgbm as lgb
    import pandas as pd

    iris = load_iris()
    data = iris.data
    target = iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

    gbm = lgb.LGBMRegressor(learning_rate=0.03, n_estimators=200, max_depth=8)
    gbm.fit(X_train, y_train)

    # 预测结果
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    print(y_pred)


