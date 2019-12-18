#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: m2_lgb.py
@time: 2019-03-04 11:03
"""
import os
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
import time
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.conf.configuration import lgb_conf
from src.utils import log_util
from src.utils.Evaluation import cls_eva, reg_eva


log = log_util.Logger('LightGBM', level='info')


class LightGBM(object):
    def __init__(self, args):
        self.params = args.params
        self.max_round = args.max_round
        self.cv_folds = args.cv_folds
        self.early_stop_round = args.early_stop_round
        self.cv_seed = args.cv_seed
        self.save_model_path = args.save_model_path

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        best_model = None
        best_round = None
        best_score = {}
        if self.cv_folds is None:
            log.logger.info('NonCrossValidation。。。。')

            # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
            if x_valid is None and y_valid is None:
                d_train = lgb.Dataset(x_train, label=y_train)
                watchlist = [d_train]
            else:
                d_train = lgb.Dataset(x_train, label=y_train)
                d_valid = lgb.Dataset(x_valid, label=y_valid)
                watchlist = [d_train, d_valid]

            best_model = lgb.train(self.params,
                                   d_train,
                                   num_boost_round=self.max_round,
                                   valid_sets=watchlist,
                                   early_stopping_rounds=self.early_stop_round,
                                   verbose_eval=False,
                                   )
            best_round = best_model.best_iteration
            best_score['best_score'] = best_model.best_score
            log.logger.info('best_score: \n{}'.format(best_model.best_score))

        else:
            log.logger.info('CrossValidation ........')
            d_train = lgb.Dataset(x_train, label=y_train)
            if self.params['objective'] is 'multiclass':
                cv_result = self._kfold(d_train)
                log.logger.info('cv_result %s' % cv_result)
                log.logger.info('type_cv_result %s' % type(cv_result))
                if 'multi_error' in self.params['metric']:
                    best_round = len(cv_result['multi_error-mean'])
                    best_score['min_multi_error-mean'] = pd.Series(cv_result['multi_error-mean']).min()

                elif 'multi_logloss' in self.params['metric']:
                    best_score['min_multi_logloss-mean'] = pd.Series(cv_result['multi_logloss-mean']).min()

                best_model = lgb.train(self.params, d_train, best_round)

            elif self.params['objective'] is 'regression':
                cv_result = self._kfold(d_train)
                log.logger.info('cv_result %s' % cv_result)
                log.logger.info('type_cv_result %s' % type(cv_result))
                if 'l2' in self.params['metric']:
                    score = pd.Series(cv_result['l2-mean']).min()
                    best_round = pd.Series(cv_result['l2-mean']).idxmin()
                    # best_round = cv_result[cv_result['l2-mean'].isin([score])].index[0]
                    best_score['min_l2-mean'] = score

                elif 'rmse' in self.params['metric']:
                    score = pd.Series(cv_result['test-rmse-mean']).min()
                    best_round = pd.Series(cv_result['test-rmse-mean']).idxmin()
                    # best_round = cv_result[cv_result['test-rmse-mean'].isin([score])].index[0]
                    best_score['min_test-rmse-mean'] = score

                elif 'auc' in self.params['metric']:
                    best_score['max_auc-mean'] = pd.Series(cv_result['auc-mean']).max()
                    best_round = pd.Series(cv_result['auc-mean']).idxmax()

                best_model = lgb.train(self.params, d_train, best_round)
            else:
                print('ERROR: LightGBM OBJECTIVE IS NOT CLASSIFY OR REGRESSION')
                exit()
        return best_model, best_score, best_round

    def predict(self, bst_model, x_test, save_result_path=None):
        if self.params['objective'] == "multiclass":
            pre_data = bst_model.predict(x_test).argmax(axis=1)
        else:
            pre_data = bst_model.predict(x_test)

        if save_result_path:
            df_reult = pd.DataFrame()
            df_reult['result'] = pre_data
            df_reult.to_csv(save_result_path, index=False)

        return pre_data

    def _kfold(self, d_train):
        cv_result = lgb.cv(self.params,
                           d_train,
                           num_boost_round=self.max_round,
                           nfold=self.cv_folds,
                           seed=self.cv_seed,
                           verbose_eval=False,
                           early_stopping_rounds=self.early_stop_round,
                           show_stdv=False)
        return cv_result

    @staticmethod
    def plot_feature_importance(best_model, top_n=20):
        print(80 * '*')
        print(31 * '*' + 'Feature Importance' + 31 * '*')
        print(80 * '.')
        print("\n".join((".%50s => %9.5f" % x) for x in sorted(zip(best_model.feature_name(), best_model.feature_importance("gain")),
                                                               key=lambda x: x[1],
                                                               reverse=True)))
        print(80 * '.')

        feature_importance_df = pd.DataFrame(zip(best_model.feature_name(), best_model.feature_importance("gain")), columns=['Feature', 'importance'])
        best_features = feature_importance_df[["Feature", "importance"]].sort_values(by="importance", ascending=True)
        best_features['importance'] = best_features['importance'] / best_features['importance'].sum()
        # best_features = feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=True)[:top_n].reset_index()
        plt.figure(figsize=(16, 10))
        # # plot feature importance of top_n
        best_features[:top_n].plot(kind='barh', x='Feature', y='importance', legend=False, figsize=(16, 10))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative Features')
        plt.ylabel('Feature Importance Score')
        # save picture
        # plt.gcf().savefig('feature_importance_xgb.png')
        plt.show()

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


def plot_figure(y_pred, y_test, fig_name):
    fig1 = plt.figure(num=fig_name, figsize=(10, 3), dpi=75, facecolor='#FFFFFF', edgecolor='#0000FF')
    plt.plot(y_pred)
    plt.plot(y_test)
    plt.title(u"REGRESSION")
    plt.legend((u'Predict', u'Test'), loc='best')  # sets our legend for our graph.
    plt.show()
    plt.close()


def lgb_predict(bst_model, conf, x_test, y_test, save_result_path=None):
    # x_test = x_test.flatten()
    if conf.params['objective'] == "multiclass":
        y_pred = bst_model.predict(x_test).argmax(axis=1)
        log.logger.info(y_pred)
        log.logger.info(y_test)
        if y_test is not None:
            # AUC计算
            log.logger.info('The Accuracy:\t{}'.format(cls_eva.auc(y_test, y_pred)))

    elif conf.params['objective'] == "regression":
        y_pred = bst_model.predict(x_test)
        if y_test is None:
            log.logger.info('y_pre: {}'.format(y_pred))
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

    if save_result_path:
        if not isinstance(y_pred, pd.DataFrame):
            df_result = pd.DataFrame(y_pred, columns='result')
        else:
            df_result = y_pred
        df_result.to_csv(save_result_path, index=False)


def run_cv(x_train, x_test, y_test, y_train, conf):
    tic = time.time()
    data_message = 'x_train.shape={}, x_test.shape={}'.format(x_train.shape, x_test.shape)
    log.logger.info(data_message)

    lgb = LightGBM(conf)
    lgb_model, best_score, best_round = lgb.fit(x_train, y_train)
    log.logger.info('Time cost {}s'.format(time.time() - tic))
    result_message = 'best_round={}, best_score={}'.format(best_round, best_score)
    log.logger.info(result_message)

    # predict
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_lgb_{}-{:.4f}.csv'.format(now, best_round)
    # check_path(result_path)
    lgb_predict(lgb_model, conf, x_test, y_test, save_result_path=None)


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris, load_boston
    from sklearn.model_selection import train_test_split
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", dest="type", default="classification", choices=['classification', 'regression'], type=str, help="输入回归or分类")
    args = parser.parse_args()
    _type = args.type

    def classify_test():
        # # #===========================classify Test start==========================================
        # CLASSIFY TEST
        iris = load_iris()
        data = iris.data
        target = iris.target
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
        log.logger.info('type of x_train: {}'.format(type(X_train)))
        log.logger.info('shape of x_train: {}'.format(np.shape(X_train)))
        lgb_conf.lgb_config_c()
        log.logger.info('Model Params pre:\n{}'.format(lgb_conf.params))

        # # NonCrossValidation Test
        lgbm = LightGBM(lgb_conf)
        best_model, best_score, best_round = lgbm.fit(X_train, y_train)
        lgb_predict(best_model, lgb_conf, X_test, y_test)
        lgbm.plot_feature_importance(best_model)

        # # CrossValidation Test
        # lgb_conf.cv_folds = 5
        # run_cv(X_train, X_test, y_test, y_train, lgb_conf)
        # # #===========================classify Test end==========================================

    def regression_test():
        # #===========================REGRESSION TEST START==========================================
        boston = load_boston()
        data = boston.data
        target = boston.target
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
        log.logger.info('{},{},{},{}'.format(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test)))
        lgb_conf.lgb_config_r()

        # train model
        lgb_m = LightGBM(lgb_conf)
        best_model, best_score, best_round = lgb_m.fit(X_train, y_train)
        # eval
        lgb_predict(best_model, lgb_conf, X_test, y_test)
        lgb_m.plot_feature_importance(best_model)
        print(lgb_m.predict(best_model, X_test))
        # #===========================REGRESSION TEST END==========================================

    if _type == 'regression':
        regression_test()
    if _type == 'classification':
        classify_test()

