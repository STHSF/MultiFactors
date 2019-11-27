#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: bayes_opt.py
@time: 2019/11/26 4:23 下午
"""
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import warnings
import pandas as pd
import lightgbm as lgb
from src.utils import log_util
from src.optimization.bayes_optimization_base import BayesOptimizationBase

log = log_util.Logger('BayesOptimizationLightGBM', level='info')


class BayesOptimizationLGBM(BayesOptimizationBase):
    """
    基于贝叶斯优化的lightGBM参数寻优过程
    注意不同的eval_metric使用的best_score不一样，需要自己调整。
    """

    def __init__(self, X_train, y_train, X_valid=None, y_valid=None, kfolds=None):
        """
        init
        :param X_train: train target
        :param y_train: train label
        :param X_test: test target
        :param y_test: test label
        :param kfolds: cv folds
        """
        super(BayesOptimizationLGBM, self).__init__()
        self.BestScore = 1.  # best_score保存的时候需要注意选用的eval_metric，一般都是指标越小越好， 如果是auc，则是越大越好
        self.BestIter = 0.
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.folds = kfolds
        self.max_round = 2000
        self.early_stop_round = 100

    def lgb_cv(self, max_depth, num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, lambda_l1, lambda_l2):
        """
        XGBoost model with NonCrossValidation
        :param max_depth:
        :param lambda_l2:
        :param lambda_l1:
        :param bagging_fraction:
        :param feature_fraction:
        :param min_data_in_leaf:
        :param num_leaves:
        :param max_depth:
        :return:
        """

        d_train = lgb.Dataset(self.X_train, label=self.y_train)
        params = {'task': 'train',
                  'boosting': 'gbdt',
                  'objective': 'multiclass',
                  'num_class': 3,
                  'metric': ['multi_error', 'multi_logloss'],
                  'learning_rate': 0.01,
                  "bagging_freq": 1,
                  "verbosity": -1,
                  'num_leaves': int(num_leaves),
                  'min_data_in_leaf': int(min_data_in_leaf),
                  'max_depth': int(max_depth),
                  "feature_fraction": feature_fraction,
                  "bagging_fraction": bagging_fraction,
                  "bagging_seed": 11,
                  "lambda_l1": lambda_l1,
                  "lambda_l2": lambda_l2,
                  }

        cv_result = lgb.cv(params,
                           d_train,
                           num_boost_round=self.max_round,
                           nfold=self.folds,
                           verbose_eval=True,
                           early_stopping_rounds=self.early_stop_round,
                           show_stdv=False)

        # train_score = xgbc['train-merror-mean'].iloc[-1]
        best_round = len(cv_result['multi_error-mean'])
        val_score = pd.Series(cv_result['multi_error-mean']).min()

        log.logger.info('Stopped after %d iterations with train-score = %f val-gini = %f' %
                        (best_round, val_score, (val_score * 2 - 1)))

        if val_score < self.BestScore:
            # multi_error指标越小越好，使用AUC则是指标越大越好
            self.BestScore = val_score
            self.BestIter = best_round
        return (val_score * 2) - 1

    def lgb_no(self, max_depth, num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, lambda_l1, lambda_l2):
        """
        XGBoost model with NonCrossValidation
        :param lambda_l2:
        :param lambda_l1:
        :param bagging_fraction:
        :param feature_fraction:
        :param min_data_in_leaf:
        :param num_leaves:
        :param max_depth:
        :return:
        """
        d_train = lgb.Dataset(self.X_train, label=self.y_train)
        if self.X_valid is not None and self.y_valid is not None:
            d_valid = lgb.Dataset(self.X_valid, label=self.y_valid)
            watchlist = [d_train, d_valid]
        else:
            watchlist = [d_train]

        params = {'task': 'train',
                  'boosting': 'gbdt',
                  'objective': 'multiclass',
                  'num_class': 3,
                  'metric': ['multi_error', 'multi_logloss'],
                  'learning_rate': 0.01,
                  "bagging_freq": 1,
                  "verbosity": -1,
                  'num_leaves': int(num_leaves),
                  'min_data_in_leaf': int(min_data_in_leaf),
                  'max_depth': int(max_depth),
                  "feature_fraction": feature_fraction,
                  "bagging_fraction": bagging_fraction,
                  "bagging_seed": 11,
                  "lambda_l1": lambda_l1,
                  "lambda_l2": lambda_l2,
                  }
        best_model = lgb.train(params,
                               d_train,
                               num_boost_round=self.max_round,
                               valid_sets=watchlist,
                               early_stopping_rounds=self.early_stop_round)
        best_round = best_model.best_iteration
        # 不同的metric可能有不同的best_score类型，使用时需要注意。
        best_score = best_model.best_score['valid_1']['multi_logloss']
        log.logger.info(' Stopped after %d iterations with train-score = %f ' % (best_round, best_score))
        if best_score < self.BestScore:
            # m_error指标越小越好，使用AUC则是指标越大越好
            self.BestScore = best_score
            self.BestIter = best_round
        return (best_score * 2) - 1

    def train_opt(self, parameters, gp_params=None):
        """
        贝叶斯优化模型训练
        :param parameters: 待优化参数
        :param gp_params: 优化模型参数
        :return:
        """
        if gp_params is None:
            # gp_params = {"init_points": 10, "n_iter": 2, "acq": 'ucb', "xi": 0.0, "alpha": 1e-4, "kappa": 10}
            # gp_params = {"init_points": 10, "n_iter": 50, "acq": 'ucb', "xi": 0.0, "alpha": 1e-4, "kappa": 10}
            # gp_params = {"init_points": 2, "n_iter": 50, "acq": 'ei', "xi": 0.0, "alpha": 1e-4}
            gp_params = {"init_points": 2, "n_iter": 2, "acq": 'ei', "xi": 0.0, "alpha": 1e-4}

        if self.folds is None:
            self.function = self.lgb_no
        else:
            self.function = self.lgb_cv

        params_opt = super().train_opt(parameters, gp_params)
        # 注意优化参数的实际取值范围, 与需要优化的参数一一对应即可
        params_opt['max_depth'] = int(params_opt['max_depth'])
        params_opt['num_leaves'] = int(params_opt['num_leaves'])
        params_opt['min_data_in_leaf'] = int(params_opt['min_data_in_leaf'])
        return params_opt


if __name__ == '__main__':
    # Classify Parameter Optimization Test
    log.logger.info('Classify Parameter Optimization Test')
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    data = iris.data
    target = iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
    log.logger.info('{},{},{},{}'.format(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test)))

    opt_parameters = {'max_depth': (4, 10),
                      'num_leaves': (5, 130),
                      'min_data_in_leaf': (10, 150),
                      'feature_fraction': (0.7, 1.0),
                      'bagging_fraction': (0.7, 1.0),
                      'lambda_l1': (0, 6),
                      'lambda_l2': (0, 6)
                      }

    gp_params = {"init_points": 2, "n_iter": 2, "acq": 'ei', "xi": 0.0, "alpha": 1e-4}
    opt_lgb = BayesOptimizationLGBM(X_train, y_train, X_test, y_test)
    params_op = opt_lgb.train_opt(opt_parameters, gp_params)
    log.logger.info('BestScore: {}, BestIter: {}'.format(opt_lgb.BestScore, opt_lgb.BestIter))
