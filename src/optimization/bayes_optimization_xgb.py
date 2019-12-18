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
import xgboost as xgb
from utils import log_util
from optimization.bayes_optimization_base import BayesOptimizationBase

log = log_util.Logger('BayesOptimizationXGBoost', level='info')


class BayesOptimizationXGB(BayesOptimizationBase):
    """
    基于贝叶斯优化的XGBoost参数寻优过程
    注意不同的eval_metric使用的best_score不一样，需要自己调整。
    同样注意贝叶斯优化为最大化目标值，所以在选取best_score的指标时，需要注意方向。
    """

    def __init__(self, opt_type, X_train, y_train, X_test=None, y_test=None, kfolds=None):
        """
        init
        :param X_train: train target
        :param y_train: train label
        :param X_test: test target
        :param y_test: test label
        :param kfolds: cv folds
        """
        super(BayesOptimizationXGB, self).__init__()
        self.BestScore = 1.  # best_score保存的时候需要注意选用的eval_metric，一般都是指标越小越好， 如果是auc，则是越大越好
        self.BestIter = 0.
        self.opt_type = opt_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.folds = kfolds
        self.max_round = 1000
        self.early_stop_round = 100

        self.params = {}
        if self.opt_type == 'multi':
            self.params = {'objective': 'multi:softmax',
                           'num_class': 3,
                           "eval_metric": ["mlogloss", "merror"],
                           }
        elif self.opt_type == 'regression':
            self.params = {'task': 'train',
                           'objective': 'reg:linear',  # 目标函数
                           'booster': 'dart',  # 设置提升类型
                           'eval_metric': ['rmse', 'logloss'],
                           }
        else:
            log.logger.error('请指定lightGBM模型的类型')
            exit()

    def xgb_cv(self, max_depth, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree):
        """
        XGBoost model with CrossValidation
        :param max_depth:
        :param gamma:
        :param min_child_weight:
        :param max_delta_step:
        :param subsample:
        :param colsample_bytree:
        :return:
        """
        opt_params: dict = {'eta': 0.1,
                            'nthread': 4,
                            'silent': 0,
                            'max_depth': int(max_depth),
                            'gamma': int(gamma),
                            'subsample': max(min(subsample, 1), 0),
                            'colsample_bytree': max(min(colsample_bytree, 1), 0),
                            'min_child_weight': int(min_child_weight),
                            'max_delta_step': int(max_delta_step),
                            'cv_seed': 30}
        self.params.update(opt_params)

        data_train = xgb.DMatrix(self.X_train, label=self.y_train)
        cv_result = xgb.cv(self.params,
                           data_train,
                           num_boost_round=self.max_round,
                           stratified=True,
                           nfold=self.folds,
                           early_stopping_rounds=self.early_stop_round,
                           verbose_eval=True,
                           show_stdv=True)

        log.logger.info('params: \n{}'.format(self.params))
        val_score = cv_result['test-mlogloss-mean'].iloc[-1]
        train_score = cv_result['train-mlogloss-mean'].iloc[-1]
        log.logger.info(
            'Stopped after %d iterations with train-score= %f val-score= %f (diff= %f) train-gini= %f '
            'val-gini = %f' % (len(cv_result),
                               train_score,
                               val_score,
                               (train_score - val_score),
                               (train_score * 2 - 1),
                               (val_score * 2 - 1)))

        if val_score < self.BestScore:
            # merror指标越小越好，使用AUC则是指标越大越好
            self.BestScore = val_score
            self.BestIter = len(cv_result)
        return -val_score

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
        opt_params: dict = {'nthread': 4,
                            'silent': 0,
                            'eta': 0.1,
                            'max_depth': int(max_depth),
                            'gamma': int(gamma),
                            'subsample': max(min(subsample, 1), 0),
                            'colsample_bytree': max(min(colsample_bytree, 1), 0),
                            'min_child_weight': int(min_child_weight),
                            'max_delta_step': int(max_delta_step),
                            'cv_seed': 2019
                            }
        self.params.update(opt_params)
        log.logger.info('parameters: \n{}'.format(self.params))

        data_train = xgb.DMatrix(self.X_train, label=self.y_train)
        if self.X_test is not None and self.y_test is not None:
            data_test = xgb.DMatrix(self.X_test, label=self.y_test)
            watchlist = [(data_test, 'eval'), (data_train, 'train')]
        else:
            watchlist = [(data_train, 'train')]

        best_model = xgb.train(params=self.params,
                               dtrain=data_train,
                               num_boost_round=self.max_round,
                               evals=watchlist,
                               early_stopping_rounds=self.early_stop_round)
        best_round = best_model.best_iteration
        best_score = best_model.best_score
        log.logger.info('params: \n{}'.format(self.params))
        log.logger.info(' Stopped after %d iterations with train-score = %f train-gini = %f' %
                        (best_round, best_score, (best_score * 2 - 1)))
        if best_score < self.BestScore:
            # m_error指标越小越好，使用AUC则是指标越大越好
            self.BestScore = best_score
            self.BestIter = best_round
        # 注意，贝叶斯优化的目标是最大化best_score, 如果模型的best_score为error或者logloss时，优化目标改为相反数
        return -best_score

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
            self.function = self.xgb_no
        else:
            self.function = self.xgb_cv

        params_opt = super().train_opt(parameters, gp_params)
        # 注意优化参数的实际取值范围, 与需要优化的参数一一对应即可
        params_opt['max_depth'] = int(params_opt['max_depth'])
        params_opt['gamma'] = int(params_opt['gamma'])
        params_opt['min_child_weight'] = int(params_opt['min_child_weight'])
        params_opt['max_delta_step'] = int(params_opt['max_delta_step'])
        params_opt['subsample'] = max(min(params_opt['subsample'], 1), 0)
        params_opt['colsample_bytree'] = max(min(params_opt['colsample_bytree'], 1), 0)
        return params_opt


if __name__ == '__main__':
    # Classify Parameter Optimization Test
    log.logger.info('Classify Parameter Optimization Test')
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris, load_boston
    from sklearn.model_selection import train_test_split
    from src.models.m1_xgb import XGBooster, xgb_predict
    from src.conf.configuration import xgb_conf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', dest='type', type=str, required=True,
                        choices=['classification', 'regression'], help="测试类型", default="regression")
    args = parser.parse_args()
    _type = args.type

    def classify_test():
        # ===========================classify Test start==========================================
        iris = load_iris()
        data = iris.data
        target = iris.target
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
        log.logger.info('{},{},{},{}'.format(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test)))
        xgb_conf.xgb_config_c()

        opt_parameters = {'max_depth': (2, 12),
                          'gamma': (0.001, 10.0),
                          'min_child_weight': (0, 20),
                          'max_delta_step': (0, 10),
                          'subsample': (0.01, 0.99),
                          'colsample_bytree': (0.01, 0.99)}

        gp_params = {"init_points": 2, "n_iter": 20, "acq": 'ei', "xi": 0.0, "alpha": 1e-4}
        opt_xgb = BayesOptimizationXGB('multi', X_train, y_train, X_test, y_test, kfolds=5)
        params_op = opt_xgb.train_opt(opt_parameters, gp_params=None)
        log.logger.info('Best params: \n{}'.format(params_op))
        log.logger.info('BestScore: {}, BestIter: {}'.format(opt_xgb.BestScore, opt_xgb.BestIter))

        # # update hyperparameters
        xgb_conf.params.update(params_op)
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
        opt_parameters = {'max_depth': (2, 12),
                          'gamma': (0.001, 10.0),
                          'min_child_weight': (0, 20),
                          'max_delta_step': (0, 10),
                          'subsample': (0.01, 0.99),
                          'colsample_bytree': (0.01, 0.99)}
        # 贝叶斯优化
        gp_params = {"init_points": 2, "n_iter": 20, "acq": 'ei', "xi": 0.0, "alpha": 1e-4}
        opt_xgb = BayesOptimizationXGB('regression', X_train, y_train, X_test, y_test)
        params_op = opt_xgb.train_opt(opt_parameters, gp_params=None)
        log.logger.info('Best params: \n{}'.format(params_op))
        log.logger.info('BestScore: {}, BestIter: {}'.format(opt_xgb.BestScore, opt_xgb.BestIter))
        # update hyperparameters
        xgb_conf.params.update(params_op)

        # train model
        xgbc = XGBooster(xgb_conf)
        best_score, best_round, best_model = xgbc.fit(X_train, y_train)
        # eval
        xgb_predict(best_model, xgb_conf, X_test, y_test)
        xgbc.plot_feature_importance(best_model)
        # ===========================REGRESSION TEST END==========================================

    if _type == 'classification':
        classify_test()
    if _type == 'regression':
        regression_test()
