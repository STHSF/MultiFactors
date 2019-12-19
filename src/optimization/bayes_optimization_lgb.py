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
import pandas as pd
import lightgbm as lgb
from src.utils import log_util
from src.optimization.bayes_optimization_base import BayesOptimizationBase

log = log_util.Logger('BayesOptimizationLightGBM', level='info')


class BayesOptimizationLGBM(BayesOptimizationBase):
    """
    基于贝叶斯优化的lightGBM参数寻优过程
    注意不同的eval_metric使用的best_score不一样，需要自己调整。
    同样注意贝叶斯优化为最大化目标值，所以在选取best_score的指标时，需要注意方向。
    """

    def __init__(self, opt_type, X_train, y_train, X_valid=None, y_valid=None, kfolds=None):
        """
        init
        :param opt_type: lightGBM模型类型; {'classification', }
        :param X_train: train target
        :param y_train: train label
        :param X_valid: test target
        :param y_valid: test label
        :param kfolds: cv folds
        """
        super(BayesOptimizationLGBM, self).__init__()
        self.opt_type = opt_type
        self.BestScore = 1.  # best_score保存的时候需要注意选用的eval_metric，一般都是指标越小越好， 如果是auc，则是越大越好
        self.BestIter = 0.
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.folds = kfolds
        self.max_round = 1000
        self.early_stop_round = 100

        self.params = {}
        if self.opt_type == 'multiclass':
            self.params = {'task': 'train',
                           'boosting': 'gbdt',
                           'objective': 'multiclass',
                           'num_class': 3,
                           'metric': ['multi_error', 'multi_logloss'],
                           }

        elif self.opt_type == 'regression':
            self.params = {'task': 'train',
                           'boosting': 'gbdt',  # 设置提升类型
                           'objective': 'regression',  # 目标函数
                           'metric': {'l2', 'mean_squared_error'},  # 评估函数
                           }
        else:
            log.logger.error('请指定lightGBM模型的类型')
            exit()

    def lgb_cv(self, max_depth, num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, lambda_l1, lambda_l2):
        """
        LightGBM model with NonCrossValidation
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
        opt_params = {'learning_rate': 0.01,
                      'verbosity': -1,
                      'num_leaves': int(num_leaves),
                      'min_data_in_leaf': int(min_data_in_leaf),
                      'max_depth': int(max_depth),
                      "feature_fraction": feature_fraction,
                      "bagging_fraction": bagging_fraction,
                      "bagging_seed": 11,
                      "bagging_freq": 1,
                      "lambda_l1": lambda_l1,
                      "lambda_l2": lambda_l2,
                      }
        self.params.update(opt_params)

        d_train = lgb.Dataset(self.X_train, label=self.y_train)
        cv_result = lgb.cv(self.params,
                           d_train,
                           num_boost_round=self.max_round,
                           nfold=self.folds,
                           verbose_eval=False,
                           early_stopping_rounds=self.early_stop_round,
                           show_stdv=True)

        log.logger.info('cv_result: \n{}'.format(cv_result))
        best_round = len(cv_result['multi_logloss-mean'])
        val_score = pd.Series(cv_result['multi_logloss-mean']).min()
        log.logger.info('parameters: \n{}'.format(self.params))
        log.logger.info('Stopped after %d iterations with train-score = %f val-gini = %f' %
                        (best_round, val_score, (val_score * 2 - 1)))

        if val_score < self.BestScore:
            # multi_error指标越小越好，使用AUC则是指标越大越好
            self.BestScore = val_score
            self.BestIter = best_round
        return -val_score

    def lgb_no(self, max_depth, num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, lambda_l1, lambda_l2):
        """
        LightGBM model with NonCrossValidation
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
        if self.X_valid is None or self.y_valid is None:
            watchlist = [d_train]
        else:
            d_valid = lgb.Dataset(self.X_valid, label=self.y_valid)
            watchlist = [d_train, d_valid]

        opt_params = {'max_bin': 63,  # 表示 feature 将存入的 bin 的最大数量
                      'metric_freq': 1,
                      'learning_rate': 0.05,
                      'verbosity': -1,
                      'num_leaves': int(num_leaves),
                      'min_data_in_leaf': int(min_data_in_leaf),
                      'max_depth': int(max_depth),
                      "feature_fraction": feature_fraction,
                      "bagging_fraction": bagging_fraction,
                      'bagging_freq': 5,
                      "bagging_seed": 11,
                      "lambda_l1": lambda_l1,
                      "lambda_l2": lambda_l2,
                      }
        self.params.update(opt_params)
        best_model = lgb.train(self.params,
                               d_train,
                               num_boost_round=self.max_round,
                               valid_sets=watchlist,
                               early_stopping_rounds=self.early_stop_round,
                               verbose_eval=False,
                               )

        best_round = best_model.best_iteration
        # 不同的metric可能有不同的best_score类型，使用时需要注意。
        score = list(self.params['metric'])[0]
        log.logger.info('best_score: \n{}'.format(best_model.best_score))
        best_score = best_model.best_score['training'][score]
        log.logger.info('parameters: \n{}'.format(self.params))
        log.logger.info(' Stopped after %d iterations with train-score = %f ' % (best_round, best_score))
        if best_score < self.BestScore:
            # m_error指标越小越好，使用AUC则是指标越大越好
            self.BestScore = best_score
            self.BestIter = best_round
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
    # Parameter Optimization Test
    log.logger.info('Parameter Optimization Test')
    import numpy as np
    from sklearn.datasets import load_boston, load_iris
    from sklearn.model_selection import train_test_split
    from src.conf.configuration import lgb_conf
    from src.models.m2_lgb import LightGBM, lgb_predict
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', dest='type', type=str, required=True,
                        choices=['classification', 'regression'], help="测试类型", default="regression")
    args = parser.parse_args()
    _type = args.type

    def classify_test():
        # #===========================classify Test start==========================================
        iris = load_iris()
        data = iris.data
        target = iris.target
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
        log.logger.info('Detail of IRIS: {},{},{},{}'.format(np.shape(X_train), np.shape(X_test),
                                                             np.shape(y_train), np.shape(y_test)))
        lgb_conf.lgb_config_c()
        log.logger.info('Model Params before Optimization:\n{}'.format(lgb_conf.params))
        # Hyper Parameters Optimization
        # 超参
        opt_parameters = {'max_depth': (4, 10),
                          'num_leaves': (5, 130),
                          'min_data_in_leaf': (10, 150),
                          'feature_fraction': (0.1, 1.0),
                          'bagging_fraction': (0.1, 1.0),
                          'lambda_l1': (0, 10),
                          'lambda_l2': (0, 10)
                          }
        # 贝叶斯优化参数设置
        gp_params = {"init_points": 10, "n_iter": 50, "acq": 'ei', "xi": 0.0, "alpha": 1e-4}
        # 贝叶斯优化
        opt_lgb = BayesOptimizationLGBM('multiclass', X_train, y_train, X_test, y_test)
        params_op = opt_lgb.train_opt(opt_parameters, gp_params)
        log.logger.info('Best params: \n{}'.format(params_op))
        log.logger.info('BestScore: {}, BestIter: {}'.format(opt_lgb.BestScore, opt_lgb.BestIter))
        lgb_conf.params.update(params_op)
        log.logger.info('Model Params after Optimization:\n{}'.format(lgb_conf.params))

        # NonCrossValidation Test
        lgbm = LightGBM(lgb_conf)
        best_model, best_score, best_round = lgbm.fit(X_train, y_train)

        # eval
        lgb_predict(best_model, X_test, y_test, lgb_conf)

        # feature important
        lgbm.plot_feature_importance(best_model)
        # #===========================classify Test end==========================================

    def regression_test():
        # #===========================REGRESSION TEST START==========================================
        boston = load_boston()
        data = boston.data
        target = boston.target
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
        log.logger.info('Detail of Boston Data: {},{},{},{}'.format(np.shape(X_train), np.shape(X_test),
                                                                    np.shape(y_train), np.shape(y_test)))
        lgb_conf.lgb_config_r()
        log.logger.info('Model Params before Optimization:\n{}'.format(lgb_conf.params))

        opt_parameters = {'max_depth': (4, 10),
                          'num_leaves': (5, 130),
                          'min_data_in_leaf': (10, 150),
                          'feature_fraction': (0.1, 1.0),
                          'bagging_fraction': (0.1, 1.0),
                          'lambda_l1': (0, 10),
                          'lambda_l2': (0, 10)
                          }

        gp_params = {"init_points": 2, "n_iter": 20, "acq": 'ei', "xi": 0.0, "alpha": 1e-4}
        opt_lgb = BayesOptimizationLGBM('regression', X_train, y_train, X_test, y_test)
        params_op = opt_lgb.train_opt(opt_parameters, gp_params=None)
        log.logger.info('Best params: \n{}'.format(params_op))
        log.logger.info('BestScore: {}, BestIter: {}'.format(opt_lgb.BestScore, opt_lgb.BestIter))
        # # Update HyperParameters
        lgb_conf.params.update(params_op)

        # train model
        lgb_m = LightGBM(lgb_conf)
        best_model, best_score, best_round = lgb_m.fit(X_train, y_train)

        # eval
        lgb_predict(best_model, lgb_conf, X_test, y_test)

        # feature important
        lgb_m.plot_feature_importance(best_model)
        # #===========================REGRESSION TEST END==========================================

    if _type == 'classification':
        classify_test()
    if _type == 'regression':
        regression_test()
