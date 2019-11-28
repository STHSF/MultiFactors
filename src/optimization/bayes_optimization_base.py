#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: BayesianOptimization.py
@time: 2019/11/26 4:23 下午
"""
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import warnings
from src.utils import log_util
from bayes_opt import BayesianOptimization


class BayesOptimizationBase(object):
    """
    BayesOptimizationTool
    """
    def __init__(self):
        self.function = self.black_box_function

    def black_box_function(self):
        """
        初始化目标函数， 贝叶斯优化会最大化目标函数的返回值，所以在定义优化目标函数的时候注意返回值的方向。
        """
        return

    @staticmethod
    def bayesian_optimization(function, opt_parameters, gp_params):
        """
        BayesianOptimization
        :param function: 目标函数
        :param opt_parameters: 待优化的模型参数
        :param gp_params: 贝叶斯优化模型的参数
        :return:
        """
        bayesian_opt_model = BayesianOptimization(function, opt_parameters)
        bayesian_opt_model.maximize(**gp_params)
        return bayesian_opt_model.max

    def train_opt(self, parameters, gp_params):
        """
        Train Optimization model
        :param parameters: 待优化的模型参数
        :param gp_params: 贝叶斯优化模型的参数
        :return:
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            best_solution = self.bayesian_optimization(self.function, parameters, gp_params)
        params_opt = best_solution["params"]
        return params_opt


if __name__ == '__main__':
    def black_box_function(x, y):
        score = -x ** 2 - (y - 1) ** 2 + 1
        return score

    bayes_opt = BayesOptimizationBase()
    bayes_opt.function = black_box_function
    #
    pbounds = {'x': (2, 4), 'y': (-3, 3)}
    gp_p = {'init_points': 2, 'n_iter': 3}

    params = bayes_opt.train_opt(pbounds, gp_p)
    print(params)

    #
    # | iter | target | x | y |
    # -------------------------------------------------
    # | 1 | -26.31 | 3.61 | -2.779 |
    # | 2 | -12.49 | 3.551 | 0.06221 |
    # | 3 | -25.32 | 3.942 | -2.284 |
    # | 4 | -12.08 | 3.535 | 0.235 |
    # | 5 | -13.65 | 3.773 | 0.3519 |
    # =================================================
    # >>>> {'x': 2.0, 'y': 1.9538635158702904}
