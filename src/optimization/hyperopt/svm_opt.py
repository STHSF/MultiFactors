#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: svm_opt.py
@time: 2020/1/14 2:43 下午
"""

from sklearn import datasets
from sklearn.preprocessing import normalize, scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# 这里的warnings实在太多了，我们加入代码不再让其显示
import warnings
warnings.filterwarnings("ignore")
from hyperopt import fmin, tpe, hp

from hyperopt import Trials
iris = datasets.load_iris()
X = iris.data
y = iris.target


def hyperopt_model_score_svm(params):
    X_ = X[:]

    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
        del params['normalize']

    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
        del params['scale']

    clf = SVC(**params)
    return cross_val_score(clf, X_, y).mean()


space_svm = {
    'C': hp.uniform('C', 0, 20),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
    'gamma': hp.uniform('gamma', 0, 20),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
}


def f_svm(params):
    acc = hyperopt_model_score_svm(params)
    return -acc


trials = Trials()
best = fmin(f_svm, space_svm, algo=tpe.suggest, max_evals=1000, trials=trials)
print('best:')
print(best)

parameters = ['C', 'kernel', 'gamma', 'scale', 'normalize']
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20, 5))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    axes[i].scatter(
        xs,
        ys,
        s=20,
        linewidth=0.01,
        alpha=0.25,
        c=cmap(float(i) / len(parameters)))
    axes[i].set_title(val)
    axes[i].set_ylim([0.9, 1.0])

plt.show()