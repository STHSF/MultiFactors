#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: demo.py
@time: 2020/1/14 11:13 上午
"""


# from hyperopt import fmin, tpe, hp
#
#
# def fuc(params):
#     x = params['x']
#     y = params['y']
#
#     return x ** 2 - y
#
#
# best = fmin(fn=fuc,
#             space={'x': hp.uniform('x', -10, 10),
#                    'y': hp.uniform('y', -10, 10)},
#             algo=tpe.suggest,
#             max_evals=100)
#
#
# print(best)


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt

fspace = {
    'x': hp.uniform('x', -5, 5),
    'y': hp.uniform('y', -5, 5)
}


def f(params):
    x = params['x']
    y = params['y']

    val = x**2 - y
    return {'loss': val, 'status': STATUS_OK}


trials = Trials()
best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=50, trials=trials)

print('best:', best)

# f, ax = plt.subplots(1)
# xs = [t['tid'] for t in trials.trials]
# ys = [t['misc']['vals']['x'] for t in trials.trials]
# ax.set_xlim(xs[0]-10, xs[-1]+10)
# ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
# ax.set_title('$x$ $vs$ $t$ ', fontsize=18)
# ax.set_xlabel('$t$', fontsize=16)
# ax.set_ylabel('$x$', fontsize=16)
# f.show()


f, ax = plt.subplots(1)
xs = [t['misc']['vals']['x'] for t in trials.trials]
ys = [t['result']['loss'] for t in trials.trials]
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
ax.set_title('$val$ $vs$ $x$ ', fontsize=18)
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$val$', fontsize=16)
f.show()

