#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: m3_svm.py
@time: 2019/11/20 8:35 下午
"""

from src.stacking.models.libsvm.python.svmutil import *

y, x = svm_read_problem('./libsvm/heart_scale')

m = svm_train(y[:200], x[:200], '-t 2 -c 4')
p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)