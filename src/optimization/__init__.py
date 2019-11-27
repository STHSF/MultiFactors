#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: __init__.py.py
@time: 2019-05-08 15:01
"""


# # 父类
# class Dog:
#     @staticmethod
#     def bark(name):
#         print(name)
#         return name
#
#     def eat(self, name):
#         return self.bark()
#
#
# # 子类 继承
# class XiaoTianQuan(Dog):
#
#     # 可以重写父类中的同名方法
#     def bark(self):
#         name = 'xiaotianquan'
#         print("神一样的叫唤...")
#         res = super().bark(name)
#         return res
#
#
# xtq = XiaoTianQuan()
# print(xtq.eat('jijiji'))


a = {'colsample_bytree': 0.7270718895051846, 'gamma': 6.942242690087745, 'max_delta_step': 0.8699125900359417, 'max_depth': 5.339221068023104, 'min_child_weight': 8.510226254442712, 'subsample': 0.8733565776551393}

a['colsample_bytree'] = int(a['colsample_bytree'])
print(a)