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