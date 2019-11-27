#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: __init__.py.py
@time: 2019-05-08 15:01
"""

# 父类
class Dog:

    @staticmethod
    def bark(name):
        print(name)
        return name
# 子类 继承
class XiaoTianQuan(Dog):
    def fly(self):
        print("我会飞")
# 可以重写父类中的同名方法

    def bark(self):
        name = 'kk'
        print("神一样的叫唤...")
        res = super().bark(name)
        return res



xtq = XiaoTianQuan()
print(xtq.bark())