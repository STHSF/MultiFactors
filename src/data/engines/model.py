#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: model.py
@time: 2019/10/30 10:16 上午
"""

from sqlalchemy import Column, String, Float, Integer, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base

# 映射对象的基类
Base = declarative_base()


class Record(Base):
    __tablename__ = 'pos_record'
    # id = Column(String(20), primary_key=True)  # 标识为主键
    trade_date = Column(TIMESTAMP, primary_key=True)
    weight = Column(Float(20))
    industry = Column(String(20))
    er = Column(Float(20))
    code = Column(Integer)

