#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 0.1
@author: li
@file: bst_model.py
@time: 2019/10/30 10:16 上午
"""

from sqlalchemy import Column, String, Float, Integer, TIMESTAMP, Date
from sqlalchemy.ext.declarative import declarative_base

# 映射对象的基类
Base = declarative_base()


class Record(Base):
    __tablename__ = 'pos_record'
    trade_date = Column(Date(), primary_key=True)
    adjust_date = Column(Date())
    weight = Column(Float(20))
    industry = Column(String(20))
    er = Column(Float(20))
    code = Column(String(20))


class RecordBack(Base):
    __tablename__ = 'pos_record_back'
    trade_date = Column(Date(), primary_key=True)
    adjust_date = Column(Date())
    weight = Column(Float(20))
    industry = Column(String(20))
    er = Column(Float(20))
    code = Column(String(20))


