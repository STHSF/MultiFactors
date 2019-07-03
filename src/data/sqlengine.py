#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: sqlengine.py
@time: 2019-06-14 10:40
"""

import sqlalchemy as sa
from sqlalchemy import orm


class SqlEngine(object):
    def __int__(self, db_url):
        self.engine = sa.create_engine(db_url)
        self.session = self.create_session()

    def create_session(self):
        session = orm.sessionmaker(bind=self.engine)
        return session
