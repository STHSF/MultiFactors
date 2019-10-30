#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: sqlengine.py
@time: 2019-06-14 10:40
"""
import pandas as pd
import sqlalchemy as sa
import sqlalchemy.orm as orm
from src.data.engines.model import Record
from datetime import datetime

class SQLEngine(object):

    def __init__(self, db_url):
        self.engine = sa.create_engine(db_url)
        self.session = self.create_session()

    def __del__(self):
        if self.session:
            self.session.close()

    def create_session(self):
        db_session = orm.sessionmaker(bind=self.engine)
        return db_session()

    def fetch_record_meta(self):
        query = self.session.query(Record)
        return pd.read_sql(query.statement, query.session.bind)

    def fetch_data(self, db_sql):
        # db_sql =
        records = self.session.execute(db_sql)
        return records

    def del_historical_data(self, table_name, trade_date):
        """
        删除某张表中指定日期的数据
        :param table_name:
        :param trade_date:
        :return:
        """
        trade_date = datetime.strptime(trade_date, '%Y-%m-%d')
        self.session.execute('''delete from `{0}` where trade_date=\'{1}\''''.format(table_name, trade_date))
        self.session.commit()
        self.session.close()


if __name__ == '__main__':
    engine = SQLEngine('sqlite:////Users/li/PycharmProjects/MultiFactors/src/stacking/notebooks/real_tune_record.db')

    engine.del_historical_data('pos_record', '2019-10-15')

    data = engine.fetch_record_meta()
    print(data)

    # data = engine.fetch_data('''select * from pos_record where trade_date=\'{}\''''.format(date))
    # print(data.first())
    # >>> (1.6203373031145196e-10, '房地产', 0.0028628408908843994, 6, '2019-10-15 00:00:00')
