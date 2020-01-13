#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: sqlengine.py
@time: 2019-06-14 10:40
"""
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import pandas as pd
import sqlalchemy as sa
import sqlalchemy.orm as orm
from data.engines.model import Record2
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

    def fetch_record_meta(self, module, trade_date=None):
        """
        :param module:
        :param trade_date:
        :return:
        """
        if type(trade_date) == str:
            trade_date = datetime.strptime(trade_date, '%Y-%m-%d')
        else:
            trade_date = trade_date

        if trade_date:
            query = self.session.query(module).filter(module.trade_date == trade_date)
        else:
            query = self.session.query(module)
        return pd.read_sql(query.statement, query.session.bind)

    def fetch_record(self, table_name, chunk_size=10000):
        df_list = []
        for chunk in pd.read_sql(table_name, self.engine, chunksize=chunk_size):
            df_list.append(chunk)
        if len(df_list) <= 0:
            df_data = pd.DataFrame()
        else:
            df_data = pd.concat(df_list, ignore_index=True)
        return df_data

    def fetch_data(self, db_sql):
        records = self.session.execute(db_sql)
        return records

    def del_historical_data(self, module, trade_date):
        """
        删除某张表中指定日期的数据
        :param module:
        :param trade_date:
        :return:
        """
        if type(trade_date) == str:
            trade_date = datetime.strptime(trade_date, '%Y-%m-%d')
        else:
            trade_date = trade_date
        # self.session.query(Record).filter(Record.trade_date == trade_date).delete()
        self.session.query(module).filter(module.trade_date == trade_date).delete()
        # self.session.execute('''delete from `{0}` where trade_date=\'{1}\''''.format(table_name, trade_date))
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            # 输出异常信息
            print("del_historical_data(): ======={}=======".format(e))
        finally:
            self.session.close()

    def write_data(self, table_name, df_data: pd.DataFrame, if_exists='append'):
        """
        数据写入对应的表中
        :param if_exists:
        :param table_name:
        :param df_data:
        :return:
        """
        if if_exists == 'replace':
            df_data.to_sql(table_name, self.engine, index=False, if_exists='replace', chunksize=100)
        else:
            df_data.to_sql(table_name, self.engine, index=False, if_exists='append', chunksize=100)


if __name__ == '__main__':
    engine = SQLEngine('sqlite:////Users/li/PycharmProjects/MultiFactors/src/stacking/notebooks/cross_section/RealOperation/s1/real_tune_record_without_alpha-Copy1.db')
    date = datetime.strptime('2019-12-18', '%Y-%m-%d')
    data2 = engine.fetch_record_meta(Record2, trade_date=None)
    print(data2)
    # print(data2[data2['trade_date'] == date])

    engine.del_historical_data(Record2, date)
    # data = engine.fetch_record('pos_record')
    data2 = engine.fetch_record_meta(Record2, trade_date=None)

    # print(data[data['trade_date'] == date])
    print(data2)

    # data = engine.fetch_data('''select * from pos_record where trade_date=\'{}\''''.format(date))
    # print(data.first())
    # >>> (1.6203373031145196e-10, '房地产', 0.0028628408908843994, 6, '2019-10-15 00:00:00')
