#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: log_util.py
@time: 2018-12-26 16:33
"""

import logging
from src.utils import file_util
from logging import handlers
from colorama import Fore, Style


class Logger(object):
    level_relations = {
        # 日志级别 critical > error > warning > info > debug
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self, log_file_name,
                 log_file_path='../log/',
                 level='debug',
                 when='D',
                 backup_count=5,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):

        log_file = log_file_path + '%s.log' % log_file_name
        file_util.check_path(log_file)
        self.logger = logging.getLogger(log_file)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 指定最低的日志级别

        console = logging.StreamHandler()  # 往屏幕上输出
        console.setFormatter(format_str)  # 设置屏幕上显示的格式

        # 实例化TimedRotatingFileHandler
        handler = handlers.TimedRotatingFileHandler(filename=log_file,  # 写入文件目录
                                                    when=when,  # 指定间隔时间自动生成文件的处理器
                                                    backupCount=backup_count,
                                                    encoding='utf-8')
        # interval是时间间隔
        # backupCount是备份文件的个数，如果超过这个个数，就会自动删除，
        # when是间隔的时间单位，单位有以下几种：S 秒、M 分、H 小时、D 天、W 每星期（interval==0时代表星期一）midnight 每天凌晨
        handler.setFormatter(format_str)  # 设置文件里写入的格式

        self.logger.addHandler(console)  # 把对象加到logger里
        self.logger.addHandler(handler)

    def debug(self, msg):
        """
        定义输出的颜色debug--white，info--green，warning/error/critical--red
        :param msg: 输出的log文字
        :return:
        """
        self.logger.debug(Fore.WHITE + str(msg) + Style.RESET_ALL)

    def info(self, msg):
        self.logger.info(Fore.GREEN + str(msg) + Style.RESET_ALL)

    def warning(self, msg):
        self.logger.warning(Fore.YELLOW + str(msg) + Style.RESET_ALL)

    def error(self, msg):
        self.logger.error(Fore.RED + str(msg) + Style.RESET_ALL)

    def critical(self, msg):
        self.logger.critical(Fore.RESET + str(msg) + Style.RESET_ALL)


if __name__ == '__main__':
    formatter = '%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s: %(message)s'
    log = Logger('all', fmt=formatter)

    log.debug("Debug Information")
    log.info("Info Information")
    log.error("Error Information")
    log.warning("Warning Information")
    log.critical("Critical Information")
