#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: graph.py
@time: 2019/12/3 5:03 下午
"""

import tensorflow as tf
from src.models.bi_lstm import args


class Graph(object):
    def __init__(self):
        self.p = tf.placeholder(name='p', shape=(None, args.max_len), dtype=tf.int32)
        self.h = tf.placeholder(name='h', shape=(None, args.max_len), dtype=tf.int32)

        self.p_vec = tf.placeholder(name='p_word', shape=(None, args.max_word_len, args.word_embedding_len), dtype=tf.float32)
        self.h_vec = tf.placeholder(name='h_word', shape=(None, args.max_word_len, args.word_embedding_len), dtype=tf.float32)

        self.y = tf.placeholder(name='y', shape=(None, ), dtype=tf.float64)

        self.keep_prob = tf.placeholder(name='keep_prob', dtype=tf.float32)

        self.embed = tf.get_variable(name='embed', shape=(args.char_vocab_len, args.char_hidden_size), dtype=tf.float32)

        for i in range(1, 9):
            setattr(self, f'w{i}', tf.get_variable(name=f'w{i}', shape=(args.num_perspective, args.char_hidden_size),
                                                   dtype=tf.float32))

        self.forward()
        self.train()

    @staticmethod
    def dropout(x, keep_prob):
        return tf.nn.dropout(x, keep_prob)

    @staticmethod
    def BiLSTM(x):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(args.lstm_hidden)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(args.lstm_hidden)
        outputs, outputs_state = tf.contrib.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
        # outputs, outputs_state = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
        return outputs

    def forward(self):

        self.BiLSTM(self.p_vec)
        pass

    def train(self):
        pass

