#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: d1_bi_lstm.py
@time: 2019/11/25 8:27 下午
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiLSTM(nn.Module):
    def __int__(self, vocab_size, hidden_dim, emb_dim, out_dim, pretrained_vec):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(pretrained_vec)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=2, dropout=0.1, bidirectional=True)
        self.linear = nn.Linear(hidden_dim*2, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, out_dim)

    def forward(self, seq):
        emb = self.embedding(seq)
        hdn, _ = self.encoder(emb)
        feature = hdn[-1, :, :]
        feature = self.linear(feature)
        preds = self.predictor(feature)
        return preds
