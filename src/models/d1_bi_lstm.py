#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: d1_bi_lstm.py
@time: 2019/11/25 8:27 下午
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
embedding = True


class BiLSTM(nn.Module):
    def __int__(self, vocab_size, hidden_dim, emb_dim, out_dim, pretrained_vec):
        super(BiLSTM, self).__init__()
        if embedding:
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


class Bilstm(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(Bilstm, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.LSTM(self.input_dim, hidden, bidirectional=True, batch_first=True)
        self.linner1 = nn.Linear(hidden*2, int(hidden/2))

        self.linner2 = nn.Linear(int(hidden/2), int(hidden/2))
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(int(hidden/4), output_dim)

    def forward(self, x):
        out, _ = self.encoder(x)
        out = torch.mean(out, 1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out = self.relu(out)
        h_conc_linear = self.output(out)
        return h_conc_linear


model_stock = Bilstm(4642, 1024, 3)
optimizer = torch.optim.Adam(model_stock.parameters(), lr=0.001)
loss_F = torch.nn.MSELoss()
print(model_stock)
