#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: d1_bi_lstm.py
@time: 2019/11/25 8:27 下午
"""
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
embedding = True


class BiRNN(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(BiRNN, self).__init__()
        self.rnn = 'gru'
        self.input_dim = input_dim
        self.output_dim = output_dim
        if self.rnn == 'lstm':
            self.encoder = nn.LSTM(self.input_dim, hidden, bidirectional=True, batch_first=True)
        else:
            self.encoder = nn.GRU(self.input_dim, hidden, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(hidden*2, int(hidden/2))
        self.linear2 = nn.Linear(int(hidden/2), int(hidden/2))
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(int(hidden/4), output_dim)

    def forward(self, x):
        # out: [batch, seq_len, hidden_size * num_directions]
        out, _ = self.encoder(x)
        # out: [batch, hidden_size * num_directions]
        out = torch.mean(out, 1)
        # out: [batch, hidden_size/2]
        out = self.linear1(out)
        # out: [batch, hidden_size/4]
        out = self.linear2(out)
        # out: [batch, hidden_size/4]
        out = self.dropout(out)
        # out: [batch, hidden_size/4]
        out = self.relu(out)
        # out: [batch, output_dim]
        h_conc_linear = self.output(out)
        return h_conc_linear


model = BiRNN(4642, 1024, 3)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)


for param in model.parameters():
    print(type(param.data), param.size())
