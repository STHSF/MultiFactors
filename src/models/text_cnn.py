#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: text_cnn.py
@time: 2019/12/5 2:27 下午
"""


import torch
import torch.nn as nn
import numpy as np


class BasicModule(nn.Module):
    # 封装了nn.Module， 主要是提供了save和load两个方法
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 模型的默认名字

    def get_optimizer(self, lr1, lr2=0, weight_decay=0):
        embed_params = list(map(id, self.embedding.parameters()))
        base_params = filter(lambda p: id(p) not in embed_params, self.parameters())
        optimizer = torch.optim.Adam([
            {'params': self.embedding.parameters(), 'lr': lr2},
            {'params': base_params, 'lr': lr1, 'weight_decay': weight_decay}
        ])
        return optimizer


kernal_sizes = [1, 2, 3, 4, 5]


class TextCNN(BasicModule):
    def __init__(self, vectors=None):
        super(TextCNN, self).__init__()
        embedding_dim = 7,
        kernel_num = 1024,
        kernel_size = 7
        linear_hidden_size = 512
        convs = [
            nn.Sequential(
                nn.Conv1d(in_channels=600, out_channels=1024, kernel_size=7),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=1024,  out_channels=1024, kernel_size=7),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),

                nn.MaxPool1d(kernel_size=(600 - 7*k + 2))
            )
            for k in kernal_sizes  # 池化层划窗
        ]
        self.convs = nn.ModuleList(convs)

        self.fc = nn.Sequential(
            nn.Linear(5120, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 102)
        )

    def forward(self, inputs):
        # embeds = self.embedding(inputs)  # seq * batch * embed
        # 进入卷积层前需要将Tensor第二个维度变成emb_dim，作为卷积的通道数
        # conv_out = [conv(embeds.permute(1, 2, 0)) for conv in self.convs]
        conv_out = [conv(inputs) for conv in self.convs]
        conv_out = torch.cat(conv_out, dim=1)
        flatten = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(flatten)
        logits = torch.sum(logits, 0, True)  # 多个预测值合并
        return logits

sk = TextCNN()
print(sk)
