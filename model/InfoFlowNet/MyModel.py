import copy
import math

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict

torch.set_printoptions(precision=2)


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print(f'attention size: {scores.size()}')
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # print(f'x: {x.size()}')
        # print(f'attention score: {self.attn.size()}')
        # print(f'attention score: {self.attn}')

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), self.attn


class MyModel(nn.Module):
    def __init__(self, in_channel=1, time_len=1, kernel_size=1, head=1, d_model=1):
        super(MyModel, self).__init__()
        self.head = head
        self.a = None

        self.Conv1d01 = nn.Conv1d(in_channel, out_channels=in_channel, kernel_size=kernel_size, padding='same')
        self.BN1d01 = nn.BatchNorm1d(in_channel)
        self.ReLu01 = nn.ReLU(inplace=True)
        self.Conv1d02 = nn.Conv1d(in_channel, out_channels=in_channel, kernel_size=kernel_size, padding='same')
        self.BN1d02 = nn.BatchNorm1d(in_channel)
        self.ReLu02 = nn.ReLU(inplace=True)

        if self.head != 0:
            self.Encode = nn.Linear(time_len, d_model)
            self.Q = nn.Linear(d_model, d_model)
            self.K = nn.Linear(d_model, d_model)
            self.V = nn.Linear(d_model, d_model)
            self.Attentionlayer = MultiHeadedAttention(h=self.head, d_model=d_model)
            self.Decode = nn.Linear(d_model, time_len)

        self.Conv1d03 = nn.Conv1d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, padding="same")

    def forward(self, x):
        x = self.ReLu01(self.BN1d01(self.Conv1d01(x)))
        h = self.ReLu02(self.BN1d02(self.Conv1d02(x)))
        if self.head != 0:
            h = self.Encode(h)
            q, k, v = self.Q(h), self.K(h), self.V(h)
            h, self.a = self.Attentionlayer(q, k, v)
            h = self.Decode(h)
        output = self.Conv1d03(h)
        return output, self.a


if __name__ == '__main__':
    data = pd.read_csv(
        "C:\\Users\\user\\CausalityByAttention\\data\\s13_overlap1\\csvdata\\001_001.csv").to_numpy().transpose()
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).cuda()

    print(f'data size: {data.size()}')

    head = 8
    in_channel = 3
    kernel_size = 15
    time_len = 100
    d_model = 512

    myModel = MyModel(in_channel=in_channel, time_len=time_len, kernel_size=kernel_size, head=head, d_model=d_model).cuda()

    for name, param in myModel.named_parameters():
        print(name, param.size())

    y, a = myModel(data)
    print(f'y size: {y.size()}, a size: {a.size()}')
    print(f'loss: {nn.functional.mse_loss(data, y)}')
