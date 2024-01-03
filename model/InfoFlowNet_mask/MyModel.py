import copy
import math

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict

from torch.utils.data import DataLoader

from MyDataset import myDataset

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
        scores = scores.masked_fill(mask == 1, -1e9)
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
            mask = torch.eye(x.size(1)).unsqueeze(0).unsqueeze(0).cuda()
            mask = mask.expand(x.size(0), self.head, x.size(1), x.size(1))
            h = self.Encode(h)
            q, k, v = self.Q(h), self.K(h), self.V(h)
            h, self.a = self.Attentionlayer(q, k, v, mask)
            h = self.Decode(h)
        output = self.Conv1d03(h)
        return output, self.a


if __name__ == '__main__':
    # ch_name = ['sine', 'sawtooth', 'random']
    ch_name = ['Fz', 'Cz', 'Pz', 'Oz']
    # ch_name = ['Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'Oz']
    head = 1
    in_channel = len(ch_name)
    data_name = 'Lanekeeping'
    label = f"G:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\{data_name}\\{data_name}_baseline_overlap10_label.csv"
    csvfilepath = f"G:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\{data_name}\\csvdata\\"
    WOI = True
    batch_size = 16
    MyTrainDataset = myDataset(label=label, csvfilepath=csvfilepath, WOI=WOI)
    Train_Dataloader = DataLoader(MyTrainDataset, shuffle=True, batch_size=batch_size)
    use_cuda = 1
    device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
    total_epoch = 100
    batch_size = 128
    time_len = 100
    d_model = 512
    kernel_size = 15
    lr = 0.001

    model = MyModel(in_channel=in_channel, kernel_size=kernel_size,
                    head=head, d_model=d_model, time_len=time_len).to(device)

    for idx, (x, name) in enumerate(Train_Dataloader):
        x = x.to(device)
        y, a = model(x)
