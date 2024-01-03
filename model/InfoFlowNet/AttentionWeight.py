import copy
import math
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from MyDataset import myDataset, myinferenceDataset

import torch
from torch import nn, optim
import torch.nn.functional as F
import seaborn as sns

import plotfigure

from MyModel import MyModel, DecoderLayer

# from Old_Model import MyModel

torch.set_printoptions(precision=5)

proj_start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
ch_name = ['sine', 'sawtooth', 'random']

use_cuda = 1
device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")

total_epoch = 100
batch_size = 128
layer = 1
head = 8
in_channel = 3
time_len = 100
d_model = 512
kernel_size = 15
lr = 0.001
WOI = False

label = "C:\\Users\\user\\CausalityByAttention\\data\\s13_overlap1\\s13_overlap1_label.csv"
csvfilepath = "C:\\Users\\user\\CausalityByAttention\\data\\s13_overlap1\\csvdata\\"

proj_path = "C:\\Users\\user\\CausalityByAttention\\result\\s13_head8_WOI_20230710155640_done\\"

import matplotlib

matplotlib.use('Agg')

"""simulate dataset"""
subject = 1
trial = 100
windows = [1, 101, 201, 301, 401]
windows_size = len(windows)

MyInfDataset = myinferenceDataset(label=label, csvfilepath=csvfilepath, windows=windows)
MyInfDataloader = DataLoader(dataset=MyInfDataset, batch_size=1)

attn_weight = np.zeros([in_channel, in_channel, head, windows_size, trial, subject])

torch.set_printoptions(precision=2)

model = MyModel(in_channel=in_channel, kernel_size=kernel_size,
                head=head, d_model=d_model, time_len=time_len).to(device)

model.load_state_dict(torch.load(f'{proj_path}model_state_dict_best.pt'))

model.eval()


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

class AttentionLayer(nn.Module):
    def __init__(self, head=1, d_model=1):
        super(AttentionLayer, self).__init__()
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)

        self.MAlayer = MultiHeadedAttention(h=head, d_model=d_model)

    def forward(self, x):
        query = self.Q(x)
        key = self.K(x)
        value = self.V(x)
        # print(f'query size: {query.size()}')
        output, attn = self.MAlayer(query, key, value)
        return output, attn


class MytempModel(nn.Module):
    def __init__(self, in_channel=1, time_len=1, kernel_size=1, head=1, d_model=1):
        super(MytempModel, self).__init__()

        self.Conv1d01 = nn.Conv1d(in_channel, out_channels=in_channel, kernel_size=kernel_size, padding='same')
        self.BN1d01 = nn.BatchNorm1d(in_channel)
        self.ReLu01 = nn.ReLU(inplace=True)
        self.Conv1d02 = nn.Conv1d(in_channel, out_channels=in_channel, kernel_size=kernel_size, padding='same')
        self.BN1d02 = nn.BatchNorm1d(in_channel)
        self.ReLu02 = nn.ReLU(inplace=True)
        self.encode = DecoderLayer(time_len=time_len, d_model=d_model)
        self.MAlayer = AttentionLayer(head=head, d_model=d_model)

    def forward(self, x):
        x = self.ReLu01(self.BN1d01(self.Conv1d01(x)))
        x = self.ReLu02(self.BN1d02(self.Conv1d02(x)))
        x = self.encode(x)
        x = self.MAlayer(x)
        return x


tempmodel = MytempModel(in_channel=in_channel, kernel_size=kernel_size,
                        head=head, d_model=d_model, time_len=time_len).to(device)

print("tempmodel before weights:", tempmodel.MAlayer.K.weight.data)
tempmodel.Conv1d01.weight.data = model.conv1.Conv1d01.weight.data.clone()
tempmodel.Conv1d01.bias.data = model.conv1.Conv1d01.bias.data.clone()
tempmodel.BN1d01.weight.data = model.conv1.BN1d01.weight.data.clone()
tempmodel.BN1d01.bias.data = model.conv1.BN1d01.bias.data.clone()
tempmodel.Conv1d02.weight.data = model.conv1.Conv1d02.weight.data.clone()
tempmodel.Conv1d02.bias.data = model.conv1.Conv1d02.bias.data.clone()
tempmodel.BN1d02.weight.data = model.conv1.BN1d02.weight.data.clone()
tempmodel.BN1d02.bias.data = model.conv1.BN1d02.bias.data.clone()
tempmodel.encode.fc.weight.data = model.Block1.Encoder01.fc.weight.data.clone()
tempmodel.encode.fc.bias.data = model.Block1.Encoder01.fc.bias.data.clone()
tempmodel.MAlayer.Q.weight.data = model.Block1.Attention01.Q.weight.data.clone()
tempmodel.MAlayer.Q.bias.data = model.Block1.Attention01.Q.bias.data.clone()
tempmodel.MAlayer.K.weight.data = model.Block1.Attention01.K.weight.data.clone()
tempmodel.MAlayer.K.bias.data = model.Block1.Attention01.K.bias.data.clone()
tempmodel.MAlayer.V.weight.data = model.Block1.Attention01.V.weight.data.clone()
tempmodel.MAlayer.V.bias.data = model.Block1.Attention01.V.bias.data.clone()
print("tempmodel after weights:", tempmodel.MAlayer.K.weight.data)

tempmodel.eval()

with torch.no_grad():
    for idx, (x, name, sub, tr, wsp) in enumerate(MyInfDataloader):
        x = x.to(device)
        x, attn = tempmodel(x)
        print(attn.size())
#
# data = pd.read_csv(
#         "C:\\Users\\user\\CausalityByAttention\\data\\s13_overlap1\\csvdata\\001_201.csv").to_numpy().transpose()
# data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).cuda()
#
# y, attn = tempmodel(data)
# print(f'y size: {attn.size()}')
