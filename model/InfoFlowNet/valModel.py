import os
import time

from MyDataset import myDataset

import torch
from torch import nn, optim
import torch.nn.functional as F
import seaborn as sns

import plotfigure

from MyModel import MyModel

# from Old_Model import MyModel

torch.set_printoptions(precision=5)

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

label = "C:\\Users\\user\\CausalityByAttention\\data\\s13_Inde\\s13_label.csv"
csvfilepath = "C:\\Users\\user\\CausalityByAttention\\data\\s13_Inde\\csvdata\\"

proj_path = "C:\\Users\\user\\CausalityByAttention\\result\\s13_1_8_20230706221144\\"

import matplotlib

matplotlib.use('Agg')


def valModel():
    MyValDataset = myDataset(label=label, csvfilepath=csvfilepath)
    model = MyModel(in_channel=in_channel, kernel_size=kernel_size,
                    head=head, d_model=d_model, time_len=time_len).to(device)

    model.load_state_dict(torch.load(f'{proj_path}model_state_dict_best.pt'))

    model.eval()

    for idx, (x, name) in enumerate(MyValDataset):
        x = x.unsqueeze(0).to(device)
        y = model(x)

        plotfigure.plot_signal(input=x, predict=y, name=f'val_{name[:-4]}', figure_path=proj_path,
                               time_len=time_len, ch_name=ch_name, in_channel=in_channel)

        print(f'plot {name[:-4]} is done.')

        if idx == 400:
            break


if __name__ == '__main__':
    valModel()