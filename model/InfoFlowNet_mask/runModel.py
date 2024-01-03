import csv
import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from MyDataset import myDataset

import torch
from torch import nn, optim
import torch.nn.functional as F
import seaborn as sns

import plotfigure

from MyModel import MyModel

# from Old_Model import MyModel

torch.set_printoptions(precision=5)

"""training實驗用的 hyper parameter"""
# ch_name = ['sine', 'sawtooth', 'random']
ch_name = ['Fz', 'Cz', 'Pz', 'Oz']
# ch_name = ['Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'Oz']
head = 8
in_channel = len(ch_name)
data_name = 'Multitasking'
label = f"G:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\{data_name}\\{data_name}_overlap10_label.csv"
csvfilepath = f"G:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\{data_name}\\csvdata\\"
WOI = True
"""training實驗用的 hyper parameter"""

proj_start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())


use_cuda = 1
device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")

total_epoch = 100
batch_size = 128
time_len = 100
d_model = 512
kernel_size = 15
lr = 0.001

proj_name = f"{data_name}_{in_channel}ch_head{head}_eye"
if WOI:
    proj_name = proj_name + "_WOI"

proj_dir = "C:\\Users\\user\\CausalityByAttention\\"

save_result_path = f'{proj_dir}result\\{proj_name}_{proj_start_time}\\'
if not os.path.exists(save_result_path):
    os.mkdir(save_result_path)

import matplotlib

matplotlib.use('Agg')

train_dict = {
    'total epoch': total_epoch,
    'batch size': batch_size,
    'head': head,
    'channel size': in_channel,
    'time len': time_len,
    'd model': d_model,
    'kernel size': kernel_size,
    'learning rate': lr,
    'WOI': WOI,
    'label path': label,
    'csvfile path': csvfilepath
}

with open(f'{save_result_path}train_dict.csv', 'w', newline="", encoding='utf-8') as f:
    writer = csv.writer(f)
    for k, v in train_dict.items():
       writer.writerow([k, v])


def trainModel():
    train_info = {}
    plt_loss_train, train_cost_time, val_cost_time = [], [], []
    MyTrainDataset = myDataset(label=label, csvfilepath=csvfilepath, WOI=WOI)
    Train_Dataloader = DataLoader(MyTrainDataset, shuffle=True, batch_size=batch_size)

    print(f'train dataset: {len(MyTrainDataset)}')

    model = MyModel(in_channel=in_channel, kernel_size=kernel_size,
                    head=head, d_model=d_model, time_len=time_len).to(device)
    MymseLoss = nn.MSELoss()
    MycosLoss = nn.CosineSimilarity(dim=2, eps=1e-6)
    MyOptim = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(MyOptim, step_size=20, gamma=0.5)
    best_loss = 100000

    for epoch in range(total_epoch):
        train_loss = 0

        train_start_time = time.time()
        model.train()

        print('best loss: {:.5f} --'.format(best_loss), end=' ')

        for idx, (x, name) in enumerate(Train_Dataloader):
            x = x.to(device)
            MyOptim.zero_grad()
            y, a = model(x)
            mseloss = MymseLoss(y, x)
            cosloss = 1-MycosLoss(y, x).mean()
            loss = (mseloss + cosloss) / 2
            loss.backward()
            MyOptim.step()
            train_loss += loss

        scheduler.step()
        train_epoch_time = (time.time()) - train_start_time
        train_cost_time.append(train_epoch_time)

        train_loss /= len(Train_Dataloader.dataset)
        plt_loss_train.append(train_loss.cpu().detach().item())

        model.eval()
        x, name = MyTrainDataset[0]
        x = x.to(device)
        y, a = model(x.unsqueeze(0))
        val_mse_loss = MymseLoss(y, x.unsqueeze(0))
        val_cos_loss = 1-MycosLoss(y, x.unsqueeze(0)).mean()

        print('epoch: {:>3d}/{} -- '
              'Average loss (Train): {:.5f} -- '
              'train cost time: {:.2f} -- val mse loss: {:.5f} -- val cos loss: {:.5f}'.format(
            epoch + 1,
            total_epoch,
            train_loss,
            train_epoch_time,
            val_mse_loss.cpu().detach().item(),
            val_cos_loss.cpu().detach().item()))

        plotfigure.plot_signal(input=x, predict=y, name=f'epoch_{str(epoch + 1).zfill(3)}_{name[:-4]}',
                               figure_path=save_result_path, time_len=time_len, ch_name=ch_name, in_channel=in_channel)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model, f'{save_result_path}model.pt')
            torch.save(model.state_dict(), f'{save_result_path}model_state_dict_best.pt')
            x, name = MyTrainDataset[0]
            x = x.to(device)
            y, a = model(x.unsqueeze(0))
            plotfigure.plot_signal(input=x, predict=y, name=f'best_{name[:-4]}', figure_path=save_result_path,
                                   time_len=time_len, ch_name=ch_name, in_channel=in_channel)

        train_info['train loss'] = plt_loss_train
        train_info['train cost time'] = train_cost_time

    info = pd.DataFrame.from_dict(train_info)
    info.to_csv(f'{save_result_path}train_info.csv')


def valModel():
    MyValDataset = myDataset(label=label, csvfilepath=csvfilepath)
    model = MyModel(in_channel=in_channel, kernel_size=kernel_size,
                    head=head, d_model=d_model, time_len=time_len).to(device)
    model.load_state_dict(torch.load(f'{save_result_path}model_state_dict_best.pt'))

    model.eval()

    for idx, (x, name) in enumerate(MyValDataset):
        x = x.unsqueeze(0).to(device)
        y, a = model(x)

        plotfigure.plot_signal(input=x, predict=y, name=f'val_{name[:-4]}_{str(idx).zfill(3)}',
                               figure_path=save_result_path, time_len=time_len, ch_name=ch_name, in_channel=in_channel)

        print(f'plot {name[:-4]} is done.')

        if idx == 400:
            break


if __name__ == '__main__':
    trainModel()
    valModel()
