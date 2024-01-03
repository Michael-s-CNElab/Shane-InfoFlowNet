import csv
import os
import time

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import savemat
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw
from astropy.stats import circcorrcoef
from torch.utils.data import DataLoader

from plotfigure import plot_shuffle_signal

from MyDataset import myinferenceDataset

import torch
import tqdm

from MyModel import MyModel

# from Old_Model import MyModel

"""inference 實驗用的 hyper parameter"""
ch_name = ['sine', 'sawtooth', 'random']
# ch_name = ['Fz', 'Cz', 'Pz', 'Oz']
# ch_name = ['Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'Oz']
head = 2
in_channel = len(ch_name)
data_name = 'Multitasking'
proj_path = "C:\\Users\\user\\CausalityByAttention\\result\\s13_head2_3ch_WOI_20230919101828\\"
label = f"G:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\s13\\s13_overlap1_label.csv"
csvfilepath = f"G:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\s13\\csvdata\\"
WOI = True
windows = [1, 101, 201, 301, 401]
windows_size = len(windows)
print(windows_size)
shuffle = 30
"""inference 實驗用的 hyper parameter"""

torch.set_printoptions(precision=5)

inference_start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
inference_data_name = f"{data_name}_overlap10"

total_epoch = 100
time_len = 100
d_model = 512
kernel_size = 15
lr = 0.001

use_cuda = 1
device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")

save_result_path = f'{proj_path}\\inference_{inference_start_time}\\'
if not os.path.exists(save_result_path):
    os.mkdir(save_result_path)

save_record_path = f'{save_result_path}\\record\\'
if not os.path.exists(save_record_path):
    os.mkdir(save_record_path)

import matplotlib

matplotlib.use('Agg')

MyInfDataset = myinferenceDataset(label=label, csvfilepath=csvfilepath, windows=windows)
MyInfDataloader = DataLoader(dataset=MyInfDataset, batch_size=1)
model = MyModel(in_channel=in_channel, kernel_size=kernel_size,
                head=head, d_model=d_model, time_len=time_len).to(device)

model.load_state_dict(torch.load(f'{proj_path}model_state_dict_best.pt'))

model.eval()

# err1 = np.zeros([in_channel, windows_size, trial, subject])
# err2 = np.zeros([in_channel, in_channel, windows_size, trial, subject, shuffle])
# mse1 = np.zeros([in_channel, windows_size, trial, subject])
# mse2 = np.zeros([in_channel, in_channel, windows_size, trial, subject, shuffle])
# corr1 = np.zeros([in_channel, windows_size, trial, subject])
# corr2 = np.zeros([in_channel, in_channel, windows_size, trial, subject, shuffle])
# xcorr1 = np.zeros([in_channel, windows_size, trial, subject])
# xcorr2 = np.zeros([in_channel, in_channel, windows_size, trial, subject, shuffle])
# xcorr1_lag = np.zeros([in_channel, windows_size, trial, subject])
# xcorr2_lag = np.zeros([in_channel, in_channel, windows_size, trial, subject, shuffle])
# cos1 = np.zeros([in_channel, windows_size, trial, subject])
# cos2 = np.zeros([in_channel, in_channel, windows_size, trial, subject, shuffle])
# DTW1 = np.zeros([in_channel, windows_size, trial, subject])
# DTW2 = np.zeros([in_channel, in_channel, windows_size, trial, subject, shuffle])
#
# attention_weight = np.zeros([head, in_channel, in_channel, windows_size, trial, subject])

Inf_Record = pd.DataFrame(columns={'subject', 'trial', 'window', 'mse time', 'corr time', 'xcorr time', 'cos time'})
record_filename = f"{save_result_path}Record.csv"
Inf_Record.to_csv(record_filename)

with torch.no_grad():
    for idx, (x, name, sub, tr, wsp) in enumerate(MyInfDataloader):

        err1 = np.zeros([in_channel])
        err2 = np.zeros([in_channel, in_channel, shuffle])
        mse1 = np.zeros([in_channel])
        mse2 = np.zeros([in_channel, in_channel, shuffle])
        corr1 = np.zeros([in_channel])
        corr2 = np.zeros([in_channel, in_channel, shuffle])
        xcorr1 = np.zeros([in_channel])
        xcorr2 = np.zeros([in_channel, in_channel, shuffle])
        xcorr1_lag = np.zeros([in_channel])
        xcorr2_lag = np.zeros([in_channel, in_channel, shuffle])
        cos1 = np.zeros([in_channel])
        cos2 = np.zeros([in_channel, in_channel, shuffle])

        mse_time = []
        corr_time = []
        xcorr_time = []
        cosine_time = []
        # DTW1 = np.zeros([in_channel, windows_size])
        # DTW2 = np.zeros([in_channel, in_channel, windows_size, shuffle])

        attention_weight = np.zeros([head, in_channel, in_channel])
        x = x.to(device)
        x_origin = x.clone()
        x_shuffle = x.clone()
        y, a = model(x)

        x = x.detach().cpu().numpy().squeeze()
        y = y.detach().cpu().numpy().squeeze()

        if head != 0:
            # attention_weight[:, :, :, windows.index(wsp) - 1, tr - 1, sub - 1] = a.detach().cpu().numpy().squeeze()
            attention_weight[:, :, :] = a.detach().cpu().numpy().squeeze()

        sub = sub.item()
        tr = tr.item()
        wsp = wsp.item()

        for s in range(shuffle):

            for ch_shuffle in range(in_channel):
                shuffle_idx = torch.randperm(x_shuffle.size(2))
                x_shuffle = x_origin.clone()
                x_shuffle[:, ch_shuffle, :] = x_shuffle[:, ch_shuffle, shuffle_idx]
                y_shuffle, a_shuffle = model(x_shuffle)

                x_shuffle_np = x_shuffle.detach().cpu().numpy().squeeze()
                y_shuffle_np = y_shuffle.detach().cpu().numpy().squeeze()

                for ch in range(in_channel):
                    w = windows.index(wsp)

                    print(f'origin -- subject: {sub} -- trial: {tr}'
                          f' -- window: {wsp} --- {mean_squared_error(x[ch], y[ch])}')

                    print(f'shuffle {s} ch: {ch_shuffle} -- subject: {sub} -- trial: {tr}'
                          f' -- window: {wsp} --- {mean_squared_error(x[ch], y_shuffle_np[ch])}')

                    mse_start_time = time.time()
                    # mse1[ch, w - 1, tr - 1, sub - 1] = mean_squared_error(x[ch], y[ch])
                    # mse2[ch, ch_shuffle, w - 1, tr - 1, sub - 1, s] = mean_squared_error(x[ch],
                    #                                                                      y_shuffle_np[ch])

                    mse1[ch] = mean_squared_error(x[ch], y[ch])
                    mse2[ch, ch_shuffle, s] = mean_squared_error(x[ch], y_shuffle_np[ch])
                    mse_end_time = time.time()
                    mse_time.append(mse_end_time - mse_start_time)

                    corr_start_time = time.time()

                    # corr1[ch, w - 1, tr - 1, sub - 1] = circcorrcoef(x[ch], y[ch])
                    # corr2[ch, ch_shuffle, w - 1, tr - 1, sub - 1, s] = circcorrcoef(x[ch], y_shuffle_np[ch])

                    corr1[ch] = circcorrcoef(x[ch], y[ch])
                    corr2[ch, ch_shuffle, s] = circcorrcoef(x[ch], y_shuffle_np[ch])

                    corr_end_time = time.time()
                    corr_time.append(corr_end_time - corr_start_time)

                    xcorr_start_time = time.time()
                    xc1 = signal.correlate(x[ch], y[ch], mode='full', method='fft') / time_len
                    lags1 = signal.correlation_lags(x[ch].size, y[ch].size, mode='full')

                    # xcorr1_lag[ch, w - 1, tr - 1, sub - 1] = lags1[np.argmax(xc1)]
                    # xcorr1[ch, w - 1, tr - 1, sub - 1] = np.max(xc1)

                    xcorr1_lag[ch] = lags1[np.argmax(xc1)]
                    xcorr1[ch] = np.max(xc1)

                    xc2 = signal.correlate(x[ch], y_shuffle_np[ch], mode='full', method='fft') / time_len
                    lags2 = signal.correlation_lags(x[ch].size, y_shuffle_np[ch].size, mode='full')

                    # xcorr2_lag[ch, ch_shuffle, w - 1, tr - 1, sub - 1, s] = lags2[np.argmax(xc2)]
                    # xcorr2[ch, ch_shuffle, w - 1, tr - 1, sub - 1, s] = np.max(xc2)

                    xcorr2_lag[ch, ch_shuffle, s] = lags2[np.argmax(xc2)]
                    xcorr2[ch, ch_shuffle, s] = np.max(xc2)
                    xcorr_end_time = time.time()
                    xcorr_time.append(xcorr_end_time - xcorr_start_time)

                    cos_start_time = time.time()

                    # cos1[ch, w - 1, tr - 1, sub - 1] = cosine_similarity(x[ch].reshape(1, -1), y[ch].reshape(1, -1))
                    # cos2[ch, ch_shuffle, w - 1, tr - 1, sub - 1, s] = cosine_similarity(x[ch].reshape(1, -1),
                    #                                                                     y_shuffle_np[ch].reshape(1, -1))

                    cos1[ch] = cosine_similarity(x[ch].reshape(1, -1), y[ch].reshape(1, -1))
                    cos2[ch, ch_shuffle, s] = cosine_similarity(x[ch].reshape(1, -1), y_shuffle_np[ch].reshape(1, -1))
                    cos_end_time = time.time()
                    cosine_time.append(cos_end_time - cos_start_time)

                    # DTW_start_time = time.time()
                    # DTW1[ch, w - 1, tr - 1, sub - 1] = fastdtw(x[ch], y[ch], dist=distance.euclidean)[0]
                    # DTW2[ch, ch_shuffle, w - 1, tr - 1, sub - 1, s] = \
                    #     fastdtw(x[ch], y_shuffle_np[ch], dist=distance.euclidean)[0]
                    # DTW_end_time = time.time()
                    # DTW_time.append(DTW_end_time - DTW_start_time)

                    # if s < 3 and sub % 10 == 1 and tr % 10 == 1:
                    #     plot_shuffle_signal(x_shuffle=x_shuffle_np, y_shuffle=y_shuffle_np, gt=x, shuffle_time=s,
                    #                         c_shuffle=ch_shuffle, figure_path=save_result_path, ch_name=ch_name,
                    #                         name=f"{name[0][:-4]}_{str(wsp).zfill(4)}")

        savemat(f'{save_record_path}{str(sub).zfill(2)}_{str(tr).zfill(3)}_{str(wsp).zfill(4)}.mat',
                mdict={'mse1': mse1, 'mse2': mse2,
                       'corr1': corr1, 'corr2': corr2,
                       'xcorr1': xcorr1, 'xcorr2': xcorr2,
                       'xcorr1_lag': xcorr1_lag, 'xcorr2_lag': xcorr2_lag,
                       'cos1': cos1, 'cos2': cos2,
                       # 'DTW1': DTW1, 'DTW2': DTW2,
                       'attention_weight': attention_weight
                       })

        new_data = [sub, tr, wsp, sum(mse_time), sum(corr_time), sum(xcorr_time), sum(cosine_time)]

        with open(record_filename, "a+", newline='', encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(new_data)

        f.close()
