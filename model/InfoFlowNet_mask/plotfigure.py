import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_signal(input: torch.Tensor, predict: torch.Tensor, name, figure_path, time_len, ch_name, in_channel):
    input = input.squeeze(0).detach().cpu().numpy()
    predict = predict.squeeze(0).detach().cpu().numpy()
    x = np.linspace(0, 20, time_len)

    fig, ax = plt.subplots(in_channel, 1, figsize=[16, in_channel * 3])

    for i in range(in_channel):
        ax[i].plot(x, input[i], label='ground truth')
        ax[i].plot(x, predict[i], label='predict')
        ax[i].set_xticks([])
        ax[i].set_yticks([-4, 4])
        ax[i].set_xlim(0, 20)
        ax[i].set_title(ch_name[i], fontsize=16)
        ax[i].legend()

    if not os.path.exists(f"{figure_path}figure_signal"):
        os.mkdir(f"{figure_path}figure_signal")
    plt.savefig(f"{figure_path}figure_signal\\{name}.png")
    plt.clf()
    plt.cla()
    plt.close('all')


def plot_shuffle_signal(x_shuffle, y_shuffle, gt, shuffle_time, c_shuffle, figure_path, ch_name, name):
    x = np.linspace(0, 20, 100)

    fig, ax = plt.subplots(len(ch_name), 1, figsize=[16, 9])

    for i in range(len(ch_name)):
        ax[i].plot(x, x_shuffle[i], label='input')
        ax[i].plot(x, y_shuffle[i], label='predict')
        ax[i].plot(x, gt[i], label='ground truth')
        ax[i].set_xticks([])
        ax[i].set_yticks([-4, 4])
        ax[i].set_xlim(0, 20)
        ax[i].set_title(ch_name[i], fontsize=16)
        ax[i].legend()

    if not os.path.exists(f"{figure_path}figure_shuffle_signal"):
        os.mkdir(f"{figure_path}figure_shuffle_signal")

    figure_name = f"shuffle{str(shuffle_time + 1).zfill(3)}_{name}_{ch_name[c_shuffle]}.png"

    plt.savefig(f"{figure_path}figure_shuffle_signal\\{figure_name}")
    plt.clf()
    plt.cla()
    plt.close('all')
