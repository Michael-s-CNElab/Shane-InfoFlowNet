import csv
import sys
import time

import numpy as np
import pandas as pd
from os import walk

import os
import glob
import shutil
import torch
import shutil

from matplotlib import pyplot as plt
from torch.autograd import Variable

import TCDF
from model import ADDSTCN
import logging

result_save_path = "D:\\TCDF_result\\"
TCDF_result_201300 = []

channel_location = np.array(["FP1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9", "CP5", "CP1", "PZ",
                             "P3", "P7", "O1", "OZ", "O2", "P4", "P8", "TP10", "CP6", "CP2", "CZ", "C4", "T8",
                             "FT10", "FC6", "FC2", "F4", "F8", "FP2"])

ch_name = ['Fz', 'Cz', 'Pz', 'Oz']


# ch_name = ['Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'Oz']
# ch_name = ['sine', 'sawtooth', 'random']


def transpose_addchannel(dir: str, csvfilename: str):
    file = dir + "\\" + csvfilename
    eegdata = np.genfromtxt(file, delimiter=",").T
    df = pd.DataFrame(eegdata, columns=channel_location)
    save_dir = dir[:-10] + "\\transpose\\" + dir[43:]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + "\\" + csvfilename
    print(save_dir)
    df.to_csv(save_dir, index=False)


def save_result(alldelays, columns, datafile):
    path, savefilename = os.path.split(datafile)
    path = result_save_path
    if not os.path.exists(path):
        os.makedirs(path)
    savefilename = savefilename[:-4] + "_TCDFresult.csv"
    savefilename = path + savefilename
    data = []
    for pair in alldelays:
        data.append([columns[pair[1]], columns[pair[0]], alldelays[pair]])
    pd_data = pd.DataFrame(data)
    pd_data.to_csv(savefilename, index=False, header=False)
    print("result : ", savefilename)


def combine_onetrial2onepeople():
    dir = 'F:\\TCDF\\data\\Multitasking\\rt_trial\\slow_v1\\transpose\\1001_1500\\'
    csvfilelist = os.listdir(dir)
    people = []
    session = []
    for i in range(len(csvfilelist)):
        if '.csv' in csvfilelist[i]:
            if i == 0:
                session.append(csvfilelist[i])
                continue

            if i == len(csvfilelist) - 3:
                people.append(session)
                session = []

            if csvfilelist[i][0:4] == csvfilelist[i - 1][0:4]:
                session.append(csvfilelist[i])
            else:
                people.append(session)
                session = []
                session.append(csvfilelist[i])

    savedir = "F:\\TCDF\\data\\Multitasking\\rt_trial\\slow_v1\\transpose\\1001_1500_onepeople_onetrial\\"

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for s in range(len(people)):
        df = pd.DataFrame()
        for i in range(len(people[s])):
            file = dir + people[s][i]
            data = pd.read_csv(file)
            df = df.append(data)
        filename = savedir + people[s][0][:4] + ".csv"
        df.to_csv(filename, index=False)
        print(f"%s is done" % filename)


def combine_onetrial2allpeople():
    dir = 'F:\\TCDF\\data\\Multitasking\\rt_trial_downsample250\\fast_v1\\transpose\\1001_1500\\'
    csvfilelist = os.listdir(dir)
    people = []
    session1 = []
    session2 = []
    session3 = []
    session4 = []
    for csvfile in csvfilelist:
        if '.csv' in csvfile:
            splitcsvfile = csvfile.split('_')
            if splitcsvfile[1] == '1':
                session1.append(csvfile)
            if splitcsvfile[1] == '2':
                session2.append(csvfile)
            if splitcsvfile[1] == '3':
                session3.append(csvfile)
            if splitcsvfile[1] == '4':
                session4.append(csvfile)

    df = pd.DataFrame()

    savedir = "F:\\TCDF\\data\\Multitasking\\rt_trial_downsample250\\fast_v1\\transpose\\1001_1500_allpeople_onetrial\\"
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savefilename = ""
    for ss in [session1, session2, session3, session4]:
        for s in ss:
            file = dir + s
            data = pd.read_csv(file)
            df = df.append(data)
            savefilename = s.split("_")[1]
        filename = savedir + savefilename + ".csv"
        df.to_csv(filename, index=False)
        print(f"%s is done" % filename)


def test():
    datafile = "F:\\TCDF\\data\\Multitasking\\rt_trial\\fast_v1\\transpose\\1001_1500_onepeople_onetrial\\01_1.csv"
    df_data = pd.read_csv(datafile)
    row = len(df_data)
    split = int(row / 500)
    columns = list(df_data)
    for c in columns:
        idx = df_data.columns.get_loc(c)
        X_train, Y_train, _ = TCDF.preparedata(datafile, c)

        Y_train = Y_train.reshape(row, 1)
        X_train = np.split(X_train, split, axis=1)
        Y_train = np.split(Y_train, split, axis=0)
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        X_train, Y_train = Variable(torch.from_numpy(X_train)), Variable(torch.from_numpy(Y_train))

        print("X dim2", X_train.shape)
        print("Y dim2", Y_train.shape)

        input_channels = X_train.size()[1]
        targetidx = pd.read_csv(datafile).columns.get_loc(c)
        # print(targetidx)
        x, y = X_train[0:1], Y_train[0:1]
        # print("x ", x.shape)
        # print("y ", y.shape)


def aaa():
    dir = "F:\\TCDF\\data\\Multitasking\\transpose\\TCDFresult"
    csvfilelist = os.listdir(dir)
    for csvfile in csvfilelist:
        if ".csv" in csvfile:
            df = pd.read_csv(dir + "\\" + csvfile, usecols=[2])
            for i in df:
                a = int(i)
                if a > 4:
                    print(csvfile)


def move_data():
    dir = "F:\\TCDF\\TCDF\\data\\Multitasking\\transpose\\"
    csvfilelist = os.listdir(dir)

    for i in range(len(csvfilelist)):
        if '.csv' in csvfilelist[i]:
            folder = dir + csvfilelist[i][:2]
            if not os.path.exists(folder):
                os.mkdir(folder)
            shutil.copy(dir + csvfilelist[i], folder + "\\" + csvfilelist[i])


def Scores_Matrix(s, datafile):
    path, savefilename = os.path.split(datafile)
    path = result_save_path
    if not os.path.exists(path):
        os.makedirs(path)
    savecsvfilename = path + savefilename[:-4] + "_Attention_Scores.csv"
    savefilename = savefilename[:-4] + "_Scores_Matrix.png"
    savefilename = path + savefilename
    fig = plt.figure(figsize=(32, 32))
    ax = fig.gca()
    res = np.array([list(item) for item in s.values()])
    pd_data = pd.DataFrame(res)
    pd_data.to_csv(savecsvfilename, index=False, header=False)
    plt.matshow(res, origin='upper')
    plt.xticks(range(0, res.shape[0]), ch_name)
    plt.yticks(range(0, res.shape[0]), ch_name)
    for i in range(res.shape[0]):
        for j in range(res.shape[0]):
            plt.text(j, i, round(res[i, j], 2),
                     ha="center", va="center", color="w")
    plt.colorbar()
    plt.savefig(savefilename)
    plt.clf()
    # path, savefilename = os.path.split(datafile)
    # path = result_save_path
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # savecsvfilename = path + savefilename[:-4] + "_Attention_Scores.csv"
    # savefilename = savefilename[:-4] + "_Scores_Matrix.png"
    # savefilename = path + savefilename
    # fig = plt.figure(figsize=(64, 64))
    # ax = fig.gca()
    # res = np.array([list(item) for item in s.values()])
    # pd_data = pd.DataFrame(res)
    # pd_data.to_csv(savecsvfilename, index=False, header=False)
    # res = np.flipud(res)
    # plt.matshow(res, origin='lower')
    # plt.xticks(range(0, 32), channel_location, rotation=90, fontsize=8)
    # plt.yticks(range(0, 32), np.flipud(channel_location), fontsize=8)
    # ax.xaxis.set_ticks_position('bottom')
    # plt.colorbar()
    # plt.savefig(savefilename)
    # plt.clf()


def train():
    dir = 'F:\\SMC\\data\\Mutlitasking\\4ch_spilt_csvdata'
    csvfilelist = os.listdir(dir)
    for csvfilename in csvfilelist:
        if ".csv" in csvfilename:
            # print(f".\\runTCDF.py --data %s\\%s --plot --hidden_layer 1" % (dir, csvfilename))
            os.system(
                f".\\runTCDF.py --data %s\\%s --plot --hidden_layer 1 --cuda" % (dir, csvfilename))


def eval():
    datafile = "F:\\TCDF\\data\\Multitasking\\rt_trial\\fast_v1\\transpose\\0001_0500\\01_1_006.csv"
    df_data = pd.read_csv(datafile)
    columns = list(df_data)
    for c in columns:
        X_train, Y_train = TCDF.preparedata(datafile, c)

        X_train = X_train.unsqueeze(0).contiguous()
        Y_train = Y_train.unsqueeze(2).contiguous()

        input_channels = X_train.size()[1]
        targetidx = pd.read_csv(datafile).columns.get_loc(c)
        x, y = X_train[0:1], Y_train[0:1]
        # print("x ", x.shape)
        # print("y ", y.shape)

        model = ADDSTCN(targetidx, input_channels, 0, kernel_size=4, cuda=False, dilation_c=4)

        if os.path.isfile("best-model.pt"):
            model.load_state_dict(torch.load("best-model.pt"), False)
        model.eval()
        with torch.no_grad():
            prediction = model(x)
            scores = model.fs_attention
            s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True)
            indices = np.argsort(-1 * scores.view(-1).cpu().detach().numpy())
            print(s)


def runTCDFmodel(dir, csvfilename):
    os.system(
        f"python .\\runTCDF.py --data %s\\%s --plot --hidden_layer 1 --cuda --epoch 100 --log_interval 50" % (
            dir, csvfilename))


def movefile():
    destination = 'H:\\TCDF\\TCDF_result\\LK_Ch8\\'
    for f in os.listdir(result_save_path):
        if 'Attention_Scores.csv' in f:
            print(f)
            shutil.move(result_save_path + f, destination + f)


def save_file():
    o_dir = f"G:\\我的雲端硬碟\\"
    s_dir = f"D:\\TCDF_result_20230926\\"

    csv1_files = glob.glob(os.path.join(o_dir, '*_TCDFresult.csv'))
    csv2_files = glob.glob(os.path.join(o_dir, '*_Attention_Scores.csv'))
    png1_files = glob.glob(os.path.join(o_dir, '*_TCDFresult.png'))
    png2_files = glob.glob(os.path.join(o_dir, '*_Scores_Matrix.png'))

    for csv_file in png2_files:
        print(csv_file)
        shutil.move(csv_file, os.path.join(s_dir, os.path.basename(csv_file)))


if __name__ == '__main__':
    # save_file()
    start_time = time.time()
    dir = 'C:\\Users\\user\\CausalityByLoss\\MyModel\\data\\Mt_Mid_r_norm\\csvdata\\'
    csvfilelist = os.listdir(dir)
    for csvfilename in csvfilelist:
        runTCDFmodel(dir, csvfilename)
    end_time = time.time()
    print(f"cost time: {end_time - start_time}")
    # movefile()

    # train_10time = []
    # for n in range(10):
    #     start_time = time.time()
    #     dir = 'C:\\Users\\user\\CausalityByLoss\\MyModel\\data\\s13_Inde\\csvdata\\'
    #     csvfilelist = os.listdir(dir)
    #     for csvfilename in csvfilelist:
    #         runTCDFmodel(dir, csvfilename)
    #     end_time = time.time()
    #     train_10time.append((end_time - start_time))
    #
    # print(f"train 10 time: {np.mean(train_10time)}  {np.std(train_10time)}")
    '''
    dir = 'data\\Multitasking\\'
    csvfilelist = os.listdir(dir)e
    print(csvfilelist)
    for csvfilename in csvfilelist:
        transpose_addchannel(dir, csvfilename)
        
    time_window = 100
    for i in range(15):
        start_time = str((time_window * i)+1).zfill(4)
        end_time = str((i+1) * time_window).zfill(4)
        ddd = dir + start_time + "_" + end_time
        os.mkdir(ddd)
        
   
    
    
    a = "F:\\TCDF\\TCDF\\data\\Multitasking\\"
    dir = 'F:\\TCDF\\TCDF\\data\\Multitasking\\rt_trial\\slow\\'
    for root, dirs, files in walk(dir):
        print(root)
        for file in files:
            if '.csv' in file:
                transpose_addchannel(root, file)
    '''
    # train()
    '''
    folder = 'F:\\TCDF\\data\\Simulation\\s50_x1_normpdfg_x2_100trial\\'
    for i in range(100):
        newfolder = folder + f's50_%s' % str(i+1).zfill(3)
        os.mkdir(newfolder)
    '''
    '''
    dir = 'F:\\TCDF\\data\\Simulation\\s50_x1_normpdfg_x2_100trial\\'
    folderlist = os.listdir(dir)
    for folder in folderlist:
        dir_s = dir + str(folder)
        if os.path.isdir(dir_s):
            csvlist = os.listdir(dir_s)
            for csv in csvlist:
                if '.csv' in csv and '401500' in csv and len(csv) <= 19:
                    os.system(
                       f".\\runTCDF.py --data %s\\%s --plot --hidden_layer 1" % (dir_s, csv))
    '''
