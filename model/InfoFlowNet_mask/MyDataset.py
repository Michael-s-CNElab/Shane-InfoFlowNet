import os
from torch.utils.data import DataLoader

import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd
from functools import reduce


def collate_fn(batch):
    print(type(batch[0][0]))
    print(batch[0][0].shape)
    data = list()
    dataname = list()
    sub = list()
    tr = list()
    wsp = list()

    for b in batch:
        data.append(b[0])
        dataname.append(b[1])
        sub.append(b[2])
        tr.append(b[3])
        wsp.append(b[4])

    print(dataname)
    data = torch.stack(data, dim=0)
    dataname = torch.stack(dataname, dim=0)
    sub = torch.stack(sub, dim=0)
    tr = torch.stack(tr, dim=0)
    wsp = torch.stack(wsp, dim=0)
    return data, dataname, sub, tr, wsp


class myDataset(Dataset):
    def __init__(self, label, csvfilepath, WOI=False):
        self.label = pd.read_csv(label)
        self.csvfilepath = csvfilepath

        if WOI:
            self.label = self.label[self.label['WOI'] == 1].reset_index()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        csvfile = self.csvfilepath + self.label['csvname'][idx]
        wsp = self.label['window start pt'][idx]
        wep = self.label['window end pt'][idx]
        data = pd.read_csv(csvfile).to_numpy().transpose()[:, wsp - 1:wep]
        data = torch.tensor(data, dtype=torch.float32)

        return data, self.label['csvname'][idx]


class myinferenceDataset(Dataset):
    def __init__(self, label, csvfilepath, windows=None):
        self.label = pd.read_csv(label)
        self.csvfilepath = csvfilepath

        if windows is not None:
            conditions = [(self.label['window start pt'] == w) for w in windows]
            combined_conditions = reduce(lambda x, y: x | y, conditions)

            self.label = self.label[combined_conditions].reset_index()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        csvfile = self.csvfilepath + self.label['csvname'][idx]
        csvname = self.label['csvname'][idx]
        sub = self.label['subject'][idx]
        tr = self.label['trial'][idx]
        wsp = self.label['window start pt'][idx]
        wep = self.label['window end pt'][idx]
        data = pd.read_csv(csvfile).to_numpy().transpose()[:, wsp - 1:wep]
        data = torch.tensor(data, dtype=torch.float32)

        return data, csvname, sub, tr, wsp


if __name__ == '__main__':
    # label = "G:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\s13\\s13_overlap1_label.csv"
    # csvfilepath = "G:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\s13\\csvdata\\"

    label = f"G:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\Multitasking\\Multitasking_overlap10_label.csv"
    csvfilepath = f"G:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\Multitasking\\csvdata\\"

    # dataset = myDataset(label=label, csvfilepath=csvfilepath, WOI=True)
    #
    # dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    #
    # for idx, (data, name) in enumerate(dataloader):
    #     print(f'{idx} ---- {data.size()} --- {name}')
    #     print(data)

    subjects = 1
    trials = 100
    windows = [1, 101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 1101, 1201, 1301, 1401]

    dataset = myinferenceDataset(label=label, csvfilepath=csvfilepath, windows=windows)

    dataloader = DataLoader(dataset=dataset, batch_size=16, collate_fn=collate_fn)

    print(len(dataset))

    for idx, (data, name, sub, tr, wsp) in enumerate(dataloader):
        print(f'{idx} ---- {data.size()} --- {name} --- {sub} --- {tr} --- {wsp}')

    print(len(dataset))
