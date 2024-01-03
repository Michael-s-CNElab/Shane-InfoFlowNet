import pandas as pd
from torch import nn
import torch


def cosine():
    data = pd.read_csv(
        "C:\\Users\\user\\CausalityByAttention\\data\\s13_overlap1\\csvdata\\001_001.csv").to_numpy().transpose()
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).cuda()

    x = torch.rand([1, 3, 100]).cuda()
    data = torch.randn([1, 3, 100]).cuda()

    cosloss = nn.CosineSimilarity(dim=2, eps=1e-6)
    loss = 1-cosloss(data, x).mean()

    print(loss)


if __name__ == '__main__':
    cosine()
