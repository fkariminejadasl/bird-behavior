import gc
import math
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
from omegaconf import OmegaConf
from torch.nn.functional import normalize
from torch.utils import tensorboard
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as tvt2

from behavior import data as bd
from behavior import data_augmentation as bau
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.utils import new_label_inds


def inference(data, model, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        data = data.to(device)  # N x C x L
        outputs = model(data)  # N x C
        # prob = torch.nn.functional.softmax(outputs, dim=-1)  # N x C
        pred = torch.argmax(outputs.data, 1)  # N

    prob = outputs.cpu()  # .numpy()
    pred = pred.cpu().numpy()
    return prob, pred


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


train_transform = tvt2.RandomChoice(
    [
        bau.RandomJitter(sigma=0.05),
        bau.RandomScaling(sigma=0.05),
        # bau.TimeWarp(sigma=0.05),
        # bau.MagnitudeWarp(sigma=0.05, knot=4),
    ]
)
train_transform = ContrastiveLearningViewGenerator(
    base_transform=train_transform, n_views=2
)

cfg = dict(
    test_data_file=Path("/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv"),
    model_checkpoint=Path("/home/fatemeh/Downloads/bird/result/125_best.pth"),
    labels_to_use=[0, 1, 2, 3, 4, 5, 6, 8, 9],
    channel_first=False,
    in_channel=4,
    mid_channel=30,
    out_channel=9,
    seed=123,
    batch_size=4338,
)
cfg = OmegaConf.create(cfg)

bu.set_seed(cfg.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(cfg.test_data_file, header=None)
df = df[df[3].isin(cfg.labels_to_use)]
mapping = {l: i for i, l in enumerate(cfg.labels_to_use)}
df[3] = df[3].map(mapping)
all_measurements = df[[4, 5, 6, 7]].values.reshape(-1, 20, 4)
label_ids = df[[3, 0, 0]].iloc[::20].values
trainset = bd.BirdDataset(
    all_measurements,
    label_ids,
    channel_first=cfg.channel_first,
    transform=train_transform,
)
loader = torch.utils.data.DataLoader(
    trainset, batch_size=len(trainset), shuffle=False, num_workers=4, drop_last=False
)
a = next(iter(loader))

model = bm.BirdModel(cfg.in_channel, 30, cfg.out_channel)
bm.load_model(cfg.model_checkpoint, model, device)

b1 = inference(a[0][0].transpose(2, 1), model, device)[0][3]  # 8
b2 = inference(a[0][1].transpose(2, 1), model, device)[0][3]  # 8
b2 = inference(a[0][0].transpose(2, 1), model, device)[0][4]  # 6
b2 = inference(a[0][0].transpose(2, 1), model, device)[0][5]  # 8


logits0, preds0 = inference(a[0][0].transpose(2, 1), model, device)
logits1, preds1 = inference(a[0][1].transpose(2, 1), model, device)
idxs6 = np.where(label_ids == 6)[0]
idxs8 = np.where(label_ids == 8)[0]
idxs = idxs6
b1 = logits0
b2 = logits1
vals = []
for i in range(len(idxs)):
    val = normalize(b1[idxs[i]], dim=-1, p=2).dot(normalize(b2[idxs[i]], dim=-1, p=2))
    val = round(val.item(), 2)
    vals.append(val)
# preds[idxs[np.argsort(vals)]]
# np.array(vals)[np.argsort(vals)]
cosim = torch.nn.functional.cosine_similarity(b1[idxs], b2[idxs], dim=1)
sidxs = cosim.sort()[1].numpy()
preds0[idxs[sidxs]]
preds1[idxs[sidxs]]
cosim[sidxs]
print("Done")

"""
bu.plot_one(df.iloc[:20, 4:7].values)
bu.plot_one(a[0][0][0])
bu.plot_one(a[0][1][0])
a[0][0].shape # torch.Size([64, 20, 4])
np.array(df.iloc[::20,3])[:10] # equal label_ids[:10,0]
"""

"""
image_size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
interpolation = 3
crop_pct = 0.875
train_transform = transforms.Compose([
    transforms.Resize(int(image_size / crop_pct), interpolation),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=torch.tensor(mean),
        std=torch.tensor(std))
])
train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=2)
trainset = torchvision.datasets.CIFAR10(root='/home/fatemeh/data', train=True,download=True, transform=train_transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
a = next(iter(loader))
"""
