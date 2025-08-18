from collections import Counter
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, confusion_matrix
from torch.utils.data import DataLoader, random_split

from behavior import data as bd
from behavior import map as bmap
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu

cfg = dict(
    seed=32984,
    glen=20,
    exp=125,
    labels_to_use=[0, 1, 2, 3, 4, 5, 6, 8, 9],
    in_channe=4,
    width=30,
    n_classes=None,
    data_file=Path("/home/fatemeh/Downloads/bird/data/ssl/final/gull/298.csv"),
    checkpoint_file=Path(f"/home/fatemeh/Downloads/bird/result"),
)

cfg = SimpleNamespace(**cfg)
cfg.n_classes = len(cfg.labels_to_use)
cfg.checkpoint_file = cfg.checkpoint_file / f"{cfg.exp}_best.pth"

bu.set_seed(cfg.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_confmat():
    # fmt: off
    expected = np.array([
       [ 638,    1,    0,    0,    0,    0,    0,    4,    0],
       [   1,   36,    0,    0,    0,    0,    0,    1,    0],
       [   0,    0,  515,    4,   11,    0,    0,    7,    0],
       [   0,    0,   11,  157,    1,    0,    2,    0,    5],
       [   0,    0,   10,    0,  719,    0,    0,    0,    0],
       [   0,    0,    7,    0,    2, 1475,    9,    0,    9],
       [   0,    0,    2,    0,    0,    5,  327,    0,    3],
       [   2,    1,   26,    0,    0,    0,    0,  121,    1],
       [   0,    0,    4,    1,    0,   14,   26,    0,  180]])
    # fmt: on
    seed = 32984
    glen = 20
    exp = 125
    labels_to_use = [0, 1, 2, 3, 4, 5, 6, 8, 9]
    in_channel, width, n_classes = 4, 30, len(labels_to_use)
    data_file = Path("/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv")
    checkpoint_file = Path(f"/home/fatemeh/Downloads/bird/result/{exp}_best.pth")

    bu.set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = bm.BirdModel(in_channel, width, n_classes).to(device)
    bm.load_model(checkpoint_file, model, device)
    model.eval()

    df = pd.read_csv(data_file, header=None)
    df = df[df[3].isin(labels_to_use)].reset_index(drop=True)
    df[3] = df[3].map({lab: i for i, lab in enumerate(labels_to_use)})
    igs = df[[4, 5, 6, 7]].values.reshape(-1, glen, 4)
    labels = df.iloc[::glen, 3].values

    dataset = bd.BirdDataset(igs)
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    data = next(iter(loader))

    probs, preds = bu.inference(data, model, device)
    label_names = [bu.ind2name[i] for i in labels_to_use]
    confmat = confusion_matrix(labels, preds, labels=np.arange(len(label_names)))
    assert np.testing.assert_equal(confmat, expected)


# test_confmat()


def get_behavior_classes():
    pass


def visualize_imu_behavior():
    pass


def visualize_gps_behavior():
    pass


class Mapper:
    def __init__(self, old2new: dict):
        # old is a list like [0,2,4,5,6,9], new is [0, ..., 5]
        self.old2new = old2new
        self.new2old = {n: o for o, n in old2new.items()}

    def encode(self, orig):
        """Map original labels → 0…K-1 space"""
        return np.array([self.old2new[int(i)] for i in orig])

    def decode(self, chang):
        """Map 0…K-1 predictions back → original labels"""
        return np.array([self.new2old[int(i)] for i in chang])


# Model
model = bm.BirdModel(4, 30, cfg.n_classes).to(device)
bm.load_model(cfg.checkpoint_file, model, device)
model.eval()

# Data
df = pd.read_csv(cfg.data_file, header=None)
df = df.sort_values([0, 1, 2])
igs = df[[4, 5, 6, 7]].values.reshape(-1, cfg.glen, 4)
dataset = bd.BirdDataset(igs)
loader = DataLoader(
    dataset,
    batch_size=len(dataset),
    shuffle=False,
    num_workers=1,
    drop_last=False,
)
data = next(iter(loader))

# Predictions
probs, preds = bu.inference(data, model, device)
mapper = Mapper({l: i for i, l in enumerate(cfg.labels_to_use)})
preds = mapper.decode(preds)

# Visualize IMU and behavior
pred_df = df.copy()
pred_df.iloc[:, 3] = preds[:, np.newaxis].repeat(cfg.glen, axis=1).reshape(-1)

dt = (298, "2010-06-07 09:43:05")

unique_dt = pred_df.groupby(by=[0, 1])
dataframe = unique_dt.get_group(dt)
fig = bu.plot_labeled_data(dataframe, dataframe, bu.ind2name)
plt.title(f"{dt[0]},{dt[1]},{dataframe.iloc[0,7]:.2f}")

# visualize map
lat, lon = 52.00947, 4.34438
map_image = bmap.get_centered_map_image(lat, lon, zoom=15)
map_image.show()

# TODO
# Download GPS data from database

print("done")
