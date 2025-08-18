import os
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

seed = 32984
bu.set_seed(seed)


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


def fetch_gps_data(database_url, device_id, start_time, end_time):
    """
    Fetch GPS data from the database.
    `device_info_serial, date_time, latitude, longitude, altitude
    """

    sql_query = f"""
    SELECT *
    FROM gps.ee_tracking_speed_limited
    WHERE device_info_serial = {device_id} AND date_time BETWEEN '{start_time}' AND '{end_time}'
    ORDER BY date_time
    """
    results = bd.query_database_improved(database_url, sql_query)
    if len(results) == 0:
        raise ValueError("No GPS data found")

    return [
        [
            result[0],  # device_info_serial
            str(result[1]),  # date_time
            result[2],  # latitude
            result[3],  # longitude
        ]
        for result in results
        if result[-4] is not None  # speed_2d
    ]


def infer_update_classes(df, glen, labels_to_use, checkpoint_file, n_classes):
    """
    Inference and update classes
    -> df is mutated
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    model = bm.BirdModel(4, 30, n_classes).to(device)
    bm.load_model(checkpoint_file, model, device)
    model.eval()

    # Data
    igs = df[[4, 5, 6, 7]].values.reshape(-1, glen, 4)
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
    mapper = Mapper({l: i for i, l in enumerate(labels_to_use)})
    preds = mapper.decode(preds)

    # Change dataframe
    df.iloc[:, 3] = preds[:, np.newaxis].repeat(glen, axis=1).reshape(-1)
    max_probs = np.max(probs, axis=1)
    df[8] = max_probs[:, np.newaxis].repeat(glen, axis=1).reshape(-1)

    return df


def fetch_merge_gps(df, database_url):
    """
    Fetch GPS data and merge with IMU data
    -> df is mutated
    """
    start_time = df.iloc[0, 1]
    end_time = df.iloc[-1, 1]
    device_id = df.iloc[0, 0]
    # gps_data = fetch_gps_data(database_url, 298, '2010-06-07 09:43:05', '2010-06-07 09:53:17')
    gps_data = fetch_gps_data(database_url, device_id, start_time, end_time)
    gps_data = pd.DataFrame(gps_data)
    df = df.merge(gps_data, on=[0, 1], how="inner")
    df = df.rename(columns={"2_x": 2, "3_x": 3, "2_y": 9, "3_y": 10})

    return df


def prepare_imu_gps_class_data(data_file, save_file, cfg):
    df = pd.read_csv(data_file, header=None)
    df = df.sort_values([0, 1, 2])
    df = infer_update_classes(
        df, cfg.glen, cfg.labels_to_use, cfg.checkpoint_file, cfg.n_classes
    )
    df = fetch_merge_gps(df, cfg.database_url)
    df.to_csv(save_file, index=False, header=None, float_format="%.6f")


# Prepare data
cfg = dict(
    glen=20,
    exp=125,
    labels_to_use=[0, 1, 2, 3, 4, 5, 6, 8, 9],
    in_channe=4,
    width=30,
    n_classes=None,
    data_path=Path("/home/fatemeh/Downloads/bird/data/ssl/final/gull"),
    save_path=Path("/home/fatemeh/Downloads/bird/data/ssl/gimu_behavior/gull"),
    checkpoint_file=Path(f"/home/fatemeh/Downloads/bird/result"),
    database_url=None,
)

cfg = SimpleNamespace(**cfg)
cfg.n_classes = len(cfg.labels_to_use)
cfg.checkpoint_file = cfg.checkpoint_file / f"{cfg.exp}_best.pth"
cfg.database_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@pub.e-ecology.nl:5432/eecology"
cfg.save_path.mkdir(parents=True, exist_ok=True)

"""
data_paths = sorted(list(cfg.data_path.glob("*")), key=lambda x:int(x.stem))
for data_file in data_paths:
    device_id = int(data_file.stem)
    save_file = cfg.save_path/f"{device_id}.csv"
    prepare_imu_gps_class_data(data_file, save_file, cfg)
"""


# Visualize IMU and behavior
gimu_beh_file = Path("/home/fatemeh/Downloads/bird/data/ssl/gimu_behavior/gull/298.csv")
df = pd.read_csv(gimu_beh_file, header=None)
dt = (298, "2010-06-07 09:43:05")

unique_dt = df.groupby(by=[0, 1])
dataframe = unique_dt.get_group(dt)
fig = bu.plot_labeled_data(dataframe, dataframe, bu.ind2name)
plt.title(f"{dt[0]},{dt[1]},{dataframe.iloc[0,7]:.2f}")

# Visualize map
lat, lon = df.iloc[0, [9, 10]]  # 52.00947, 4.34438
map_image = bmap.get_centered_map_image(lat, lon, zoom=15)
map_image.show()

print("done")
