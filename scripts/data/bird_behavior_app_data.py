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
from behavior import model_utils as bmu
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
    data_file = Path("/home/fatemeh/Downloads/bird/data/final/starts.csv")
    checkpoint_file = Path(f"/home/fatemeh/Downloads/bird/results/{exp}_best.pth")

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


def fetch_merge_gps(df, database_url):
    """
    Fetch GPS data and merge with IMU data
    -> df is mutated
    """

    groups = df.groupby([0])
    gps_data = []
    for k, group in groups:
        start_time = group.iloc[0, 1]
        end_time = group.iloc[-1, 1]
        device_id = group.iloc[0, 0]
        per_group = fetch_gps_data(database_url, device_id, start_time, end_time)
        gps_data.extend(per_group)
    gps_data = pd.DataFrame(gps_data)

    # Change dataframe: append columns at the end
    last_col = int(df.columns[-1])
    df = df.merge(gps_data, on=[0, 1], how="inner")
    df = df.rename(
        columns={"2_x": 2, "3_x": 3, "2_y": last_col + 1, "3_y": last_col + 2}
    )

    return df


def prepare_imu_gps_class_data(data_file, save_file, cfg):
    df = pd.read_csv(data_file, header=None)
    df = df.sort_values([0, 1, 2])
    df = bmu.infer_update_classes(
        df, cfg.glen, cfg.labels_to_use, cfg.checkpoint_file, cfg.n_classes
    )
    df = fetch_merge_gps(df, cfg.database_url)
    df.to_csv(save_file, index=False, header=None, float_format="%.6f")


def prepare_rose_app_data(data_file: Path, save_file: Path) -> None:
    cols = [
        "device_id",
        "UTC_datetime",
        "datatype",
        "Latitude",
        "Longitude",
        "Altitude_m",
        "speed_km_h",
        "x_g",
        "y_g",
        "z_g",
    ]

    df = pd.read_csv(data_file, usecols=cols)
    df["UTC_datetime"] = pd.to_datetime(df["UTC_datetime"])

    is_gps = df["datatype"].eq("GPS")
    is_sensor = df["datatype"].eq("SENSORS")

    gps_next_sensor = (
        is_gps
        & df["datatype"].shift(-1).eq("SENSORS")
        & df["device_id"].eq(df["device_id"].shift(-1))
        & ~(df["Latitude"].eq(0) & df["Longitude"].eq(0))
    )

    time_diff = df["UTC_datetime"].shift(-1) - df["UTC_datetime"]

    valid_gps = gps_next_sensor & (time_diff.abs() <= pd.Timedelta(seconds=2))

    # The latest GPS row before each SENSOR row.
    df["gps_datetime"] = df["UTC_datetime"].where(is_gps)
    df["gps_datetime"] = df.groupby("device_id")["gps_datetime"].ffill()

    # The latest valid GPS row before each SENSOR row.
    df["valid_gps_datetime"] = df["UTC_datetime"].where(valid_gps)
    df["valid_gps_datetime"] = df.groupby("device_id")["valid_gps_datetime"].ffill()

    # Keep GPS values only from valid GPS rows, then forward-fill them.
    gps_cols = ["Latitude", "Longitude", "Altitude_m", "speed_km_h"]

    for col in gps_cols:
        df[col] = df[col].where(valid_gps)
        df[col] = df.groupby("device_id")[col].ffill()

    # Keep SENSOR rows only when the most recent GPS is valid.
    df_app = df[
        is_sensor
        & df["gps_datetime"].eq(df["valid_gps_datetime"])
        & df["Latitude"].notna()
        & df["Longitude"].notna()
    ].copy()

    # Keep only rows up to a count divisible by 20 per device_id and SENSOR datetime.
    df_app["index"] = df_app.groupby(["device_id", "UTC_datetime"]).cumcount()

    n_per_group = df_app.groupby(["device_id", "UTC_datetime"])["index"].transform(
        "size"
    )
    keep_n = n_per_group - (n_per_group % 20)

    df_app = df_app[df_app["index"] < keep_n].copy()

    # Recreate index after filtering.
    df_app["index"] = df_app.groupby(["device_id", "UTC_datetime"]).cumcount()

    out = pd.DataFrame(
        {
            0: df_app["device_id"],
            1: df_app["UTC_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            2: df_app["index"],
            3: -1,
            4: df_app["x_g"],
            5: df_app["y_g"],
            6: df_app["z_g"],
            7: df_app["speed_km_h"] / 3.6,  # speed m/s
            8: -1,
            9: -1,
            10: df_app["Latitude"],
            11: df_app["Longitude"],
            12: df_app["Altitude_m"],
        }
    )

    out.to_csv(save_file, index=False, header=False, float_format="%.6f")


def check_consecutive_gps_sensor_time_diff(data_file):
    """
    Check the time difference between consecutive GPS and SENSORS rows for the same device_id.
    """
    cols = [
        "device_id",
        "UTC_datetime",
        "datatype",
        "Latitude",
        "Longitude",
        "Altitude_m",
        "speed_km_h",
        "x_g",
        "y_g",
        "z_g",
    ]

    df = pd.read_csv(data_file, usecols=cols)
    df["UTC_datetime"] = pd.to_datetime(df["UTC_datetime"])

    gps_next_sensor = (
        df["datatype"].eq("GPS")
        & df["datatype"].shift(-1).eq("SENSORS")
        & df["device_id"].eq(df["device_id"].shift(-1))
        & ~(df["Latitude"].eq(0) & df["Longitude"].eq(0))
    )

    time_diff = (
        df["UTC_datetime"].shift(-1)[gps_next_sensor]
        - df["UTC_datetime"][gps_next_sensor]
    )

    print(time_diff.value_counts().sort_index())
    print("max:", time_diff.max())

    print(
        df.loc[
            gps_next_sensor,
            ["device_id", "UTC_datetime", "Latitude", "Longitude", "Altitude_m"],
        ]
        .assign(
            gps_time=df["UTC_datetime"][gps_next_sensor],
            next_sensor_time=df["UTC_datetime"].shift(-1)[gps_next_sensor],
            time_diff=time_diff,
        )
        .sort_values("time_diff", ascending=False)
        .head(20)
    )

    print("done")


def plot_labeled_data(df, ind2name, glen=20):

    y_limits = [-3.5, 3.5]

    # Data
    df = df.sort_values([0, 1, 2]).reset_index(drop=True)
    assert len(np.unique(df[0])) == 1
    assert len(np.unique(df[1])) == 1
    assert len(np.unique(df[7])) == 1
    data = df[[4, 5, 6]].values  # IMU data
    indices = df[2].values  # Indices

    # Draw IMU data
    width = 4 * len(df) // glen + 2
    fig, ax = plt.subplots(1, 1, figsize=(width, 4.3))
    mgr = plt.get_current_fig_manager()
    # # x=100, y=100 on screen
    mgr.window.wm_geometry("+100+100")  # TkAgg
    # mgr.window.setGeometry(100, 100, width_px, height_px) # Qt5Agg
    ax.plot(
        indices,
        data[:, 0],
        "r-*",
        indices,
        data[:, 1],
        "b-*",
        indices,
        data[:, 2],
        "g-*",
    )

    # Set axis limits
    ax.set_xlim(indices[0], indices[-1] + 1)
    ax.set_ylim(*y_limits)
    ax.set_yticks([y_limits[0], 0, y_limits[1]])
    ax.set_xticks(range(indices[0], indices[-1] + glen, glen))

    # Draw horizontal zero line
    ax.plot([indices[0], indices[-1]], [0, 0], "-", color="black")

    # Draw vertical lines
    for i in range(0, len(df), glen):
        s_ind = df.iloc[i, 2]
        e_ind = s_ind + glen
        label_id = df.iloc[i, 3]
        pred_id = df.iloc[i, 8]
        prob = df.iloc[i, 9]
        label = ind2name[label_id] if label_id != -1 else None
        pred = ind2name[pred_id]
        text_loc = [s_ind + (e_ind - s_ind + 1) // 4, y_limits[1] - 0.5]
        text = f"{label}, {pred}, {prob:.3f}"
        ax.text(*text_loc, text, color="black", fontsize=9)
        ax.plot([s_ind, s_ind], y_limits, "-", color="black")
    ax.plot([e_ind, e_ind], y_limits, "-", color="black")

    text = f"{df.iloc[0,0]}, {df.iloc[0,1]},gps:{df.iloc[0,7]:.2f}"
    plt.title(f"{text}", fontsize=10)  # GPS info
    fig.tight_layout()
    # # ig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.bottom, fig.subplotpars.top
    # fig.subplots_adjust(left=.038,right=.97,bottom=.09,top=.95)
    return fig


if __name__ == "__main__":
    # Prepare data
    cfg = dict(
        glen=20,
        exp=125,
        labels_to_use=[0, 1, 2, 3, 4, 5, 6, 8, 9],
        in_channe=4,
        width=30,
        n_classes=None,
        checkpoint_file=Path(f"/home/fatemeh/Downloads/bird/results"),
        database_url=None,
    )

    cfg = OmegaConf.create(cfg)
    cfg.n_classes = len(cfg.labels_to_use)
    cfg.checkpoint_file = cfg.checkpoint_file / f"{cfg.exp}_best.pth"
    cfg.database_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@pub.e-ecology.nl:5432/eecology"

    # data_file = Path("/home/fatemeh/Downloads/bird/data/simon/all_devices_calibrated.csv")
    # save_file = Path("/home/fatemeh/Downloads/bird/data/simon/rose_data.csv")

    # prepare_rose_app_data(data_file, save_file)
    # check_consecutive_gps_sensor_time_diff(data_file)


"""
# On unlabeled data
data_path = Path("/home/fatemeh/Downloads/bird/data/ssl/final/gull")
save_path = Path("/home/fatemeh/Downloads/bird/data/ssl/gimu_behavior/gull")
save_path.mkdir(parents=True, exist_ok=True)
data_paths = sorted(list(data_path.glob("*")), key=lambda x: int(x.stem))
for data_file in data_paths:
    device_id = int(data_file.stem)
    save_file = save_path / f"{device_id}.csv"
    prepare_imu_gps_class_data(data_file, save_file, cfg)

# On the ground truth data
data_file = Path("/home/fatemeh/Downloads/bird/data/final/starts.csv")
save_file = Path(
    "/home/fatemeh/Downloads/bird/data/final/starts_gimu_behavior.csv"
)
prepare_imu_gps_class_data(data_file, save_file, cfg)
"""


"""
# Visualize IMU and behavior
# gimu_beh_file = Path("/home/fatemeh/Downloads/bird/data/ssl/gimu_behavior/gull/298.csv")
# dt = (298, "2010-06-07 09:43:05")
gimu_beh_file = Path("/home/fatemeh/Downloads/bird/data/final/starts_gimu_behavior.csv")
dt = (806, "2014-05-16 12:56:53")
dt = (6210, "2016-05-09 10:26:25")
df = pd.read_csv(gimu_beh_file, header=None)


groups = df.groupby(by=[0, 1])
dts = list(groups.groups.keys())
# for _ in range(5):
# i = np.random.randint(0, len(dts))
# print(i)
for i in [1536, 179]:
    dt = dts[i]
    dataframe = groups.get_group(dt)
    fig = plot_labeled_data(dataframe, bu.ind2name)

    # Visualize map
    lat, lon = dataframe.iloc[0, [10, 11]]  # 52.00947, 4.34438
    map_image = bmap.get_centered_map_image(lat, lon, zoom=15)
    map_image.show()

print("done")
"""


"""
simon steps:
 - create birds_full from all_devices_calibrated.csv
 - classify_birds -> no ll
 - merge_simon_data

bird_behavior_app_data: get GPS from database and append. Then it does inference.

Simpler for me to use bird_behavior_app_data or fix classify_birds for simon and others.

bird_behavior_app_data:
- functionality: getting data for three apps and visualization is optional
- add the Rose data reading and make it as device_id, date_time, imu_x, imu_y, imu_z, speed, lat, lon, alt
"""
