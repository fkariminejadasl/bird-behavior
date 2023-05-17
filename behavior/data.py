# df["gpsRecord"][0]["longitude"] # df["longitude"].values[0]
import json
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset

"""
about data:
Per location, there is 20 accelaration measurements and the time stamp and gps speed are the same
for that location. For now, I use accelarations and gps speed as features (4 features). 
I might encode time stamps as a 5th feature. I think I need to sort them. the 
naive way might not work.

all_measurements = N x L X C = 1420 x 20 x 4
labels: portion of N
{'SitStand': 352, 'Flap': 268, 'Float': 235, 'Soar': 203, 'TerLoco': 126, 'Pecking': 81, 'Boat': 63, 'Manouvre': 60, 'Other': 10, 'ExFlap': 4}
labels: percentage
{'SitStand': 25.1, 'Flap': 19.1, 'Float': 16.7, 'Soar': 14.4, 'TerLoco': 8.9, 'Pecking': 5.7, 'Boat': 4.4, 'Manouvre': 4.2, 'Other': 0.7, 'ExFlap': 0.2}
device_ids: portion of N
{608: 85, 805: 281, 806: 36, 871: 25, 781: 12, 782: 302, 798: 68, 754: 21, 757: 112, 534: 180, 533: 32, 537: 42, 541: 70, 606: 136}
time stamps:
[datetime.utcfromtimestamp(time/1000).strftime('%Y-%m-%d %H:%M:%S') for time in time_stamps]
labels:ids:
{0: 'Flap', 1: 'ExFlap', 2: 'Soar', 3: 'Boat', 4: 'Float', 5: 'SitStand', 6: 'TerLoco', 7: 'Other', 8: 'Manouvre', 9: 'Pecking'}
{'ExFlap', 'Soar', 'Float', 'Manouvre', 'Pecking', 'SitStand', 'Flap', 'Other', 'TerLoco', 'Boat'}
labels: ids, label: percentage, ids: percentage
{'Boat': 3, 'ExFlap': 1, 'Flap': 0, 'Float': 4, 'Manouvre': 8, 'Other': 7, 'Pecking': 9, 'SitStand': 5, 'Soar': 2, 'TerLoco': 6}
{'Boat': 0.044, 'ExFlap': 0.002, 'Flap': 0.191, 'Float': 0.167, 'Manouvre': 0.042, 'Other': 0.007, 'Pecking': 0.057, 'SitStand': 0.251, 'Soar': 0.144, 'TerLoco': 0.089}
{0: 0.191, 1: 0.002, 2: 0.144, 3: 0.044, 4: 0.167, 5: 0.251, 6: 0.089, 7: 0.007, 8: 0.042, 9: 0.057}

label_to_ids = bd.map_label_id_to_label(label_ids, labels)[1]
label_counts = bd.get_stat_len_measures_per_label(all_measurements, labels)
label_pers = {k:int((v/sum(label_counts.values()))*1000)/1000 for k, v in label_counts.items()}

label_to_ids = dict(sorted(label_to_ids.items(), key=lambda x: x[0]))
label_pers = dict(sorted(label_pers.items(), key=lambda x: x[0]))
id_pers = dict(zip(label_to_ids.values(), label_pers.values()))
id_pers = dict(sorted(id_pers.items(), key=lambda x: x[0]))
"""

train_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data/train_set.json")
valid_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data/validation_set.json")
test_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data/test_set.json")
# labels, label_ids, device_ids, time_stamps, all_measurements = read_data(train_path)


def get_labels_weights(train_path):
    # Here is weights for loss calculation. I can also do sample weights in WeightedRandomSampler.
    labels, label_ids, device_ids, time_stamps, all_measurements = read_data(train_path)

    label_ids = np.array(label_ids)
    max_id = label_ids.max() + 1
    class_weights = np.zeros(max_id, dtype=np.float32)
    for i in range(max_id):
        class_weights[i] = np.float32(1 / (label_ids == i).sum())
    return class_weights


def plot_measurements_per_label(labels, all_measurements, uniq_label="Soar"):
    label_measurs = []
    for label, measur in zip(labels, all_measurements):
        if label == uniq_label:
            label_measurs.extend(measur)
    label_measurs = np.array(label_measurs)
    fig, axs = plt.subplots(3, 1)
    plt.suptitle(uniq_label, x=0.5)
    [axs[i].plot(label_measurs[:, i], "*") for i in range(3)]
    plt.show(block=False)


def plot_data_distribution(all_measurements):
    data = all_measurements[..., :3].reshape(-1, 3).copy()
    ndata = (data - data.min(0)) / (data.max(0) - data.min(0))
    fig, axs = plt.subplots(3, 1)
    plt.suptitle("data", x=0.5)
    [axs[i].plot(data[:, i], "*") for i in range(3)]
    plt.show(block=False)
    fig, axs = plt.subplots(3, 1)
    plt.suptitle("normalized data", x=0.5)
    [axs[i].plot(ndata[:, i], "*") for i in range(3)]
    plt.show(block=False)


def plot_some_data(labels, label_ids, device_ids, time_stamps, all_measurements):
    unique_labels = set(labels)
    len_data = all_measurements.shape[0]
    inds = np.random.permutation(len_data)
    data = all_measurements
    all_measurements = (data - data.min(0)) / (data.max(0) - data.min(0))
    for unique_label in unique_labels:
        n_plots = 0
        for ind in inds:
            if labels[ind] == unique_label:
                ms_loc = all_measurements[ind]
                # fig, axs = plt.subplots(1, 4)
                # axs[0].plot(ms_loc[:, 0], "-*")
                # axs[1].plot(ms_loc[:, 1], "-*")
                # axs[2].plot(ms_loc[:, 2], "-*")
                # axs[3].plot(ms_loc[:, 3], "-*")
                # plt.suptitle(
                #     f"label:{labels[ind]}_{label_ids[ind]}_device:{device_ids[ind]}_time:{time_stamps[ind]}",
                #     x=0.5,
                # )

                fig, ax = plt.subplots(1, 1)
                ax.plot(ms_loc[:, 0], "r-*", ms_loc[:, 1], "b-*", ms_loc[:, 2], "g-*")
                plt.title(
                    f"label:{labels[ind]}_{label_ids[ind]}_device:{device_ids[ind]}_time:{time_stamps[ind]}"
                )

                plt.show(block=False)
                n_plots += 1
                if n_plots == 10:
                    break
        plt.waitforbuttonpress()
        plt.close("all")


def get_stat_len_measures_per_label(all_measurements, labels):
    unique_labels = set(labels)
    stats = dict([(unique_label, 0) for unique_label in unique_labels])
    for ind in range(len(all_measurements)):
        stats[labels[ind]] += 1
    return stats


def get_stat_device_ids_lengths(device_ids):
    """Per bird it says how many times the measurements are recorded.
    e.g. for id=608 has 85 recordings with the same id.
    So for 20 measurements per location, there is 20 x 85 measurements.
    """
    unique_device_ids = set(device_ids)
    stats = dict([(unique_device_id, 0) for unique_device_id in unique_device_ids])
    for id_ in unique_device_ids:
        same_ids = [device_id for device_id in device_ids if device_id == id_]
        stats[id_] = len(same_ids)
    return stats


def map_label_id_to_label(label_ids, labels):
    ids_labels = dict()
    labels_ids = dict()
    for label_id, label in zip(label_ids, labels):
        ids_labels[label_id] = label
        labels_ids[label] = label_id
    return ids_labels, labels_ids


def get_per_location_measurements(item):
    measurements = []
    for measurement in item:
        measurements.append(
            [
                measurement["x"],
                measurement["y"],
                measurement["z"],
                measurement["gpsSpeed"],
            ]
        )
    return measurements


def read_data(json_path: Path | str):
    with open(json_path, "r") as rfile:
        data = json.load(rfile)

    labels = []
    label_ids = []
    device_ids = []
    time_stamps = []
    all_measurements = []
    for item in data:
        label = item["labelDetail"]["description"]
        label_id = item["labelDetail"]["labelId"] - 1  # change it to zero based
        device_id = item["gpsRecord"]["deviceId"]
        time_stamp = item["gpsRecord"]["timeStamp"]
        measurements = get_per_location_measurements(item["gpsRecord"]["measurements"])
        labels.append(label)
        label_ids.append(label_id)
        device_ids.append(device_id)
        time_stamps.append(time_stamp)
        all_measurements.append(measurements)
    all_measurements = np.array(all_measurements)
    return labels, label_ids, device_ids, time_stamps, all_measurements


class BirdDataset(Dataset):
    def __init__(self, json_path: Path, transform=None):
        (
            labels,
            self.label_ids,
            device_ids,
            time_stamps,
            all_measurements,
        ) = read_data(json_path)
        # TODO: for now remove gpsSpeed
        data = all_measurements[:, :, :3].copy()
        tdata = (data.reshape(-1, 3) - data.reshape(-1, 3).min(0)) / (
            data.reshape(-1, 3).max(0) - data.reshape(-1, 3).min(0)
        )
        self.data = tdata.reshape(data.shape)
        # self.data = all_measurements.copy()
        self.data = self.data.astype(np.float32)

        self.transform = transform

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, ind):
        data = self.data[ind].transpose((1, 0))  # LxC -> CxL
        label = self.label_ids[ind]

        if self.transform:
            data = self.transform(data)

        return data, label


# 'ExFlap': 4, 'Other': 10, 'Manouvre': 60, 'Boat': 63, 'Pecking': 81
# [1, 7, 8, 3, 9] -> change

# a = [1 if i in [1, 7, 8, 3, 9] else i for i in label_ids]
# a = [3 if i==6 else i for i in a]
