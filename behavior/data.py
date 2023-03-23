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
{'Soar': 203, 'Float': 235, 'Other': 10, 'Manouvre': 60, 'Pecking': 81, 'Flap': 268, 'ExFlap': 4, 'SitStand': 352, 'TerLoco': 126, 'Boat': 63}
device_ids: portion of N
{608: 85, 805: 281, 806: 36, 871: 25, 781: 12, 782: 302, 798: 68, 754: 21, 757: 112, 534: 180, 533: 32, 537: 42, 541: 70, 606: 136}
time stamps:
[datetime.utcfromtimestamp(time/1000).strftime('%Y-%m-%d %H:%M:%S') for time in time_stamps]
labels:
{0: 'Flap', 1: 'ExFlap', 2: 'Soar', 3: 'Boat', 4: 'Float', 5: 'SitStand', 6: 'TerLoco', 7: 'Other', 8: 'Manouvre', 9: 'Pecking'}
"""
json_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data/train_set.json")
# labels, label_ids, device_ids, time_stamps, all_measurements = read_data(json_path)


def plot_some_data(labels, label_ids, device_ids, time_stamps, all_measurements):
    unique_labels = set(labels)
    len_data = all_measurements.shape[0]
    inds = np.random.permutation(len_data)
    for unique_label in unique_labels:
        n_plots = 0
        for ind in inds:
            if labels[ind] == unique_label:
                ms_loc = all_measurements[ind]
                fig, axs = plt.subplots(1, 4)
                axs[0].plot(ms_loc[:, 0], "-*")
                axs[1].plot(ms_loc[:, 1], "-*")
                axs[2].plot(ms_loc[:, 2], "-*")
                axs[3].plot(ms_loc[:, 3], "-*")
                plt.suptitle(
                    f"label:{labels[ind]}_{label_ids[ind]}_device:{device_ids[ind]}_time:{time_stamps[ind]}",
                    x=0.5,
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
            self.all_measurements,
        ) = read_data(json_path)
        self.transform = transform

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, ind):
        data = self.all_measurements[ind].transpose((1, 0))  # LxC -> CxL
        label = self.label_ids[ind]

        if self.transform:
            data = self.transform(data)

        return data, label
