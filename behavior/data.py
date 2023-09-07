import json
from pathlib import Path
from typing import Union

import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset

np.random.seed(0)
"""
about data:
Per location, there is 20 accelaration measurements and the time stamp and gps speed are the same
for that location. For now, I use accelarations and gps speed as features (4 features). 
I might encode time stamps as a 5th feature. I think I need to sort them. the 
naive way might not work.

all_measurements = N x L X C = 1420 x 20 x 4 (train), train/valid/test:1402/1052/1051=3505 (old), ~3468 (new)

[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9] # classes
[268,   4, 203,  63, 235, 352, 126,  10,  60,  81] # train
[192,  13, 146,  60, 142, 280, 100,   6,  51,  62] # valid
[174,  21, 152,  53, 181, 262,  92,   9,  40,  67] # test
[634,  38, 501, 176, 558, 894, 318,  25, 151, 210] # total old
[657,  45, 505, 178, 563, 817, 345,  28, 143, 184] # total new

Flight
	0=Flap=634
	1=ExFlap=38
	2=Soar=501
	8=Manouvre/Mixed=151
Float
	4=Float=558
Sit-Stand
	5=SitStand/Stationary=894
	3=Boat=176
Terrestrial locomotion
	6=Terloco/Walk=318
	9=Pecking/Peck=210 (paper=209)
Other
	7=Other=25

# imu-gps stats
[-3.41954887, -3.03254572, -2.92030075,  0.        ] min
[ 2.68563855,  2.93441769,  3.23596358, 22.30123518] max -> 1
[-0.03496432, -0.10462683,  0.97874294,  4.00891258] mean -> 0.17976191
[0.34833199, 0.1877568 , 0.33700332, 5.04326992] std -> 0.22614308

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


def count_labels(label_ids):
    count = []
    for i in range(0, 10):
        count.append(sum([1 for label in label_ids if label == i]))
    return count


def get_labels_weights(label_ids):
    # Here is weights for loss calculation. I can also do sample weights in WeightedRandomSampler.
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


def read_data(json_path: Union[Path, str]):
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
        time_stamp = int(item["gpsRecord"]["timeStamp"] / 1000)
        measurements = get_per_location_measurements(item["gpsRecord"]["measurements"])
        labels.append(label)
        label_ids.append(label_id)
        device_ids.append(device_id)
        time_stamps.append(time_stamp)
        all_measurements.append(measurements)
    all_measurements = np.array(all_measurements)
    return labels, label_ids, device_ids, time_stamps, all_measurements


class BirdDataset(Dataset):
    def __init__(self, all_measurements: np.ndarray, label_ids: list, transform=None):
        self.label_ids = label_ids
        self.data = all_measurements.copy()
        # normalize gps speed
        # self.data[:, :, 3] = self.data[:, :, 3] / self.data[:, :, 3].max()
        self.data[:, :, 3] = (
            self.data[:, :, 3] / 22.3012351755624
        )  # all_measurements[..., 3].max()
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


class BirdDataset_old(Dataset):
    def __init__(self, json_path: Path, transform=None):
        (
            labels,
            self.label_ids,
            device_ids,
            time_stamps,
            all_measurements,
        ) = read_data(json_path)
        # TODO: for now remove gpsSpeed
        # data = all_measurements[:, :, :3].copy()
        # tdata = (data.reshape(-1, 3) - data.reshape(-1, 3).min(0)) / (
        #     data.reshape(-1, 3).max(0) - data.reshape(-1, 3).min(0)
        # )
        # self.data = tdata.reshape(data.shape)
        self.data = all_measurements.copy()
        # normalize gps speed
        self.data[:, :, 3] = self.data[:, :, 3] / self.data[:, :, 3].max()
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


# for new data
def read_csv_bird_file(data_file: Path):
    ids = []
    inds = []
    gps_imus = []
    labels = []
    confs = []
    with open(data_file, "r") as rfile:
        for row in rfile:
            items = row.split("\n")[0].split(",")
            id_ = int(items[0])
            ind = int(items[2])
            imu_x = float(items[3])
            imu_y = float(items[4])
            imu_z = float(items[5])
            gps_speed = float(items[6])
            item_last = items[7].split()
            label = int(item_last[0])
            conf = int(item_last[1])
            ids.append(id_)
            inds.append(ind)
            labels.append(label)
            confs.append(conf)
            gps_imus.append([imu_x, imu_y, imu_z, gps_speed])
    return ids, inds, gps_imus, labels, confs


def stats_per_csv_file(data_file, plot=False):
    ids, inds, gps_imus, labels, confs = read_csv_bird_file(data_file)
    if plot:
        agps_imus = np.array(gps_imus, dtype=np.float32)
        _, axs = plt.subplots(4, 1, sharex=True)
        axs[0].plot(inds)
        axs[1].plot(labels)
        axs[2].plot(confs)
        axs[3].plot(
            agps_imus[:, 0], "r-*", agps_imus[:, 1], "b-*", agps_imus[:, 2], "g-*"
        )
        plt.show(block=False)
    total_data = len(labels)
    num_no_labels = sum([1 for label in labels if label == 0])
    num_labels = total_data - num_no_labels
    num_data_points = num_labels / 20
    hist = list(np.histogram(labels, bins=range(0, 12))[0])
    print(data_file.stem)
    print(
        f"total: {total_data:6d}, # no labels: {num_no_labels:6d}, # labels: {num_labels:6d}, # data: {num_data_points}"
    )
    print(dict(zip(range(0, 11), hist)))
    return hist, num_data_points


"""
# get statistics for all data
data_dir = Path("/home/fatemeh/Downloads/bird/data")
csv_filename = "AnM533_20120515_20120516.csv"

count = 0
hists = []
files = list(data_dir.glob("*.csv"))
for data_file in files:
    hist, num_data_points = stats_per_csv_file(data_file, plot=False)
    count += num_data_points
    hists.append(hist)

hists = np.array(hists, dtype=np.int64)
print(hists[:, 1:])
print(dict(zip(range(1, 11), np.sum(hists[:, 1:], axis=0))))
print(count)

# from datetime import datetime
# datetime.utcfromtimestamp(1402132788).strftime('%Y-%m-%d %H:%M:%S')
# int(datetime.strptime('2014-06-07 11:19:48', '%Y-%m-%d %H:%M:%S').strftime("%s"))
"""


def combine_all_data():
    labels1, label_ids1, device_ids1, time_stamps1, all_measurements1 = read_data(
        train_path
    )
    labels2, label_ids2, device_ids2, time_stamps2, all_measurements2 = read_data(
        valid_path
    )
    labels3, label_ids3, device_ids3, time_stamps3, all_measurements3 = read_data(
        test_path
    )
    label_ids = label_ids1 + label_ids2 + label_ids3
    all_measurements = np.concatenate(
        (all_measurements1, all_measurements2, all_measurements3), axis=0
    )
    # labels = labels1 + labels2 + labels3
    # device_ids = device_ids1 + device_ids2 + device_ids3
    # time_stamps = time_stamps1 + time_stamps2 + time_stamps3
    inds = np.arange(all_measurements.shape[0])
    np.random.shuffle(inds)
    all_measurements = all_measurements[inds]
    label_ids = list(np.array(label_ids)[inds])
    return all_measurements, label_ids


def get_specific_labesl(all_measurements, label_ids, target_labels=[0, 2, 4, 5]):
    label_ids = np.array(label_ids, dtype=np.int64)
    agps_imus = np.empty(shape=(0, 20, 4))
    new_ids = np.empty(shape=(0,), dtype=np.int64)
    for i in target_labels:
        agps_imus = np.concatenate(
            (agps_imus, all_measurements[label_ids == i]), axis=0
        )
        new_ids = np.concatenate((new_ids, label_ids[label_ids == i]), axis=0)
    inds = np.arange(new_ids.shape[0])
    np.random.shuffle(inds)
    agps_imus = agps_imus[inds]
    new_ids = list(new_ids[inds])
    new_ids = reindex_ids(new_ids)
    return agps_imus, new_ids


def reindex_ids(label_ids):
    unique_ids = set(label_ids)
    label_ids = np.array(label_ids)
    new_ids = label_ids.copy()
    for i, unique_id in enumerate(unique_ids):
        new_ids[label_ids == unique_id] = i
    return list(new_ids)


all_measurements, label_ids = combine_all_data()
alabel_ids = np.array(label_ids, dtype=np.int64)
agps_imus = np.empty(shape=(0, 20, 4))
for i in range(0, 10):
    agps_imus = np.concatenate((agps_imus, all_measurements[alabel_ids == i]), axis=0)
agps_imus = agps_imus.reshape(-1, 4)
rep_labels = np.sort(np.repeat(alabel_ids, 20))

# agps_imus, new_ids = get_specific_labesl(
#     all_measurements, label_ids, target_labels=[0, 2, 4, 5]
# )
# agps_imus = agps_imus.reshape(-1, 4)
# rep_labels = np.repeat(new_ids, 20)

_, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(rep_labels)
axs[1].plot(agps_imus[:, 0], "r-*", agps_imus[:, 1], "b-*", agps_imus[:, 2], "g-*")
axs[2].plot(agps_imus[:, 3])
plt.show(block=False)

agps_imus[:, 3] = agps_imus[:, 3] / agps_imus[:, 3].max()
_, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(rep_labels)
axs[1].plot(agps_imus[:, 0], "r-*", agps_imus[:, 1], "b-*", agps_imus[:, 2], "g-*")
axs[2].plot(agps_imus[:, 3])
plt.show(block=False)

"""
def test(model, n=10):
    a = all_measurements[0:n].copy()
    a[..., 3] = a[..., 3] / 22.3012351755624
    a = torch.tensor(a.astype(np.float32)).transpose(2,1)
    out = model(a)
    print(torch.argmax(out, axis=1).tolist())
    print(label_ids[0:n])
"""
