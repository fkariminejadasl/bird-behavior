# df["gpsRecord"][0]["longitude"] # df["longitude"].values[0]
import json
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np


def plot_some_data(labels, label_ids, device_ids, time_stamps, all_measurements):
    # {'Manouvre', 'Boat', 'SitStand', 'Flap', 'Pecking', 'ExFlap', 'Float', 'Other', 'Soar', 'TerLoco'}
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


def get_some_statistics(all_measurements, labels):
    unique_labels = list(set(labels))
    stats = dict([(unique_label, 0) for unique_label in unique_labels])
    for ind in range(len(all_measurements)):
        stats[labels[ind]] += 1
    return stats


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
        label_id = item["labelDetail"]["labelId"]
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


# all_measurements = N x L X C = 1420 x 20 x 4
# len(data)=1420 * len(measurements)=20 * 4
json_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data/train_set.json")
# labels, label_ids, device_ids, time_stamps, all_measurements = read_data(json_path)
