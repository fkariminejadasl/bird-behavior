from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from behavior import data as bd

data_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data")
train_path = data_path / "train_set.json"
valid_path = data_path / "validation_set.json"
test_path = data_path / "test_set.json"

labels1, label_ids1, device_ids1, time_stamps1, all_measurements1 = bd.read_data(
    train_path
)
labels2, label_ids2, device_ids2, time_stamps2, all_measurements2 = bd.read_data(
    valid_path
)
labels3, label_ids3, device_ids3, time_stamps3, all_measurements3 = bd.read_data(
    test_path
)
device_ids = device_ids1 + device_ids2 + device_ids3
time_stamps = time_stamps1 + time_stamps2 + time_stamps3

td = np.stack([time_stamps, device_ids]).T
td = td[td[:, 0].argsort()]
# date_strings = [datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") for ts in td[:,0]]
date_strings = [datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") for ts in td[:, 0]]

frequency_count = Counter(date_strings)
dates = list(frequency_count.keys())
frequencies = list(frequency_count.values())

plt.figure(figsize=(10, 6))
# plt.bar(dates, frequencies)
plt.bar(date_strings, td[:, 1])
plt.xlabel("Times")
plt.ylabel("Device Ids")
plt.title("Times Device_id Plot")
plt.xticks(rotation=45)
plt.tight_layout()
unique_device = set(td[:, 1])
for freq in unique_device:
    plt.axhline(y=freq, color="r", linestyle="--", linewidth=0.8)
plt.axhline(y=533, color="g", linestyle="-", linewidth=0.8)
plt.show(block=False)
print("something")

"""
import json
def get_per_location_measurements(item):
    measurements = []
    for measurement in item:
        measurements.append(
            [
                measurement["x"],
                measurement["y"],
                measurement["z"],
                measurement["gpsSpeed"],
                measurement["index"],
            ]
        )
    return measurements

def read_data2(json_path):
    with open(json_path, "r") as rfile:
        data = json.load(rfile)

    labels = []
    label_ids = []
    device_ids = []
    time_stamps = []
    all_measurements = []
    first_indexs = []
    for item in data:
        label = item["labelDetail"]["description"]
        label_id = item["labelDetail"]["labelId"] - 1  # change it to zero based
        device_id = item["gpsRecord"]["deviceId"]
        first_index = item["gpsRecord"]["firstIndex"]
        time_stamp = item["gpsRecord"]["timeStamp"] % 1000
        measurements = get_per_location_measurements(item["gpsRecord"]["measurements"])
        labels.append(label)
        label_ids.append(label_id)
        device_ids.append(device_id)
        time_stamps.append(time_stamp)
        all_measurements.append(measurements)
        first_indexs.append(first_index)
    # all_measurements = np.array(all_measurements)
    return labels, label_ids, device_ids, time_stamps, all_measurements, first_indexs

labels1, label_ids1, device_ids1, time_stamps1, all_measurements1, first_indexs1 = read_data2(train_path)
labels2, label_ids2, device_ids2, time_stamps2, all_measurements2, first_indexs2 = read_data2(valid_path)
labels3, label_ids3, device_ids3, time_stamps3, all_measurements3, first_indexs3 = read_data2(test_path)
"""

x_o, x_s, y_o, y_s, z_o, z_s = 419.4, 1282.6, 99.47, 1322.97, -201.18, 1322.15


def raw2meas(x_m, y_m, z_m):
    x_a = (x_m - x_o) / x_s
    y_a = (y_m - y_o) / y_s
    z_a = (z_m - z_o) / z_s
    return x_a, y_a, z_a


device_ids = device_ids1 + device_ids2 + device_ids3
time_stamps = time_stamps1 + time_stamps2 + time_stamps3
label_ids = label_ids1 + label_ids2 + label_ids3
label_device_times = np.stack((label_ids, device_ids, time_stamps)).T
all_measurements = np.concatenate(
    (all_measurements1, all_measurements2, all_measurements3), axis=0
)
# 91,  446, 1313, 2012-05-15 11:38:20 (446, 1313, 91)
inds = np.where(
    (label_device_times[:, 1] == 533) & (label_device_times[:, 2] == 1337081900)
)[0]
# np.set_printoptions(precision=4, suppress=True)
b = np.concatenate(all_measurements[inds, :, :3], axis=0)
for ind in inds:
    print(
        datetime.utcfromtimestamp(label_device_times[ind, 2]).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        ),
        ind,
    )

print("otherthing")


data = np.stack((device_ids, time_stamps, all_measurements[:, 0, -1])).T
sorted_indices = np.lexsort((data[:, 0], data[:, 1]))
sorted_data = data[sorted_indices]
# sorted_data = sorted(data, key=lambda x: (x[0], x[1]))
with open(data_path / "times.txt", "w") as file:
    for dev_id, ts, gps in sorted_data:
        f_time = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")
        _ = file.write(f"{dev_id},{f_time},{gps}\n")

"""
# check the if the set2 is subset of set1
for i in range(len(sorted_data2)):
    a = np.where((sorted_data[:,0] == sorted_data2[i,0])&(sorted_data[:,1] == sorted_data2[i,1])&(sorted_data[:,2] == sorted_data2[i,2]))[0]
    if len(a) == 0:
        print(i)
"""
