from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from behavior import data as bd

data_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data")
train_path = data_path / "train_set.json"
valid_path = data_path / "validation_set.json"
test_path = data_path / "test_set.json"

labels1, label_ids1, device_ids1, time_stamps1, all_measurements1 = bd.read_json_data(
    train_path
)
labels2, label_ids2, device_ids2, time_stamps2, all_measurements2 = bd.read_json_data(
    valid_path
)
labels3, label_ids3, device_ids3, time_stamps3, all_measurements3 = bd.read_json_data(
    test_path
)
device_ids = device_ids1 + device_ids2 + device_ids3
time_stamps = time_stamps1 + time_stamps2 + time_stamps3

td = np.stack([time_stamps, device_ids]).T
td = td[td[:, 0].argsort()]
date_strings = [
    datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") for ts in td[:, 0]
]

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
        datetime.fromtimestamp(label_device_times[ind, 2], tz=timezone.utc).strftime(
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
        f_time = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
        _ = file.write(f"{dev_id},{f_time},{gps}\n")

"""
# check the if the set2 is subset of set1
for i in range(len(sorted_data2)):
    a = np.where((sorted_data[:,0] == sorted_data2[i,0])&(sorted_data[:,1] == sorted_data2[i,1])&(sorted_data[:,2] == sorted_data2[i,2]))[0]
    if len(a) == 0:
        print(i)
"""

from datetime import datetime, timezone
from pathlib import Path

data_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data")
ms, ldts = bd.load_all_data_from_json(data_path / "combined.json")
# extraannotations_533_15052012.csv
with open(
    "/home/fatemeh/Downloads/bird/judy_annotations/extraannotations_533_15052012.csv",
    "r",
) as f:
    _ = f.readline()
    for r in f:
        i = int(r.split(",")[0])
        t = int(
            datetime.strptime(r.split(",")[1], "%Y-%m-%dT%H:%M:%S.%fZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        inds = np.where((ldts[:, 1] == i) & (ldts[:, 2] == t))[0]
        if len(inds) != 0:
            print(r)
            print(ldts[inds])
