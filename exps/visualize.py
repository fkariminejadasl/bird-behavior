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
