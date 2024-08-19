import csv
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

directory = Path("/home/fatemeh/Downloads/bird/ssl/final")
# directory = Path("/home/fatemeh/Downloads/bird/ssl/tmp3")


# Pandas
t0 = time.perf_counter()
gimus = []
dis = []
timestamps = []
for csv_file in directory.glob("*.csv"):
    df = pd.read_csv(csv_file, header=None)
    gimus.append(df[[4, 5, 6, 7]].values)
    dis.append(df[[0, 2]].values)
    timestamps.extend(df[1].tolist())

gimus = np.concatenate(gimus, axis=0)
dis = np.concatenate(dis, axis=0)
timestamps = np.array(timestamps)
print("pandas", f"{time.perf_counter() - t0:.2f}", gimus.shape)


# CSV reading
t0 = time.perf_counter()
gimus = []
dis = []
timestamps = []
for csv_file in directory.glob("*.csv"):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            gimus.append([float(row[i]) for i in [4, 5, 6, 7]])  # gps, imus
            dis.append([int(row[i]) for i in [0, 2]])  # device id, indices
            timestamps.append(row[1])

gimus = np.array(gimus)
dis = np.array(dis)
timestamps = np.array(timestamps)
print("csv   ", f"{time.perf_counter() - t0:.2f}", gimus.shape)

# Pure python
t0 = time.perf_counter()
gimus = []
dis = []
timestamps = []
for csv_file in directory.glob("*.csv"):
    with open(csv_file, "r") as file:
        for row in file:
            row = row.strip().split(",")
            gimus.append([float(row[i]) for i in [4, 5, 6, 7]])  # gps, imus
            dis.append([int(row[i]) for i in [0, 2]])  # device id, indices
            timestamps.append(row[1])
    gc.collect()

gimus = np.array(gimus)
dis = np.array(dis)
timestamps = np.array(timestamps)
print("python", f"{time.perf_counter() - t0:.2f}", gimus.shape)
