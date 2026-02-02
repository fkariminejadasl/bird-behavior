import concurrent.futures
import multiprocessing
import os
import sys
import threading
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from behavior import data as bd
from behavior.data import get_data

database_url = ""
# database_url = "postgresql://username:password@host:port/database_name"


def save_gps_times(save_path, device_id, gps_info):
    items = []
    for item in gps_info:
        items.append(str(item[0]) + "\n")
    save_file = save_path / f"{device_id}.csv"
    with open(save_file, "w") as rfile:
        rfile.writelines(items)


def read_dates(file_path):
    with open(file_path, "r") as file:
        dates = file.read().splitlines()
    return dates


def read_dates_from_files(directory, device_ids=None):
    if device_ids is None:
        device_ids = [int(p.stem) for p in Path(directory).glob("*.csv")]
    all_lines = []
    for file_path in Path(directory).glob("*.csv"):
        device_id = int(file_path.stem)
        if device_id not in device_ids:
            continue
        with file_path.open("r") as file:
            dates = file.read().splitlines()
            for time in dates:
                all_lines.append((device_id, time))
    return all_lines


def get_random_entries(all_lines, num_entries):
    indices = np.random.choice(len(all_lines), size=num_entries, replace=False)
    random_entries = [all_lines[i] for i in indices]
    return random_entries


def save_random_entries_to_file(output_file, random_entries):
    with open(output_file, "w") as file:
        for device_id, time in random_entries:
            file.write(f"{device_id},{time}\n")


def read_random_entries(file_path):
    with open(file_path, "r") as file:
        lines = file.read().splitlines()
        random_entries = [tuple(line.split(",")) for line in lines]
    return random_entries


def get_device_query(device_id):
    device_query = f"""
	select date_time
	from gps.ee_tracking_speed_limited
	where device_info_serial = {device_id} 
	order by date_time"""
    return device_query


"""
# Compare the previous number of the GPS dates with the current one 
results = bd.query_database(database_url, query)
device_ids = sorted([i[0] for i in results])
gps_dates = dict()
for device_id in device_ids:
    results = bd.query_database(database_url, get_device_query(device_id))
    gps_dates[device_id] = len(results)
    print(device_id, len(results))


saved_gps_dates = dict()
main_path = Path("/home/fatemeh/Downloads/bird/data/ssl/gpsdates")
for p in main_path.glob("*.csv"):
    c = open(p).readlines()
    device_id = int(p.stem)
    saved_gps_dates[device_id] = len(c)

for device_id, l in gps_dates.items():
    if device_id in saved_gps_dates:
        if saved_gps_dates[device_id] != l:
            print(device_id)
"""

"""
# Check if we put per data item (20 rows) in one row how much memory we save about 30%
# row: device, time, (index, imu_x, imu_y, imu_z, gps)x20
# Code is not optimized and very slow
def reshape_df(df):
    rows = []
    for i in range(0, len(df), 20):
        block = df.iloc[i:i+20]
        if len(block) < 20:  # skip incomplete blocks
            continue
        
        # start with first row's col0 and col1
        new_row = block.iloc[0, [0, 1]].tolist()
        
        # then add [2,4,5,6,7] for each of the 20 rows
        for _, r in block.iterrows():
            new_row.extend(r[[2,4,5,6,7]].tolist())
        
        rows.append(new_row)
    
    # Create new dataframe
    new_df = pd.DataFrame(rows)
    return new_df

import pandas as pd
p = Path("/home/fatemeh/Downloads/bird/data/ssl/675.csv")
device_id = int(p.stem)
df = pd.read_csv(p, header=None) # df/20=132_855
reshaped = reshape_df(df)
reshaped.to_csv("/home/fatemeh/Downloads/bird/data/ssl/row_item_675.csv", index=False)
"""

'''
# Get only Gulls
query = f"""
select key_name, device_info_serial
from gps.ee_track_session_limited
order by device_info_serial
"""
results = bd.query_database(database_url, query)
g_devices = [result[1] for result in results if result[0] in ["CG_KREUPEL", "LBBG_TEXEL", "HG_TEXEL"]]
g_devices = [45, 50, 51, 52, 53, 54, 121, 130, 132, 133, 134, 135, 297, 298, 304, 311, 317, 319, 320, 324, 325, 326, 327, 329, 344, 355, 373, 533, 534, 535, 536, 537, 538, 540, 541, 542, 604, 606, 608, 752, 754, 757, 781, 782, 784, 798, 805, 806, 868, 870, 871, 1600, 5387, 5388, 5390, 5391, 5392, 5393, 5415, 5416, 5472, 5473, 5496, 5586, 5587, 5588, 5590, 5591, 5592, 5593, 5594, 5599, 5601, 5603, 5687, 5689, 5690, 5692, 5693, 5694, 5697, 5698, 5699, 5700, 5701, 5702, 5704, 5705, 5992, 5993, 5995, 5996, 5997, 5998, 5999, 6004, 6006, 6009, 6011, 6012, 6014, 6015, 6016, 6017, 6071, 6072, 6073, 6074, 6075, 6076, 6077, 6079, 6080, 6082, 6202, 6205, 6206, 6208, 6210, 6212, 6214, 6216, 6217, 6219, 6387, 6392, 6395, 6396, 6397, 6399, 6400, 6401, 6402, 6403, 6501, 6502, 6503, 6504, 6506, 6507, 6508, 6509, 6510, 6511, 6512, 6513, 6514, 6515, 6516, 6517, 6518, 6519, 6520, 6521, 7001, 7002, 7004, 7007, 7008, 7009, 7021, 7036, 7045, 7046, 7047, 7048, 7050, 7055, 7056, 7058, 7059, 7061, 7062, 7063, 7077, 7078, 7079, 7080, 7081, 7083, 7085, 7086, 7120, 7123, 7126, 7131, 7132, 7133, 7135, 7137, 7138, 7139, 7140, 7142, 7145, 7147, 7148, 7149, 7160, 7161, 7162]
'''

'''
# Discard Empty IMUs
# Still too slow. Just run the get imu and gps data instead of this.
from datetime import datetime, timedelta
def get_gps_query(device_id):
    gps_query = f"""
    SELECT  min(date_time), max(date_time), count(date_time)
    FROM gps.ee_tracking_speed_limited
    WHERE device_info_serial = {device_id}
    """
    return gps_query

def get_imu_query(device_id, start_date, end_date):
    imu_query = f"""
    SELECT count(index)
    FROM gps.ee_acceleration_limited
    WHERE device_info_serial = {device_id} and date_time between '{start_date}' and '{end_date}'
    """
    return imu_query


def is_device_empty(device_id):
    DEVICE_EMPTY = False
    query = get_gps_query(device_id)
    results = bd.query_database(database_url, query)[0]
    if results[-1] != 0:
        gps_sdate = results[0]
        gps_edate = results[1]
        # .replace(microsecond=0)
        start_date = str(gps_sdate + (gps_edate - gps_sdate) / 2 + timedelta(days=0))
        end_date = str(gps_sdate + (gps_edate - gps_sdate) / 2 + timedelta(days=365))
        print(device_id, gps_sdate, gps_edate, start_date, end_date)
        query = get_imu_query(device_id, start_date, end_date)
        counts = bd.query_database(database_url, query)[0][0]
        if counts == 0:
            DEVICE_EMPTY = True
    return DEVICE_EMPTY

empty_devices = []
for device_id in g_devices:
    print(device_id)
    DEVICE_EMPTY = is_device_empty(device_id)
    if DEVICE_EMPTY:
        print("empty: ", device_id)
        empty_devices.append(device_id)

no_imu = [45, 50, 51, 52, 53, 54, 130, 132, 135, 297, 317, 325, 326, 327, 329, 344, 373, 533, 534, 537, 540, 541, 754, 757, 781]
nan_calibration = [121, 133, 134]
twice_in_ee_track_session_limited = [6009, 7001, 7021]

# device_id, start_date, end_date, n_dates = 6210, "2015-05-23 09:22:21", "2022-08-12 16:42:15", 1562699, # 30716994 imu rows
'''


def get_gps_dates_per_device(device_id, database_url, save_path):
    count = 0
    device_query = f"""
    SELECT date_time
    FROM gps.ee_tracking_speed_limited
    WHERE device_info_serial = {device_id} 
    order by date_time
    """
    try:
        gps_info = bd.query_database_improved(database_url, device_query)
        count += len(gps_info)
        save_gps_times(save_path, device_id, gps_info)
        print(device_id, f"{len(gps_info):,}", str(gps_info[0][0]))
    except:
        print("empty ====>", device_id)
    print(f"total: {count:,}")


# Time range, per device downlaod:
# ====================
def process_dates(device_id, s_date, e_date, output_file, database_url, label):
    tmp = str(output_file) + ".tmp"

    try:
        # run the heavy DB work first; leave the filesystem alone if this fails
        gimus, idts, _ = get_data(database_url, device_id, s_date, e_date, glen=20)
        print(f"{device_id},{s_date},{e_date},{idts[0,0]},{idts.shape[0]}", flush=True)

        # write atomically
        with open(tmp, "w") as f:
            for gimu, idt in zip(gimus, idts):
                date = datetime.fromtimestamp(idt[2], tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                line = f"{device_id},{date},{int(idt[0])},{label},{gimu[0]:.8f},{gimu[1]:.8f},{gimu[2]:.8f},{gimu[3]:.8f}\n"
                f.write(
                    line
                )  # Puts the data into Python’s in-memory buffer for that file
            f.flush()  # Pushes Python’s buffer down to the OS kernel buffer (but not guaranteed on disk yet)
            os.fsync(
                f.fileno()
            )  # Forces the OS to flush its buffer and actually write the data to disk

        os.replace(tmp, output_file)  # atomic move -> no 0-byte finals
    except Exception as e:
        # if we created a tmp, clean it up; the final file name won't exist/overwrite
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        print(
            f"Error processing device {device_id}, start date {s_date}, end date {e_date}: {e}",
            flush=True,
        )


def main(device_id):
    save_path = Path(f"/home/fatemeh/Downloads/bird/data/ssl/tmp")
    save_path.mkdir(parents=True, exist_ok=True)

    label = -1
    p = Path(f"/home/fatemeh/Downloads/bird/data/ssl/gpsdates2/{device_id}.csv")
    dates = read_dates(p)
    n_entries = len(dates)
    n_div = 21000  # 10000#20736#253627 #47872# 12000
    n_files = int(np.ceil(n_entries / n_div))

    # List of dates_outputfile [device, start date, end date, ouputpath].
    # Item e.g: (658, '2011-12-12 23:45:57', '2011-12-12 23:56:48', Path('298_0.csv')
    dates_outputfile = []
    for i in range(n_files):
        sel_dates = dates[i * n_div : i * n_div + n_div]
        output_file = save_path / f"{device_id}_{i}.csv"
        print(i, device_id, sel_dates[0], sel_dates[-1])
        dates_outputfile.append((device_id, sel_dates[0], sel_dates[-1], output_file))

    partial_process_dates = partial(
        process_dates, database_url=database_url, label=label
    )

    # use a small fixed pool instead of processes=n_files
    MAX_WORKERS = 8
    with multiprocessing.Pool(
        processes=min(n_files, MAX_WORKERS), maxtasksperchild=1
    ) as pool:
        pool.starmap(partial_process_dates, dates_outputfile, chunksize=1)

    print("done", flush=True)


if __name__ == "__main__":
    # fmt: off
    device_ids = [45, 50, 51, 52, 53, 54, 121, 130, 132, 133, 134, 135, 297, 298, 304, 311, 317, 319, 320, 324, 325, 326, 327, 329, 344, 355, 373, 533, 534, 535, 536, 537, 538, 540, 541, 542, 604, 606, 608, 752, 754, 757, 781, 782, 784, 798, 805, 806, 868, 870, 871]
    device_ids = [1600, 5387, 5388, 5390, 5391, 5392, 5393, 5415, 5416, 5472, 5473, 5496, 5586, 5587, 5588, 5590, 5591, 5592, 5593, 5594, 5599, 5601, 5603, 5687, 5689, 5690, 5692, 5693, 5694, 5697, 5698, 5699, 5700, 5701, 5702, 5704, 5705, 5992, 5993, 5995, 5996, 5997, 5998, 5999]
    device_ids = [6004, 6006, 6009, 6011, 6012, 6014, 6015, 6016, 6017, 6071, 6072, 6073, 6074, 6075, 6076, 6077, 6079, 6080, 6082, 6202, 6205, 6206, 6208, 6210, 6212, 6214, 6216, 6217, 6219, 6387, 6392, 6395, 6396, 6397, 6399, 6400, 6401, 6402, 6403, 6501, 6502, 6503, 6504, 6506, 6507, 6508, 6509, 6510, 6511, 6512, 6513, 6514, 6515, 6516, 6517, 6518, 6519, 6520, 6521]
    device_ids = [7001, 7002, 7004, 7007, 7008, 7009, 7021, 7036, 7045, 7046, 7047, 7048, 7050, 7055, 7056, 7058, 7059, 7061, 7062, 7063, 7077, 7078, 7079, 7080, 7081, 7083, 7085, 7086, 7120, 7123, 7126, 7131, 7132, 7133, 7135, 7137, 7138, 7139, 7140, 7142, 7145, 7147, 7148, 7149, 7160, 7161, 7162]
    # fmt: on

    # # Download GPS times
    # total_t0 = time.perf_counter()
    # for device_id in tqdm(device_ids): #6210
    #     t0 = time.perf_counter()
    #     save_path = Path("/home/fatemeh/Downloads/bird/data/ssl/gpsdates2")
    #     get_gps_dates_per_device(device_id, database_url, save_path)
    #     print(f"{device_id} took {time.perf_counter()-t0:.2f}s")
    # print(f"total time {time.perf_counter()-total_t0:.2f}s")

    # # Download IMU and GPS
    # total_t0 = time.perf_counter()
    # for device_id in tqdm(device_ids): #6210
    #     t0 = time.perf_counter()
    #     main(device_id)
    #     print(f"{device_id} took {time.perf_counter()-t0:.2f}s")
    # print(f"total time {time.perf_counter()-total_t0:.2f}s")
