import concurrent.futures
import multiprocessing
import sys
import threading
from functools import partial
from pathlib import Path

import numpy as np

from behavior.data import get_data


def save_gps_times(save_path, device_id, gps_info):
    items = []
    for item in gps_info:
        items.append(str(item[1]) + "\n")
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


"""
Fatemeh start here:
- save file with existing entries
"""

'''
# 1. Save GPS dates per device id: 18,903,265. 671 empty
save_path = Path("/home/fatemeh/Downloads/bird/gpsdates")
save_path.mkdir(parents=True, exist_ok=True)
database_url = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
device_start_end_query = """
select device_info_serial, start_date, end_date 
from gps.ee_track_session_limited
"""
results = query_database(database_url, device_start_end_query)

count = 0
for result in results:
    device_id = result[0]
    device_query = f"""
    SELECT *
    FROM gps.ee_tracking_speed_limited
    WHERE device_info_serial = {device_id} 
    order by date_time
    """
    try:
        gps_info = query_database(database_url, device_query)
        count += len(gps_info)
        save_gps_dates(save_path, device_id, gps_info)
        print(device_id, f"{len(gps_info):,}", str(gps_info[0][1]))
    except:
        print("empty ====>", device_id)
        continue
print(f"total: {count:,}")
'''
"""
# 2.1 Check which device has 60 entries: took 45 minutes to run
np.random.seed(3840)
import time
import concurrent.futures
glen = 60
directory = Path("/home/fatemeh/Downloads/bird/gpsdates")
output_file = Path(f"/home/fatemeh/Downloads/bird/ssl/available_{glen}points.csv")
database_url = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
num_entries = 10
paths = sorted(directory.glob("*csv"), key=lambda x:int(x.stem))

TIMEOUT = 2 # in seconds
with open(output_file, 'w') as file:
    for path in paths:
        items = read_dates(path)
        random_entries = get_random_entries(items, num_entries)
        device_id = int(path.stem)
        max_length = 0
        count = 0
        t0 = time.time()
        for item in random_entries:
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(get_data, database_url, device_id, item, item, glen)
                    gimus, idts, llat = future.result(timeout=TIMEOUT)
                    
                if len(gimus):
                    count += 1
                    max_length = max(idts[:, 0])
                # print(device_id, item, gimus.shape, max(idts[:, 0]))
            except concurrent.futures.TimeoutError:
                # print(f"Timeout: device {device_id}, item {item} took longer than {TIMEOUT} seconds.")
                continue
            except Exception as e:
                # print(f"Error processing device {device_id}, item {item}: {e}")
                continue
        excecute_time = time.time() - t0
        print(device_id, count, excecute_time)
        file.write(f"{device_id},{count},{max_length},{excecute_time:.1f}\n")
        file.flush()
"""
"""
#  2.2 Check which device has 60 entries: 59 devices
output_file = Path("/home/fatemeh/Downloads/bird/ssl/available_60points.csv")
with open(output_file, 'r') as file:
    items = file.read().strip().splitlines()
    device_ids = [int(i.split(',')[0]) for i in items if int(i.split(',')[1])!=0]

# 3. Save random device id and dates in a file: 2,717,252 from 59 devices
directory = Path("/home/fatemeh/Downloads/bird/gpsdates")
output_file = Path("/home/fatemeh/Downloads/bird/ssl/random_entries.csv")
num_entries = 2_717_252
all_lines = read_dates_from_files(directory, device_ids)
random_entries = get_random_entries(all_lines, num_entries)
save_random_entries_to_file(output_file, random_entries)
"""

""" 
# TODO removed
database_url = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
device_id = 298
p = Path(f"/home/fatemeh/Downloads/bird/gpsdates/{device_id}.csv")
dates = read_dates(p)
dates = get_random_entries(dates, 30)
for date in dates:
    try:
        gimus, idts, llat = bd.get_data(database_url, device_id, date, date, glen=60)
        print(f"{date}, {idts[0,0]},{idts.shape[0]}")
    except Exception as e:
        print(f"Error processing device {device_id}, item {date}: {e}")
        continue
print("done")
"""
"""
# TODO remove
database_url = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
output_file = Path("/home/fatemeh/Downloads/bird/ssl/my_file.csv")
label = -1
device_id = 658# 298
p = Path(f"/home/fatemeh/Downloads/bird/gpsdates/{device_id}.csv")
dates = read_dates(p)
dates = get_random_entries(dates, 30)
file = open(output_file, 'w')
for date in dates:
    try:
        gimus, idts, llat = get_data(database_url, device_id, item, item, glen=60)
        print(f"{date}, {idts[0,0]},{idts.shape[0]}")
        for gimu, idt in zip(gimus, idts):
            item = f"{device_id},{date},{int(idt[0])},{label},{gimu[0]:.8f},{gimu[1]:.8f},{gimu[2]:.8f},{gimu[3]:.8f}\n"
            file.write(item)
        file.flush()
        print(f"{item}, {idts[0,0]},{idts.shape[0]}")
    except Exception as e:
        print(f"Error processing device {device_id}, item {item}: {e}")
        continue
file.close()
"""
'''
# TODO remove??
# Example get data from random time and device: single example
database_url = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
device_start_end_query = """
select device_info_serial, start_date, end_date 
from gps.ee_track_session_limited
"""
results = query_database(database_url, device_start_end_query)
time_intervals = generate_random_time_intervals(str(results[0][1]), str(results[0][2]), 10)
gimus, idts, llat = get_data(database_url, results[0][0], str(time_intervals[0][0]), str(time_intervals[0][1]))
max_datetime = datetime.strptime('2024-07-08 15:01', "%Y-%m-%d %H:%M")


# Example get data from random time and device: single example
save_file = Path("/home/fatemeh/Downloads/bird/tmp/file_ftime.csv")
# rfile = open(save_file, 'a')
count = 0
num_devices = len(results)
# for i in np.random.randint(0, num_devices, 20):
#     result = results[i]
for result in results:
    device_id, start_time, end_time = result
    if max_datetime < end_time:
        end_time = max_datetime
    if result[0] not in [640, 642, 659, 672, 676]:
        continue
    time_intervals = generate_random_time_intervals(str(start_time), str(end_time), 10, interval_minutes=30)
    for time_interval in time_intervals:
        try:
            gimus, idts, llat = get_data(database_url, result[0], str(time_interval[0]), str(time_interval[1]), glen=60)
            # if max(idts[:,0]) != 59: # For 60 data point
            #     continue
            # # append_to_csv(save_file, gimus, idts)
            # count += 1
            # if count == 120_000:
            #     break
            # rfile.write(f"{result[0]}, {time_interval[0]}, {time_interval[1]}, {gimus.shape[0]}, {max(idts[:,0])}\n")
            print(result[0], time_interval[0], time_interval[1], gimus.shape, max(idts[:,0]))
        except:
            # rfile.write(f"{result[0]}, {time_interval[0]}, {time_interval[1]}\n")
            print(result[0], time_interval[0], time_interval[1])
            continue
# rfile.close()
print('wait')
'''


def process_dates(dates, output_file, device_id, database_url, label):
    file = open(output_file, "w")
    for date in dates:
        try:
            gimus, idts, _ = get_data(database_url, device_id, date, date, glen=60)
            print(f"{date}, {idts[0,0]},{idts.shape[0]}")
            for gimu, idt in zip(gimus, idts):
                line = f"{device_id},{date},{int(idt[0])},{label},{gimu[0]:.8f},{gimu[1]:.8f},{gimu[2]:.8f},{gimu[3]:.8f}\n"
                file.write(line)
            file.flush()
        except Exception as e:
            print(f"Error processing device {device_id}, date {date}: {e}")
            continue
    file.close()


def main(task_id):
    device_ids = [658, 298]

    # Ensure task_id is within range
    if task_id < 0 or task_id >= len(device_ids):
        print(f"Invalid task ID: {task_id}")
        return

    device_id = device_ids[task_id]
    save_path = Path(
        f"/home/fatemeh/Downloads/bird/ssl"
    )  # /zfs/omics/personal/fkarimi/ssl
    save_path.mkdir(parents=True, exist_ok=True)

    # database_url = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
    label = -1
    n_entries = 10
    n_div = n_entries // 2
    p = Path(
        f"/home/fatemeh/Downloads/bird/gpsdates/{device_id}.csv"
    )  # /home/fkarimi/data/gpsdates
    dates = read_dates(p)
    dates = get_random_entries(dates, n_entries)

    dates_outputfile = []
    for i in range(2):
        sel_dates = dates[i * n_div : i * n_div + n_div]
        output_file = output_file = save_path / f"{device_id}_{i}.csv"
        dates_outputfile.append((sel_dates, output_file, device_id))

    partial_process_dates = partial(
        process_dates, database_url=database_url, label=label
    )
    with multiprocessing.Pool(processes=2) as pool:
        results = pool.starmap(partial_process_dates, dates_outputfile)

    print("done", flush=True)


if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)