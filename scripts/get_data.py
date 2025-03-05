import concurrent.futures
import multiprocessing
import sys
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

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


def read_random_entries(file_path):
    with open(file_path, "r") as file:
        lines = file.read().splitlines()
        random_entries = [tuple(line.split(",")) for line in lines]
    return random_entries


'''
# 1. Save GPS dates per device id: 18,903,265 (new 18,800,484). 671 empty
save_path = Path("/home/fatemeh/Downloads/bird/data/ssl/gpsdates")
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
        save_gps_times(save_path, device_id, gps_info)
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
output_file = Path(f"home/fatemeh/Downloads/bird/data/ssl/available_{glen}points.csv")
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
output_file = Path("home/fatemeh/Downloads/bird/data/ssl/available_60points.csv")
with open(output_file, 'r') as file:
    items = file.read().strip().splitlines()
    device_ids = [int(i.split(',')[0]) for i in items if int(i.split(',')[1])!=0]

# 3. Save random device id and dates in a file: 2,975,486 from 63 devices
directory = Path("/home/fatemeh/Downloads/bird/gpsdates")
output_file = Path("home/fatemeh/Downloads/bird/data/ssl/random_entries.csv")
all_lines = read_dates_from_files(directory, device_ids)
num_entries = len(all_lines)
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
output_file = Path("home/fatemeh/Downloads/bird/data/ssl/my_file.csv")
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

'''
# Single time query, per Device id download: Tasks are based on device ids
# low cpu usage (server high cpu usage due to overhead)
# ====================
def process_dates(dates, output_file, device_id, database_url, label):
    """
    dates: list[str]: eg. ['2010-06-14 16:21:51', '2010-06-02 04:36:12']
    outputfile: Path.
    device_id: int
    """
    file = open(output_file, "w")
    for date in dates:
        try:
            gimus, idts, _ = get_data(database_url, device_id, date, date, glen=60)
            print(f"{device_id},{date},{idts[0,0]},{idts.shape[0]}", flush=True)
            for gimu, idt in zip(gimus, idts):
                line = f"{device_id},{date},{int(idt[0])},{label},{gimu[0]:.8f},{gimu[1]:.8f},{gimu[2]:.8f},{gimu[3]:.8f}\n"
                file.write(line)
            file.flush()
        except Exception as e:
            print(f"Error processing device {device_id}, date {date}: {e}", flush=True)
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
        f"home/fatemeh/Downloads/bird/data/ssl"
    )  # /zfs/omics/personal/fkarimi/ssl
    save_path.mkdir(parents=True, exist_ok=True)

    # database_url = "postgresql://username:password@host:port/database_name"
    label = -1
    p = Path(
        f"/home/fatemeh/Downloads/bird/gpsdates/{device_id}.csv"
    )  # /home/fkarimi/data/gpsdates
    dates = read_dates(p)
    n_files = 1 
    n_entries = 200# len(dates)
    n_div = int(np.ceil(n_entries / n_files))
    dates = get_random_entries(dates, n_entries)

    # List of dates_outputfile_device_ids.
    # Item e.g: (['2010-06-14 16:21:51', '2010-06-02 04:36:12'], Path('298_0.csv'), 298)
    dates_outputfile = []
    for i in range(n_files):
        sel_dates = dates[i * n_div : i * n_div + n_div]
        output_file = save_path / f"{device_id}_{i}.csv"
        dates_outputfile.append((sel_dates, output_file, device_id))

    partial_process_dates = partial(
        process_dates, database_url=database_url, label=label
    )
    with multiprocessing.Pool(processes=n_files) as pool:
        results = pool.starmap(partial_process_dates, dates_outputfile)

    print("done", flush=True)

main(0)
if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)
'''

'''
# Sigle time query, download from random entries: 
# low cpu usage (server high cpu usage due to overhead)
# ====================
def process_dates(entries, output_file, database_url, label):
    """
    entries: list[tuple]: eg. [('446', '2011-03-28 18:23:18'), ...]
    outputfile: Path.
    """
    file = open(output_file, "w")
    for device_id, date in entries:
        try:
            gimus, idts, _ = get_data(database_url, int(device_id), date, date, glen=60)
            print(f"{device_id},{date},{idts[0,0]},{idts.shape[0]}", flush=True)
            for gimu, idt in zip(gimus, idts):
                line = f"{device_id},{date},{int(idt[0])},{label},{gimu[0]:.8f},{gimu[1]:.8f},{gimu[2]:.8f},{gimu[3]:.8f}\n"
                file.write(line)
            file.flush()
        except Exception as e:
            print(f"Error processing device {device_id}, date {date}: {e}")
            continue
    file.close()


def main(task_id):
    file_path = Path("home/fatemeh/Downloads/bird/data/ssl/random_entries.csv")
    save_path = Path(f"home/fatemeh/Downloads/bird/data/ssl")
    save_path.mkdir(parents=True, exist_ok=True)

    database_url = "postgresql://username:password@host:port/database_name"
    label = -1

    random_entries = read_random_entries(file_path)
    n_files = 1
    n_entries = 100  # len(random_entries)
    n_div = int(np.ceil(n_entries / n_files))

    # Split the entries into chunks
    chunked_entries = [
        random_entries[i * n_div : (i + 1) * n_div] for i in range(n_files)
    ]

    # Ensure task_id is within range
    if task_id < 0 or task_id >= n_files:
        print(f"Invalid task ID: {task_id}")
        return

    entries = chunked_entries[task_id]
    output_file = save_path / f"entries_{task_id}.csv"

    # No multiprocessing
    # process_dates(entries, output_file, database_url, label)

    # Use multiprocessing for parallel processing within the task
    partial_process_dates = partial(
        process_dates, database_url=database_url, label=label
    )
    num_processes = 4
    chunk_size = int(np.ceil(len(entries) / num_processes))
    entry_chunks = [
        entries[i * chunk_size : (i + 1) * chunk_size] for i in range(num_processes)
    ]

    # Directory for temporary files
    temp_dir = save_path / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_files = [temp_dir / f"temp_{task_id}_{i}.csv" for i in range(num_processes)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(partial_process_dates, zip(entry_chunks, temp_files))

    # Combine temporary files into the final output file
    with open(output_file, "w") as outfile:
        for temp_file in temp_files:
            with open(temp_file, "r") as infile:
                outfile.write(infile.read())
            temp_file.unlink()  # Remove temporary file after writing

    temp_dir.rmdir()  # Remove the temporary directory if empty

    print("done", flush=True)

main(0)
if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)

"""
# Slum with task array
#!/bin/bash

#SBATCH --job-name=result
#SBATCH --output=test_%A_%a.out
#SBATCH --error=test_%A_%a.err
#SBATCH --mem-per-cpu=1G #200G
#SBATCH --time=00:10:00 #UNLIMITED
#SBATCH --cpus-per-task=2
#SBATCH --array=0-1

source $HOME/.bashrc
conda activate p310
python ~/dev/bird-behavior/scripts/get_data.py $SLURM_ARRAY_TASK_ID
echo "finish in slurm"
"""
'''


from datetime import datetime, timezone


# Time range, per device downlaod:
# ====================
def process_dates(device_id, s_date, e_date, output_file, database_url, label):
    file = open(output_file, "w")
    try:
        gimus, idts, _ = get_data(database_url, device_id, s_date, e_date, glen=60)
        print(f"{device_id},{s_date},{e_date},{idts[0,0]},{idts.shape[0]}", flush=True)
        for gimu, idt in zip(gimus, idts):
            date = datetime.fromtimestamp(idt[2], tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            line = f"{device_id},{date},{int(idt[0])},{label},{gimu[0]:.8f},{gimu[1]:.8f},{gimu[2]:.8f},{gimu[3]:.8f}\n"
            file.write(line)
        file.flush()
    except Exception as e:
        print(
            f"Error processing device {device_id}, start date {s_date}, end date {e_date}: {e}"
        )
    file.close()


def main(device_id):
    save_path = Path(
        f"home/fatemeh/Downloads/bird/data/ssl"
    )  # /zfs/omics/personal/fkarimi/ssl
    save_path.mkdir(parents=True, exist_ok=True)

    database_url = "postgresql://username:password@host:port/database_name"
    label = -1
    p = Path(
        f"/home/fatemeh/Downloads/bird/gpsdates/{device_id}.csv"
    )  # /home/fkarimi/data/gpsdates
    dates = read_dates(p)
    n_entries = len(dates)
    # n_files = 1
    # n_div = int(np.ceil(n_entries / n_files))
    n_div = 21000  # 10000#20736#253627 #47872# 12000
    n_files = int(np.ceil(n_entries / n_div))

    # List of dates_outputfile [device, start date, end date, ouputpath].
    # Item e.g: (658, '2011-12-12 23:45:57', '2011-12-12 23:56:48', Path('298_0.csv')
    dates_outputfile = []
    for i in range(n_files):
        sel_dates = dates[i * n_div : i * n_div + n_div]
        output_file = save_path / f"{device_id}_{i}.csv"
        print(device_id, sel_dates[0], sel_dates[-1])
        dates_outputfile.append((device_id, sel_dates[0], sel_dates[-1], output_file))

    partial_process_dates = partial(
        process_dates, database_url=database_url, label=label
    )
    with multiprocessing.Pool(processes=n_files) as pool:
        results = pool.starmap(partial_process_dates, dates_outputfile)

    # Combine temporary files into the final output file
    combined_outfile = dates_outputfile[0][3].parent / f"{device_id}.csv"
    with open(combined_outfile, "w") as outfile:
        for _, _, _, output_file in dates_outputfile:
            with open(output_file, "r") as infile:
                outfile.write(infile.read())
            output_file.unlink()  # Remove temporary file after writing

    print("done", flush=True)


if __name__ == "__main__":
    avail_file = Path("home/fatemeh/Downloads/bird/data/ssl/available_60points.csv")
    with open(avail_file, "r") as file:
        items = file.read().strip().splitlines()
        device_ids = [int(i.split(",")[0]) for i in items if int(i.split(",")[1]) != 0]
    # fmt: off
    # [298, 304, 311, 320, 325, 327, 344, 446, 533, 534, 536, 537, 538, 541, 542, 604, 608, 640, 642, 644, 646, 657, 658, 659, 660, 661, 662, 663, 672, 674, 675, 676, 680, 681, 682, 683, 690, 752, 754, 781, 782, 784, 798, 868, 870, 1600, 2008, 2112, 2113, 2114, 2116, 2117, 2118, 2119, 2120, 2121, 6004, 6009, 6011, 6012, 6014, 6015, 6016]
    # fmt: on

    # device_ids = [6016]#[298, 534, 658, 6004, 6016] #[658, 298]
    # device_ids = [298]
    for device_id in tqdm(device_ids):
        t0 = time.perf_counter()
        main(device_id)
        print(f"{device_id} took {time.perf_counter()-t0:.2f}s")

'''
# Separate Gull and CP files
# ====================
from behavior import data as bd, model as bm, model1d as bm1, utils as bu
import shutil
from pathlib import Path

# Get Gull device ids
query="""
select device_info_serial -- *, start_date, end_date 
from gps.ee_track_session_limited
where key_name = 'CP_OMAN'
order by device_info_serial
"""
database_url = "postgresql://username:password@pub.e-ecology.nl:5432/eecology"
results = query_database(database_url, query)
db_cp_ids = [i[0] for i in results]

ssl_path = Path("/home/fatemeh/Downloads/bird/data/ssl/final")
all_ids = []
for p in ssl_path.glob("*.csv"):
    all_ids.append(int(p.stem))
gull_ids = all_ids.difference(db_cp_ids) # 32 gulls
cp_ids = all_ids.intersection(db_cp_ids) # 31 cp
    
# Copy gull files to a new directory
gull_path = ssl_path / "gull"
gull_path.mkdir(parents=True, exist_ok=True)
for id in gull_ids:
    p = ssl_path / f"{id}.csv"
    shutil.copy(p, gull_path)

# Copy cp files to a new directory
cp_path = ssl_path / "cp"
cp_path.mkdir(parents=True, exist_ok=True)
for id in cp_ids:
    p = ssl_path / f"{id}.csv"
    shutil.copy(p, cp_path)

# Get some statistics
# ====================
# Count number of items all files # 42,978,660
items = 0
for p in ssl_path.glob("*.csv"):
    items += sum(1 for _ in p.open())

# Count number of items gull files # 19,830,960
items = 0
for id in gull_ids:
    p = ssl_path / f"{id}.csv"
    items += sum(1 for _ in p.open())

# Count number of items co files # 23,147,700
items = 0
for id in cp_ids:
    p = ssl_path / f"{id}.csv"
    items += sum(1 for _ in p.open())
'''
