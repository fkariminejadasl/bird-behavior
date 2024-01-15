import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Union

import matplotlib.pylab as plt
import numpy as np
import psycopg2
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
[657,  45, 505, 178, 563, 808, 305,  28, 143, 155] # total new 
[657,  45, 505, 178, 563, 817, 345,  28, 143, 184] # total new with 0 confidence

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

9=Pecking (in new data, pecking replaced by):
    10=Looking_food
    11=Handling_mussel
    13=StandForage

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
    def __init__(self, all_measurements: np.ndarray, ldts: np.ndarray, transform=None):
        """
        dtype: all_measurements np.float32
        dtype: ldts np.int64
        """
        self.ldts = np.ascontiguousarray(ldts)
        self.data = all_measurements.copy()
        # normalize gps speed
        # self.data[:, :, 3] = self.data[:, :, 3] / self.data[:, :, 3].max()
        self.data[:, :, 3] = (
            self.data[:, :, 3] / 22.3012351755624
        )  # all_measurements[..., 3].max()
        self.data = self.data.astype(np.float32)

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ind):
        data = self.data[ind].transpose((1, 0))  # LxC -> CxL
        ldt = self.ldts[ind]

        if self.transform:
            data = self.transform(data)

        return data, ldt


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
    ids, dinds, gps_imus, labels, confs = read_csv_bird_file(data_file)
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

    # remove zero conf and zero labels
    lcs = np.stack((labels, confs)).T
    inds = np.where((lcs[:, 0] != 0) & (lcs[:, 1] == 1))[0]

    labels = [labels[ind] for ind in inds]
    dinds = [dinds[ind] for ind in inds]
    ids = [ids[ind] for ind in inds]

    # # try to identify locations where indices are not divided by 20
    # lds = np.stack((labels, dinds)).T
    # inds = np.where(np.diff(dinds)!=1)[0] + 1

    num_labels = len(labels)
    num_data_points = num_labels / 20
    hist = list(np.histogram(labels, bins=range(1, 12))[0])
    print(data_file.stem)
    print(
        f"total: {total_data:6d}, # labels: {num_labels:6d}, # data: {num_data_points}"
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
    hist, num_data_points = bd.stats_per_csv_file(data_file, plot=False)
    count += num_data_points
    hists.append(hist)

hists = np.array(hists, dtype=np.int64)
print(hists)
print(dict(zip(range(1, 11), np.sum(hists, axis=0))))
print(count)

count = 0
for data_file in files:
    ids, inds, gps_imus, labels, confs = bd.read_csv_bird_file(data_file)
    for l, c, i in zip(labels, confs, inds):
        if (c == 0) & (l!=0):
            print(data_file.stem, l, c, i)
            count += 1
print(count) # 78=1560/20

# from datetime import datetime
# >>> datetime.strptime('2023-11-06 14:08:11.915636', "%Y-%m-%d %H:%M:%S.%f").timestamp()
# >>> datetime.strptime('2023-11-06 13:08:11.915636', "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
# 1699276091.915636
# >>> datetime.utcfromtimestamp(1699276091.915636).strftime("%Y-%m-%d %H:%M:%S.%f")
# '2023-11-06 13:08:11.915636'
"""


def combine_jsons_to_one_json(json_files):
    # input list[Path]
    combined_data = []
    for file_name in json_files:
        with open(file_name, "r") as file:
            data = json.load(file)
            combined_data.extend(data)
    with open(data_path / "combined.json", "w") as combined_file:
        json.dump(combined_data, combined_file)


def combine_all_data(data_file):
    labels, label_ids, device_ids, time_stamps, all_measurements = read_data(data_file)
    label_device_times = np.stack((label_ids, device_ids, time_stamps)).T

    inds = np.arange(all_measurements.shape[0])
    np.random.shuffle(inds)
    all_measurements = all_measurements[inds]
    label_device_times = np.array(label_device_times)[inds]
    return all_measurements.astype(np.float64), label_device_times.astype(np.int64)


def get_specific_labesl(all_measurements, ldts, target_labels):
    """
    e.g. target_labels=[0, 2, 4, 5]
    """
    agps_imus = np.empty(shape=(0, 20, 4))
    new_ldts = np.empty(shape=(0, 3), dtype=np.int64)
    for i in target_labels:
        agps_imus = np.concatenate(
            (agps_imus, all_measurements[ldts[:, 0] == i]), axis=0
        )
        new_ldts = np.concatenate((new_ldts, ldts[ldts[:, 0] == i]), axis=0)

    inds = np.arange(new_ldts.shape[0])
    np.random.shuffle(inds)
    agps_imus = agps_imus[inds]
    new_ldts = new_ldts[inds]
    new_ldts = reindex_ids(new_ldts)
    return agps_imus, new_ldts


def combine_specific_labesl(ldts, target_labels):
    """
    e.g. target_labels=[0, 2, 4, 5]
    """
    new_id = target_labels[0]
    new_ldts = ldts.copy()
    for i in target_labels:
        new_ldts[ldts[:, 0] == i] = new_id
    return new_ldts


def reindex_ids(ldts):
    unique_ids = set(ldts[:, 0])
    new_ldts = ldts.copy()
    for i, unique_id in enumerate(unique_ids):
        new_ldts[ldts[:, 0] == unique_id, 0] = i
    return new_ldts


def raw2meas(x_m, y_m, z_m, *args):
    """
    for raw imu to measurement imu
    """
    x_o, x_s, y_o, y_s, z_o, z_s = args
    x_a = (x_m - x_o) / x_s
    y_a = (y_m - y_o) / y_s
    z_a = (z_m - z_o) / z_s
    return x_a, y_a, z_a


def query_database(database_url, sql_query):
    """
    format of database url:
    database_url = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
    """
    # connection = psycopg2.connect(dbname=database_name, user=username, password=password, host=host, port=port)
    connection = psycopg2.connect(database_url)
    cursor = connection.cursor()
    cursor.execute(sql_query)

    # Fetch all the rows
    result = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    connection.close()
    return result


def get_data(database_url, device_id, start_time, end_time):
    """
    Retrieve sensor data from a specified database within a given time range.

    Parameters
    ----------
    database_url : str
        The URL of the database to query.
        Example: "postgresql://username:password@host:port/database_name".
    device_id : int
        The unique identifier of the device whose data is being queried.
    start_time : str
        The start of the time range for data retrieval, formatted as '%Y-%m-%d %H:%M:%S'.
        Example: '2012-05-27 03:50:46'.
    end_time : str
        The end of the time range for data retrieval, formatted as '%Y-%m-%d %H:%M:%S'.
        Note: `start_time` and `end_time` can be the same.

    Returns
    -------
    tuple of np.ndarray
        The first np.ndarray is a 2D array containing IMU data (x, y, z) and GPS 2D speed.
        The second np.ndarray consists of indices, device IDs, and timestamps.


    Examples
    --------
    >>> database_url = "postgresql://username:password@host:port/database_name"
    >>> device_id = 541
    >>> start_time = '2012-05-17 00:00:59'
    >>> gimus, idts, llat = get_data(database_url, device_id, start_time, start_time)
    >>> gimus[40]
    array([0.07432701, -0.13902547,  0.96671783,  1.26196257])
    >>> idts[40]
    array([40, 541, 1337212859])
    >>> llat[40]
    [52.6001054, 4.3212097, -1, 30.5]

    Notes
    -----
    The function queries a database to retrieve calibration IMU values, 2D GPS speed,
    and IMU data for a specific device within a given time range. It processes this
    data and returns it in two structured numpy array format.
    """

    # Get calibration imu values from database
    sql_query = f"""
    select *
    from gps.ee_tracker_limited
    where device_info_serial = {device_id}
    """
    results = query_database(database_url, sql_query)
    assert len(results) != 0, "no data found"
    _, x_o, x_s, y_o, y_s, z_o, z_s = [
        float(cell) for cell in results[0] if isinstance(cell, Decimal)
    ]

    # speed_2d for gpd speed
    sql_query = f"""
    SELECT *
    FROM gps.ee_tracking_speed_limited
    WHERE device_info_serial = {device_id} and date_time between '{start_time}' and '{end_time}'
    order by date_time
    """
    results = query_database(database_url, sql_query)
    assert len(results) != 0, "no data found"
    times_gps_infos = [
        [
            int(result[1].replace(tzinfo=timezone.utc).timestamp()),
            result[-4],
            result[2],
            result[3],
            result[4],
            result[6],
        ]
        for result in results
    ]

    # get imu
    sql_query = f"""
    SELECT *
    FROM gps.ee_acceleration_limited
    WHERE device_info_serial = {device_id} and date_time between '{start_time}' and '{end_time}'
    order by date_time, index
    """
    results = query_database(database_url, sql_query)
    assert len(results) != 0, "no data found"

    # filter data: remove imu data, which has nanes
    results = [result for result in results if not is_none(*result[-3:])]

    # get data groups
    indices = [result[2] - 1 for result in results]  # make indices zero-based
    timestamps = [
        int(result[1].replace(tzinfo=timezone.utc).timestamp()) for result in results
    ]
    imus = [
        np.round(raw2meas(*result[-3:], x_o, x_s, y_o, y_s, z_o, z_s), 8)
        for result in results
    ]
    # data element: index, time, imu
    data = [[i, t, *imu] for i, t, imu in zip(indices, timestamps, imus)]
    groups = identify_and_process_groups(data)

    # match gps data: time, GPS 2d speed, latitude, longitude, altitude, temperature
    for group in groups:
        timestamps = set([i[1] for i in group])
        assert len(timestamps) == 1
        timestamp = timestamps.pop()
        gps = [gt[1:] for gt in times_gps_infos if gt[0] == timestamp][0]
        for item in group:
            item.extend(gps)

    # prepare final data
    igs = []  # element: imu, gps
    idts = []  # element: index, device_id, timestamp
    llat = []  # element: latitude, longitude, altitude, temperature
    for group in groups:
        for item in group:
            igs.append(item[2:6])
            index, timestamp = item[0], item[1]
            idts.append([index, device_id, timestamp])
            llat.append(item[6:])

    # igs, idts 2D array: Nx20 x 4, Nx20 x 3, llat: list Nx20 x 4
    return np.array(igs), np.array(idts, dtype=np.int64), llat


def is_none(x, y, z):
    if x == None or y == None or z == None:
        return True
    return False


'''
# example queries
# Get calibration imu values from database
sql_query = f"""
select *
from gps.ee_tracker_limited
where device_info_serial = {device_id}
"""

# speed_2d for gpd speed
sql_query = f"""
SELECT *
FROM gps.ee_tracking_speed_limited
WHERE device_info_serial = {device_id} and date_time between '{start_time}' and '{end_time}'
order by date_time
"""

# get imu
sql_query = f"""
SELECT *
FROM gps.ee_acceleration_limited
WHERE device_info_serial = {device_id} and date_time between '{start_time}' and '{end_time}'
order by date_time, index
"""
results = query_database(database_url, sql_query)
'''


def identify_and_process_groups(data):
    """
    Identify, filter, and process groups of items with consecutive indices in a list.

    This function processes a list of items, where each item is a list containing an index and an
    additional value (e.g., [[1, 'a'], [2, 'b'], ...]). It identifies groups of items with
    consecutive indices. The function filters out groups that are shorter than 20 elements. For
    the remaining groups, it splits them into subgroups of exactly 20 elements each. Any remaining
    items in a group after forming these subgroups are discarded.
    1. Identify groups.
    2. Remove groups that are shorter than 20 elements in length.
    3. Return only the groups of indices that have exactly 20 elements. For example, if we have
       a group like `1, 2, ..., 46`, it should be divided into two groups. The first one would be
       `1, 2, ..., 20` and the second would be `21, 22, ..., 40`. The remaining indices,
       `41, 42, ..., 46`, are discarded.

    Parameters
    ----------
    data : list of list
        A list of items, where each item is a list containing an index (int) and an additional value.
        The indices are expected to be in a sorted and potentially grouped sequential order.

    Returns
    -------
    list of list
        A list containing subgroups of the input items. Each subgroup is a list of exactly
        20 items from the original list, based on consecutive indices, and only subgroups that
        could be fully formed (i.e., with exactly 20 elements) are included.

    Examples
    --------
    >>> data = [[1, 'a'], [2, 'b'], ..., [46, 'x'], [1, 'y'], ..., [60, 'aa']]
    >>> identify_and_process_groups(data)
    [[[1, 'a'], [2, 'b'], ..., [20, 't']], [[21, 'u'], [22, 'v'], ..., [40, 'dd']]]

    Notes
    -----
    The function assumes that the input list 'data' contains items in the format [index, value],
    where 'index' is an integer. Groups are identified based on consecutive index sequences in this list.
    Repeated indices are handled by associating each index with its original position in the 'data' list.
    """

    # Extract indices
    indices = [item[0] for item in data]

    # Original logic to identify and process groups
    groups = []
    current_group = [(indices[0], 0)]  # Store index along with its position

    for i in range(1, len(indices)):
        if indices[i] == current_group[-1][0] + 1:
            current_group.append((indices[i], i))
        else:
            groups.append(current_group)
            current_group = [(indices[i], i)]

    # Add the last group
    groups.append(current_group)

    # Filter groups less than length 20
    filtered_groups = [group for group in groups if len(group) >= 20]

    # Map processed groups back to original items
    final_groups = []
    for group in filtered_groups:
        for i in range(0, len(group), 20):
            subgroup_tuples = group[i : i + 20]
            if len(subgroup_tuples) == 20:
                # Retrieve the original items using global index
                subgroup = [data[t[1]] for t in subgroup_tuples]
                final_groups.append(subgroup)

    return final_groups


def test_identify_and_process_groups():
    # fmt: off
    data = [[1, 20], [2, 14], [1, 50], [2, 34], [3, 28], [4, 22], [5, 18], [6, 15], [7, 14], [8, 13], [9, 12], [10, 11], [11, 10], [12, 9], [13, 8], [14, 7], [15, 6], [16, 5], [17, 4], [18, 3], [19, 2], [20, 1], [21, 50], [22, 49], [23, 48], [24, 47], [25, 46], [26, 45], [27, 44], [28, 43], [29, 42], [30, 41], [31, 40], [32, 39], [33, 38], [34, 37], [35, 36], [36, 35], [37, 34], [38, 33], [39, 32], [40, 31], [41, 30], [42, 29], [43, 28], [44, 27], [45, 26], [46, 25]]
    # fmt: on
    processed_groups = identify_and_process_groups(data)
    np.testing.assert_equal(np.array(data)[2:22], np.array(processed_groups[0]))
    np.testing.assert_equal(np.array(data)[22:42], np.array(processed_groups[1]))


test_identify_and_process_groups()
print("passed")
"""
# plot data 
data_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data")
train_file = data_path / "train_set.json"
valid_file = data_path / "validation_set.json"
test_file = data_path / "test_set.json"
combined_file = data_path / "combined.json"
# combine_jsons_to_one_json([train_file, valid_file, test_file])
# labels, label_ids, device_ids, time_stamps, all_measurements = read_data(combined_file)


all_measurements, ldts = combine_all_data(combined_file)
alabel_ids = ldts[:, 0]
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

from datetime import datetime, timezone

from scipy.io import loadmat

data_path = Path("/home/fatemeh/Downloads/bird/data_from_Susanne")

# dd = loadmat(data_path / "AnnAcc1600_20140520_152300-20140520_153000.mat")['outputStruct']
# dd['accX'][0][0][82][0][0] # accY, accZ contains nan and variale length 12-40
# dd['annotations'][0][0] # ind=0 for data, ind=1-2 start-end of indices in tags, ind=3 for label
# dd['tags'][0][0][0][82]
# dd['gpsSpd'][0][0]
# np.isnan(dd['accZ'][0][0][82][0][0][-1]) # True


def count_id(dd, id_):
    count = 0
    for t in dd["tags"][0][0][0]:
        id_count = sum(t[t[:, 0] == id_][:, 1])
        count += id_count
    return count


def get_start_end_inds_one_tag(i, tag, id_):
    inds = np.where(tag[:, 0] == id_)[0]
    start_ends = []
    if inds.size != 0:
        change_id_inds = np.where(np.diff(inds) != 1)[0] + 1
        change_inds = list(inds[change_id_inds])
        change_inds_before = list(inds[change_id_inds - 1] + 1)
        start_ind, end_ind = inds[0], inds[-1] + 1
        start_ends = [
            (i, start, end, end - start)
            for start, end in zip(
                [start_ind] + change_inds, change_inds_before + [end_ind]
            )
        ]
    return start_ends


def get_start_end_inds(dd, id_):
    all_start_ends = []
    for i, tag in enumerate(dd["tags"][0][0][0]):
        start_ends = get_start_end_inds_one_tag(i, tag, id_)
        if start_ends:
            all_start_ends += start_ends
            print(start_ends)
    return all_start_ends


"""
# {1: 968, 6: 606, 7: 17, 11: 60, 12: 36, 14: 8, 15: 1, 16: 10, 17: 1, 18: 171}
id_count = dict()
for id_ in [12]:  # range(1, 19):
    counts = 0
    for data_file in data_path.glob("*.mat"):
        print(data_file.stem)
        dd = loadmat(data_file)["outputStruct"]
        all_start_ends = get_start_end_inds(id_)
        for i, s, e, _ in all_start_ends:
            print(dd["sampleID"][0][0][0, i])
        counts += sum([c // 20 for i, s, e, c in all_start_ends])
    if counts != 0:
        id_count[id_] = counts


year, month, day = (
    dd["year"][0][0][0, 30],
    dd["month"][0][0][0, 30],
    dd["day"][0][0][0, 30],
)
hour, min, sec = dd["hour"][0][0][0, 30], dd["min"][0][0][0, 30], dd["sec"][0][0][30, 0]
timestamp = (
    datetime(year, month, day, hour, min, sec).replace(tzinfo=timezone.utc).timestamp()
)
tags = dd["tags"][0][0][0][30][49:76]
device_id = dd["sampleID"][0][0][0, 30]  # ?
imu_x = dd["accX"][0][0][30][0][0][49:76]
imu_y = dd["accY"][0][0][30][0][0][49:76]
imu_z = dd["accZ"][0][0][30][0][0][49:76]
gps_single = dd["gpsSpd"][0][0][30, 0]
assert len(set(tags[:, 0])) == 1
assert set(tags[:, 1]) == {1}
assert not any(np.isnan(imu_x))
assert not any(np.isnan(imu_y))
assert not any(np.isnan(imu_z))
assert not np.isnan(gps_single)
label = tags[0, 0]
gps = np.repeat(gps_single, len(imu_x))
data = np.stack((imu_x, imu_y, imu_z, gps)).T
data = data[None, ...]  # 1 x N x 4

print(id_count)
"""

"""
def test(model, n=10):
    a = all_measurements[0:n].copy()
    a[..., 3] = a[..., 3] / 22.3012351755624
    a = torch.tensor(a.astype(np.float32)).transpose(2,1)
    out = model(a)
    print(torch.argmax(out, axis=1).tolist())
    print(label_ids[0:n])

b1 = torch.concat((torch.normal(0, .01, size=(20,1)), torch.normal(0, .01, size=(20,1)), torch.normal(0, .1, size=(20,1)), torch.rand(20,1)*.001), axis=1).T.unsqueeze(0) + a1
a2 = torch.concat((a[0:1], a[6:7]), axis=2)
m = np.concatenate((all_measurements[0], all_measurements[5]), axis=0)
_, axs = plt.subplots(3, 1, sharex=True);axs[1].plot(m[:, 0], "r-*", m[ :, 1], "b-*", m[:, 2], "g-*");axs[2].plot(m[:, 3]);plt.show(block=False)
"""
