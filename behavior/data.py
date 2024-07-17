import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Union

import matplotlib.pylab as plt
import numpy as np
import psycopg2
from torch.utils.data import Dataset

# np.random.seed(0)
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
[datetime.fromtimestamp(time/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S') for time in time_stamps]
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
        if (
            "measurements" not in item["gpsRecord"]
            or not item["gpsRecord"]["measurements"]
        ):
            measurements = []
        else:
            measurements = get_per_location_measurements(
                item["gpsRecord"]["measurements"]
            )
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
        self.ldts = np.ascontiguousarray(ldts)  # Nx3
        self.data = all_measurements.copy()  # NxLxC C=4
        # normalize gps speed by max
        self.data[:, :, 3] = self.data[:, :, 3] / 22.3012351755624
        self.data = self.data.astype(np.float32)

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ind):
        data = self.data[ind].transpose((1, 0))  # LxC -> CxL
        ldt = self.ldts[ind]  # 3

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


def combine_jsons_to_one_json(json_files, save_file):
    # input list[Path]
    combined_data = []
    for file_name in json_files:
        with open(file_name, "r") as file:
            data = json.load(file)
            combined_data.extend(data)
    with open(save_file, "w") as combined_file:
        json.dump(combined_data, combined_file)


def combine_all_data(data_file):
    """
    Parameters
    ----------
    data_file: Path
        json file

    Returns
    -------
    tuple of np.ndarray
        first: N x 20 x 4, float64
        second: N x 3, int64
    """
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


def query_database(database_url, sql_query):
    '''
    Execute a SQL query and return the results.

    format of database url:
    database_url = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"

    # example queries
    device_id = 805
    start_time = '2015-05-27 09:19:34'
    end_time = '2015-05-27 09:20:34'

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

    >>> results = query_database(database_url, sql_query)
    '''
    # connection = psycopg2.connect(dbname=database_name, user=username, password=password, host=host, port=port)
    connection = psycopg2.connect(database_url)
    cursor = connection.cursor()
    cursor.execute(sql_query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result


def fetch_calibration_data(database_url, device_id):
    """
    Fetch calibration IMU values from the database.
    """
    sql_query = f"""
    SELECT *
    FROM gps.ee_tracker_limited
    WHERE device_info_serial = {device_id}
    """
    results = query_database(database_url, sql_query)
    if len(results) == 0:
        raise ValueError("No calibration data found")
    return [float(cell) for cell in results[0][5:11]]


def fetch_gps_data(database_url, device_id, start_time, end_time):
    """
    Fetch GPS data from the database.
    """
    sql_query = f"""
    SELECT *
    FROM gps.ee_tracking_speed_limited
    WHERE device_info_serial = {device_id} AND date_time BETWEEN '{start_time}' AND '{end_time}'
    ORDER BY date_time
    """
    results = query_database(database_url, sql_query)
    if len(results) == 0:
        raise ValueError("No GPS data found")

    return [
        [
            int(result[1].replace(tzinfo=timezone.utc).timestamp()),
            result[-4],
            result[2],
            result[3],
            result[4],
            result[6],
        ]
        for result in results
        if result[-4] is not None
    ]


def fetch_imu_data(database_url, device_id, start_time, end_time):
    """
    Fetch IMU data from the database.
    """
    sql_query = f"""
    SELECT *
    FROM gps.ee_acceleration_limited
    WHERE device_info_serial = {device_id} AND date_time BETWEEN '{start_time}' AND '{end_time}'
    ORDER BY date_time, index
    """
    results = query_database(database_url, sql_query)
    if len(results) == 0:
        raise ValueError("No IMU data found")

    return [result for result in results if not is_none(*result[-3:])]


def raw2meas(x_m, y_m, z_m, x_o, x_s, y_o, y_s, z_o, z_s):
    """
    Convert raw IMU measurements (x_m, y_m, z_m) to calibrated values using calibration
    values (x_o, x_s, y_o, y_s, z_o, z_s). All values are floats.
    """
    x_a = (x_m - x_o) / x_s
    y_a = (y_m - y_o) / y_s
    z_a = (z_m - z_o) / z_s
    return x_a, y_a, z_a


def is_none(x, y, z):
    """
    Check if any of the values are None.
    """
    return x is None or y is None or z is None


def identify_and_process_groups(data, glen=20):
    """
    Identify and process groups of items with consecutive indices.

    Parameters
    ----------
    data : list of list
        A list of items, where each item is a list containing an index and additional values.
        The indices are expected to be in a sorted and potentially grouped sequential order.
    glen : int
        Group length. Default is 20.

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
    """
    indices = [item[0] for item in data]
    groups = []
    current_group = [(indices[0], 0)]  # Store index along with its position

    for i in range(1, len(indices)):
        if indices[i] == current_group[-1][0] + 1:
            current_group.append((indices[i], i))
        else:
            groups.append(current_group)
            current_group = [(indices[i], i)]

    groups.append(current_group)
    filtered_groups = [group for group in groups if len(group) >= glen]

    final_groups = []
    for group in filtered_groups:
        for i in range(0, len(group), glen):
            subgroup_tuples = group[i : i + glen]
            if len(subgroup_tuples) == glen:
                subgroup = [data[t[1]] for t in subgroup_tuples]
                final_groups.append(subgroup)

    return final_groups


def match_gps_to_groups(groups, times_gps_infos):
    """
    Match GPS data to IMU groups and filter out unmatched groups.

    Parameters
    ----------
    groups : list of list
        The IMU data groups.
    times_gps_infos : list of list
        The GPS data including timestamp, GPS speed, latitude, longitude, altitude, and temperature.

    Returns
    -------
    list of list
        The matched IMU data groups with GPS data.

    Raises
    ------
    ValueError
        If a group has different timestamps.
    """
    # Filter out groups without corresponding gps information
    filtered_groups = []
    for group in groups:
        timestamps = {i[1] for i in group}
        if len(timestamps) != 1:
            raise ValueError("Different timestamps for a group")
        timestamp = timestamps.pop()
        gps = [gt[1:] for gt in times_gps_infos if gt[0] == timestamp]
        if gps:
            filtered_groups.append((group, gps[0]))

    # Extend the items in the remaining groups
    for group, gps in filtered_groups:
        for item in group:
            item.extend(gps)

    # Replace original groups with the filtered and processed ones
    return [group for group, _ in filtered_groups]


def process_data(groups, device_id):
    """
    Process the data groups into the final format.

    Parameters
    ----------
    groups : list of list
        The matched IMU and GPS data groups.
    device_id : int
        The unique identifier of the device.

    Returns
    -------
    tuple of np.ndarray
        The first np.ndarray is a 2D array containing IMU data (x, y, z) and GPS 2D speed.
        The second np.ndarray consists of indices, device IDs, and timestamps.
        The third np.ndarray consists of latitude, longitude, altitude, temperature.
    """
    igs = []  # IMU and GPS
    idts = []  # Index, device_id, timestamp
    llat = []  # Latitude, longitude, altitude, temperature

    for group in groups:
        for item in group:
            igs.append(item[2:6])
            index, timestamp = item[0], item[1]
            idts.append([index, device_id, timestamp])
            llat.append(item[6:])

    return np.array(igs), np.array(idts, dtype=np.int64), llat


def get_data(database_url, device_id, start_time, end_time, glen=20):
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
    glen : int
        Group length. Default is 20.

    Returns
    -------
    tuple of np.ndarray
        The first np.ndarray is a 2D array containing IMU data (x, y, z) and GPS 2D speed.
        The second np.ndarray consists of indices, device IDs, and timestamps.
        The third np.ndarray consists of latitude, longitude, altitude, temperature.
        igs, idts 2D array: Nx20 x 4, Nx20 x 3, llat: list Nx20 x 4

    Examples
    --------
    >>> database_url = "postgresql://username:password@host:port/database_name"
    >>> database_url = "postgresql://username:password@pub.e-ecology.nl:5432/eecology"
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
    calibration_values = fetch_calibration_data(database_url, device_id)
    gps_data = fetch_gps_data(database_url, device_id, start_time, end_time)
    imu_data = fetch_imu_data(database_url, device_id, start_time, end_time)

    indices = [result[2] - 1 for result in imu_data]
    timestamps = [
        int(result[1].replace(tzinfo=timezone.utc).timestamp()) for result in imu_data
    ]
    imus = [
        np.round(raw2meas(*result[-3:], *calibration_values), 8) for result in imu_data
    ]
    data = [[i, t, *imu] for i, t, imu in zip(indices, timestamps, imus)]
    groups = identify_and_process_groups(data, glen)

    matched_groups = match_gps_to_groups(groups, gps_data)
    if len(matched_groups) == 0:
        raise ValueError("No matching IMU and GPS data found")

    return process_data(matched_groups, device_id)


def test_identify_and_process_groups():
    # fmt: off
    data = [
        [1, 20], [2, 14], [1, 50], [2, 34], [3, 28], [4, 22], [5, 18], [6, 15], [7, 14], [8, 13],
        [9, 12], [10, 11], [11, 10], [12, 9], [13, 8], [14, 7], [15, 6], [16, 5], [17, 4], [18, 3],
        [19, 2], [20, 1], [21, 50], [22, 49], [23, 48], [24, 47], [25, 46], [26, 45], [27, 44],
        [28, 43], [29, 42], [30, 41], [31, 40], [32, 39], [33, 38], [34, 37], [35, 36], [36, 35],
        [37, 34], [38, 33], [39, 32], [40, 31], [41, 30], [42, 29], [43, 28], [44, 27], [45, 26], [46, 25]
    ]
    # fmt: on
    processed_groups = identify_and_process_groups(data)
    np.testing.assert_equal(np.array(data)[2:22], np.array(processed_groups[0]))
    np.testing.assert_equal(np.array(data)[22:42], np.array(processed_groups[1]))


# Test the function
test_identify_and_process_groups()


def random_time_between(start_time_str, end_time_str, time_format="%Y-%m-%d %H:%M:%S"):
    """
    Generate a random time between two given times.

    Parameters:
    start_time_str (str): The start time as a string.
    end_time_str (str): The end time as a string.
    time_format (str): The format of the time strings.

    Returns:
    datetime: A random datetime between the start and end times.

    Example:
    >>> start = '2012-05-17 00:00:59'
    >>> end = '2012-05-18 00:00:59'
    """
    # Convert the time strings to datetime objects
    start_time = datetime.strptime(start_time_str, time_format)
    end_time = datetime.strptime(end_time_str, time_format)

    # Calculate the difference between the two times
    time_diff = end_time - start_time

    # Generate a random number of seconds between 0 and the total difference in seconds
    random_seconds = np.random.randint(0, int(time_diff.total_seconds()))

    # Add the random number of seconds to the start time to get a random time
    random_time = start_time + timedelta(seconds=random_seconds)

    return random_time


def generate_random_time_intervals(
    start_time_str,
    end_time_str,
    n_times,
    interval_minutes=15,
    time_format="%Y-%m-%d %H:%M:%S",
):
    """
    Generate multiple random times intervals between two given times.

    Parameters:
    start_time_str (str): The start time as a string.
    end_time_str (str): The end time as a string.
    n_times (int): The number of random times to generate.
    interval_minutes (int): The interval in minutes for the next time after each random time.
    time_format (str): The format of the time strings.

    Returns:
    list of tuples: Each tuple contains a random datetime and the datetime 15 minutes later.

    Example usage:
    start = '2012-01-01 00:00:00'
    end = '2013-01-01 00:00:00'
    n_times = 10  # Generate 10 random times
    random_times = generate_random_time_intervals(start, end, n_times)

    for random_time, interval_time in random_times:
        print(f"Random time: {random_time}, 15 minutes later: {interval_time}")
    """
    # Convert the time strings to datetime objects
    start_time = datetime.strptime(start_time_str, time_format)
    end_time = datetime.strptime(end_time_str, time_format)

    # Calculate the difference between the two times
    time_diff = end_time - start_time

    # Generate n random times in seconds between the start and end times
    random_seconds = np.random.randint(0, int(time_diff.total_seconds()), n_times)

    random_times = [
        start_time + timedelta(seconds=int(seconds)) for seconds in random_seconds
    ]
    interval = timedelta(minutes=interval_minutes)

    # Create list of tuples with random time and the time 15 minutes later
    random_times_with_intervals = [(time, time + interval) for time in random_times]

    return random_times_with_intervals


def append_to_csv(save_file, gimus, idts):
    items = []
    timestamp = idts[0, 2]
    label = -1  # no label
    device_id = idts[0, 1]
    ftime = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    for idt, gimu in zip(idts, gimus):
        index = idt[0]
        item = (
            f"{device_id},{ftime},{index},{label},{gimu[0]:.8f},{gimu[1]:.8f},"
            f"{gimu[2]:.8f},{gimu[3]:.8f}\n"
        )
        items.append(item)

    with open(save_file, "a") as rfile:
        for item in items:
            rfile.write(item)


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
- indices are sometimes starts with zero, sometimes 1: correct for indices
- 4. save device, time, index, label, imu, gps in files (60_000 limits)
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
        if idts[0, 0] == -1: # indices can start from zero. I read them as zero-based.
            idts[:,0] += 1
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
import concurrent.futures
import threading
import multiprocessing
from functools import partial

# Fatemeh: 

def process_dates(dates, database_url, device_id, label, output_file):
    file = open(output_file, 'w')
    for date in dates:
        try:
            gimus, idts, _ = get_data(database_url, device_id, date, date, glen=60)
            print(f"{date}, {idts[0,0]},{idts.shape[0]}")
            if idts[0, 0] == -1: # indices can start from zero. I read them as zero-based.
                idts[:,0] += 1
            for gimu, idt in zip(gimus, idts):
                line = f"{device_id},{date},{int(idt[0])},{label},{gimu[0]:.8f},{gimu[1]:.8f},{gimu[2]:.8f},{gimu[3]:.8f}\n"
                file.write(line)
            file.flush()
        except Exception as e:
            print(f"Error processing device {device_id}, date {date}: {e}")
            continue
    file.close()

def wrapper_process_dates(dates_outputfile):
    return process_dates(dates_outputfile[0], database_url, device_id, label, dates_outputfile[1])

database_url = ""# f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
label = -1
device_id = 658# 298
p = Path(f"/home/fatemeh/Downloads/bird/gpsdates/{device_id}.csv")
dates = read_dates(p)
dates = get_random_entries(dates, 20)

dates_outputfile = []
for i in range(2):
    sel_dates = dates[i*10:i*10+10]
    output_file = Path(f"/home/fatemeh/Downloads/bird/ssl/{device_id}_{i}.csv")
    dates_outputfile.append([sel_dates, output_file])

with multiprocessing.Pool(processes=2) as pool:
        results = pool.map(wrapper_process_dates, dates_outputfile)

print("done")



'''
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


def load_csv(csv_file):
    """
    e.g. row: 757,2014-05-18 06:58:26,20,0,-0.09648467,-0.04426107,0.45049885,8.89139205

    Returns
    -------
    tuple of np.ndarray
        first: N x 20 x 4, float64
        second: N x 3, int64
    """
    igs = []
    ldts = []
    with open(csv_file, "r") as file:
        for row in file:
            items = row.strip().split(",")
            device_id = int(items[0])
            timestamp = (
                datetime.strptime(items[1], "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
            label = int(items[3])
            ig = [float(i) for i in items[4:]]
            igs.append(ig)
            ldts.append([label, device_id, timestamp])
    igs = np.array(igs).astype(np.float64).reshape(-1, 20, 4)
    ldts = np.array(ldts).astype(np.int64).reshape(-1, 20, 3)[:, 0, :]
    return igs, ldts


# TODO to check and remove

"""
# plot data 
data_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data")
train_file = data_path / "train_set.json"
valid_file = data_path / "validation_set.json"
test_file = data_path / "test_set.json"
combined_file = data_path / "combined.json"
# combine_jsons_to_one_json([train_file, valid_file, test_file], combined_file)
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

i = 30
year, month, day = (
    dd["year"][0][0][0, i],
    dd["month"][0][0][0, i],
    dd["day"][0][0][0, i],
)
hour, min, sec = dd["hour"][0][0][0, i], dd["min"][0][0][0, i], dd["sec"][0][0][i, 0]
timestamp = (
    datetime(year, month, day, hour, min, sec).replace(tzinfo=timezone.utc).timestamp()
)
tags = dd["tags"][0][0][0][i][49:76]
device_id = dd["sampleID"][0][0][0, i]  # ?
imu_x = dd["accX"][0][0][i][0][0][49:76]
imu_y = dd["accY"][0][0][i][0][0][49:76]
imu_z = dd["accZ"][0][0][i][0][0][49:76]
gps_single = dd["gpsSpd"][0][0][i, 0]
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
"""
import csv
import os


def split_csv(input_file, output_dir, lines_per_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, "r") as file:
        count = 0
        file_count = 1
        output_file = open(os.path.join(output_dir, f"part_{file_count}.csv"), "w")

        for line in file:
            items = line.split(",")
            label = int(items[3])
            if label not in [0, 1, 2, 3, 4, 5, 6, 8, 9]:
                continue
            if label in [8, 9]:
                items[3] = str(label - 1)
            line = ",".join(items[3:])  # only for label and data

            if count < lines_per_file:
                output_file.write(line)
                count += 1
                # if label in [0, 1, 2, 3, 4, 5, 6, 8, 9]:
                #     if label in [8, 9]:
                #         items[3] = str(label-1)
                #         line = ','.join(items)
                # output_file.write(line)
                # count += 1
            else:
                output_file.close()
                count = 0
                file_count += 1
                output_file = open(
                    os.path.join(output_dir, f"part_{file_count}.csv"), "w"
                )
                output_file.write(line)
                count = 1
                # if label in [0, 1, 2, 3, 4, 5, 6, 8, 9]:
                #     if label in [8, 9]:
                #         items[3] = str(label-1)
                #         line = ','.join(items)
                # output_file.write(line)
                # count = 1

        output_file.close()


def compute_group_counts(file_paths, group_size=20):
    group_counts = {}
    for file_path in file_paths:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            row_count = sum(1 for row in reader)
            group_count = row_count // group_size
            group_counts[str(file_path)] = group_count
    return group_counts


# split_csv('/home/fatemeh/Downloads/bird/tmp2/combined_s_w_m_j.csv', '/home/fatemeh/Downloads/bird/tmp4', 20)

# # List of CSV file paths
# csv_files = Path("/home/fatemeh/Downloads/bird/tmp2").glob("part*")
# csv_files = sorted(csv_files, key=lambda x: int(x.stem.split('_')[1]))
# group_counts = compute_group_counts(csv_files)

# # Save group_counts to a file (or use directly)
# import json
# with open('/home/fatemeh/Downloads/bird/tmp/group_counts.json', 'w') as f:
#     json.dump(group_counts, f)

# for csv_file in csv_files:
#     with open(csv_file,'r') as f:
#         for line in f:
#             if int(line.split(',')[3]) == 9:
#                 print(csv_files, line)


class BirdDataset2(Dataset):
    def __init__(self, file_paths, group_counts_file, group_size=20, transform=None):
        """
        Args:
            file_paths (list of str): List of paths to CSV files.
            group_counts_file (str): Path to the JSON file containing group counts for each file.
            group_size (int): Number of rows per group.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.group_size = group_size
        self.transform = transform

        # Load precomputed group counts
        with open(group_counts_file, "r") as f:
            self.group_counts = json.load(f)

        self.data_index = self._create_data_index()

    def _create_data_index(self):
        """Create an index of the data based on precomputed group counts."""
        data_index = []
        for file_path in self.file_paths:
            group_count = self.group_counts[file_path]
            for i in range(group_count):
                data_index.append((file_path, i))
        return data_index

    def __len__(self):
        return len(self.data_index)

    def _load_csv(self, file_path):
        igs = []
        ldts = []
        with open(file_path, "r") as file:
            for row in file:
                items = row.strip().split(",")
                # device_id = int(items[0])
                # timestamp = (
                #     datetime.strptime(items[1], "%Y-%m-%d %H:%M:%S")
                #     .replace(tzinfo=timezone.utc)
                #     .timestamp()
                # )
                label = int(items[3])
                ig = [float(i) for i in items[4:]]
                ig[-1] /= 22.3012351755624
                igs.append(ig)
                # ldts.append([label, device_id, timestamp])
                ldts.append(label)
        igs = np.array(igs).astype(np.float32)
        ldts = np.array(ldts).astype(np.int64)
        return igs, ldts

    def __getitem__(self, idx):
        file_path, group_idx = self.data_index[idx]
        start_row = group_idx * self.group_size

        measurements, ldts = self._load_csv(file_path)
        data = measurements[start_row : start_row + self.group_size]
        ldts = ldts[start_row : start_row + self.group_size][0]
        data = data.transpose((1, 0))  # LxC -> CxL

        if self.transform:
            data = self.transform(data)

        return data, ldts


class BirdDataset3(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        Args:
            file_paths (list of str): List of paths to CSV files.
            group_counts_file (str): Path to the JSON file containing group counts for each file.
            group_size (int): Number of rows per group.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def _load_csv(self, file_path):
        igs = []
        ldts = []
        with open(file_path, "r") as file:
            for row in file:
                items = row.strip().split(",")
                label = int(items[0])  # 3
                ig = [float(i) for i in items[1:]]  # 4
                ig[-1] /= 22.3012351755624
                igs.append(ig)
                ldts.append(label)
        igs = np.array(igs).astype(np.float32)
        ldts = np.array(ldts).astype(np.int64)
        return igs, ldts

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        data, ldts = self._load_csv(file_path)
        ldts = np.int64(ldts[0])

        data = data.transpose((1, 0))  # LxC -> CxL
        data = np.ascontiguousarray(data)

        if self.transform:
            data = self.transform(data)

        return data, ldts


"""
import torch
from torch.utils.data import DataLoader


all_measurements, label_ids = load_csv("/home/fatemeh/Downloads/bird/data/combined_s_w_m_j.csv")
all_measurements, label_ids = get_specific_labesl(all_measurements, label_ids, [0, 1, 2, 3, 4, 5, 6, 8, 9])
train_dataset = BirdDataset(all_measurements, label_ids)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

from torch.utils.data import random_split   
csv_files = Path("/home/fatemeh/Downloads/bird/tmp3").glob("part*")
csv_files = sorted(csv_files, key=lambda x: int(x.stem.split('_')[1]))
csv_files = [str(csv_file) for csv_file in csv_files]

# dataset = BirdDataset2(csv_files, "/home/fatemeh/Downloads/bird/tmp/group_counts.json", group_size=20)
dataset = BirdDataset3(csv_files)
d, l = dataset[0]
d, l = dataset[1]

# Calculate the sizes for training and validation datasets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Use random_split to divide the dataset
tr, val = random_split(dataset, [train_size, val_size])

train_loader2 = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=1,
    drop_last=True,
)

for m, l in train_loader2:
    print(m.shape)
val_loader2 = DataLoader(
    val,
    batch_size=len(val),
    shuffle=False,
    num_workers=1,
    drop_last=True,
)
# """
