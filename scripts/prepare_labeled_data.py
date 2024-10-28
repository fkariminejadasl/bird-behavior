from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from behavior import data as bd
from behavior import utils as bu

# j: json (set1, sus json)
# m: matlab (sus mat)
# w: csv (willem)

ind2name = {
    0: "Flap",
    1: "ExFlap",
    2: "Soar",
    3: "Boat",
    4: "Float",
    5: "SitStand",
    6: "TerLoco",
    7: "Other",
    8: "Manouvre",
    9: "Pecking",
}
name2ind = {v: k for k, v in ind2name.items()}

"""
# j_data (sus json)
ind2name = {
# 1-base,0-base,original(old)
    6, 5, 0: "Flap",
    5, 4, 1: "ExFlap",
    4, 3, 2: "Soar",
          3: "Boat",
    3, 2, 4: "Float",
    2, 1, 5: "SitStand",
    1, 0, 6: "TerLoco",
    8, 7, 7: "Other",
    7, 6, 8: "Manouvre",
          9: "Pecking",
}
# Handling_mussel: 10 (9), StandForage: 11(10)->pecking (boat and pecking doesn't exist, looking_food: 9(8))
# values from json files. subtracted from one to become zero-based. 

# set3 different schema
# Flap:2, XflapL:6, XflapS:8, Soar:3, Float:7, stand:1, sit: 5, TerLoco/walk: 4, other: 9 
"""


# def find_matching_index(array, target, step=20, tolerance=1e-5):
#     for i in range(0, len(array), step):
#         if all(np.isclose(array[i], target, atol=tolerance)):
#             return i
#     return -1

def find_matching_index(keys, query, glen=20, tol=1e-6):
    """
    find matching two rows
    """
    len_keys = len(keys)
    for i in range(0, len_keys-1):
        cond1 = all(np.isclose(keys[i], query[0], atol=tol)) 
        cond2 = all(np.isclose(keys[i+1], query[1], atol=tol))
        cond3 = i + glen <= len_keys
        if cond1 & cond2 & cond3:
            return i
    return -1

def test_find_matching_index():
    df_s = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/s_data_modified.csv", header=None)
    # dup_inds = list(df_s[df_s.duplicated(df_s)].index)
    dup_ind = 59621
    device_id, start_time = df_s.iloc[dup_ind][[0, 1]] # 871, "2014-05-20 06:53:09"
    gimus, idts, llat = bd.get_data(database_url, device_id, start_time, start_time)
    keys = gimus[:,:3]
    query = np.array(df_s.iloc[dup_ind:dup_ind+2][[4,5,6]])
    i = find_matching_index(keys, query)
    assert i == 1

def write_info(csv_file, save_file):
    """
    input:  device, time, index, label, imux, imuy, imuz, gps
    output: device, time, count, label, ind
    e.g.
    input:  757,2014-05-18 06:58:26,20,0,-0.09648467,-0.04426107,0.45049885,8.89139205
    output: 757,2014-05-18 06:58:26,20,0,40
    """

    info = defaultdict(list)
    with open(csv_file, "r") as rfile:
        for row in rfile:
            row = row.strip().split(",")
            device_id = row[0]
            t = row[1]
            label = row[3]
            key = (device_id, t)
            info[key].extend([label])
    with open(save_file, "w") as f:
        for k, v in info.items():
            count_labels = dict(Counter(v))
            counts = "_".join([str(i) for i in count_labels.values()])
            labels = "_".join([i for i in count_labels.keys()])
            item = f"{k[0]},{k[1]},{counts},{labels},{0}\n"
            f.write(item)


def write_j_info(json_file, csv_file):
    # output: device, time, count, label, ind
    igs, ldts = bd.combine_all_data(json_file)
    items = []
    for ig, ldt in zip(igs, ldts):
        t = datetime.fromtimestamp(ldt[2], tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        item = f"{ldt[1]},{t},{len(ig)},{ldt[0]},0\n"
        items.append(item)
    with open(csv_file, "w") as f:
        for item in items:
            f.write(item)

def write_j_data_orig(
    json_file, save_file, new2old_labels, ignored_labels
):
    """
    read json data and append indices

    input:  device, time, index, label, imux, imuy, imuz, gps
    e.g. row: 757,2014-05-18 06:58:26,20,0,-0.09648467,-0.04426107,0.45049885,8.89139205
    """

    all_measurements, ldts = bd.combine_all_data(json_file)

    items = []
    for meas, ldt in tqdm(zip(all_measurements, ldts)):  # N x {10,20} x 4
        # labels read 0-based
        if ldt[0] in ignored_labels:
            continue
        label = new2old_labels[ldt[0]]
        device_id = ldt[1]
        timestamp = ldt[2]
        start_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        for i in meas:
            item = (
                f"{device_id},{start_time},{-1},{label},{i[0]:.6f},{i[1]:.6f},"
                f"{i[2]:.6f},{i[3]:.6f}\n"
            )
            items.append(item)

    with open(save_file, "w") as f:
        for item in items:
            f.write(item)

# json_file = Path("/home/fatemeh/Downloads/bird/data/set1/data/combined.json")
# save_file = Path("/home/fatemeh/Downloads/bird/data/final/s_data_orig.csv")
# write_j_data_orig(json_file, save_file, new2old_labels = {k:k for k in ind2name}, ignored_labels=[])

# dpath = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne")
# save_file = Path("/home/fatemeh/Downloads/bird/data/final/j_data_orig.csv")
# json_file = dpath / "combined.json"
# bd.combine_jsons_to_one_json(list(dpath.glob("*json")), json_file)
# new2old_labels = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 9: 9, 10: 9}
# ignored_labels = [8, 14, 15, 16, 17]
# write_j_data_orig(json_file, save_file, new2old_labels, ignored_labels)

# b = set(list(map(int, np.unique(df[0])))) #22
# [(df[df[0]==533][1].min(), df[df[0]==i][1].max()) for i in b] 
# [(6016, '2012-05-15 03:10:11', '2015-05-01 11:30:06'), (781, '2012-05-15 03:10:11', '2013-05-29 22:18:28'), (782, '2012-05-15 03:10:11', '2014-05-31 22:43:03'), (533, '2012-05-15 03:10:11', '2012-05-27 03:50:46'), (534, '2012-05-15 03:10:11', '2013-06-10 23:03:18'), (537, '2012-05-15 03:10:11', '2013-06-06 23:04:18'), (541, '2012-05-15 03:10:11', '2012-05-18 14:55:06'), (798, '2012-05-15 03:10:11', '2012-06-13 22:38:46'), (805, '2012-05-15 03:10:11', '2014-06-07 23:09:35'), (806, '2012-05-15 03:10:11', '2014-05-16 23:34:31'), (6073, '2012-05-15 03:10:11', '2016-06-07 12:45:07'), (6206, '2012-05-15 03:10:11', '2015-06-28 20:25:54'), (1600, '2012-05-15 03:10:11', '2014-05-20 15:28:57'), (6080, '2012-05-15 03:10:11', '2014-06-26 08:17:12'), (6208, '2012-05-15 03:10:11', '2015-07-04 10:50:16'), (6210, '2012-05-15 03:10:11', '2016-05-09 11:09:55'), (606, '2012-05-15 03:10:11', '2014-06-08 12:52:07'), (608, '2012-05-15 03:10:11', '2013-05-31 23:44:40'), (871, '2012-05-15 03:10:11', '2014-05-20 13:08:17'), (754, '2012-05-15 03:10:11', '2013-06-03 07:17:02'), (757, '2012-05-15 03:10:11', '2014-05-18 23:26:24'), (6011, '2012-05-15 03:10:11', '2015-04-30 09:31:06')]# df_m[df_m.duplicated(df_m)].index


def write_j_data(
    json_file, save_file, database_url, new2old_labels, ignored_labels, glen=20
):
    """
    read json data and append indices

    input:  device, time, index, label, imux, imuy, imuz, gps
    e.g. row: 757,2014-05-18 06:58:26,20,0,-0.09648467,-0.04426107,0.45049885,8.89139205
    """

    all_measurements, ldts = bd.combine_all_data(json_file)

    file = open(save_file, "w")
    for meas, ldt in tqdm(zip(all_measurements, ldts)):  # N x {10,20} x 4
        # labels read 0-based
        if ldt[0] in ignored_labels:
            continue
        label = new2old_labels[ldt[0]]
        device_id = ldt[1]
        timestamp = ldt[2]
        start_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        igs, idts, _ = bd.get_data(
            database_url, device_id, start_time, start_time, glen
        )
        if len(igs) == 0:
            print("not in database", device_id, start_time, meas[0])
            continue
        ind = find_matching_index(igs[:, 0:3], meas[0:2, :3], glen)
        if ind == -1:
            print("not in database", device_id, start_time)
            continue
        indices = idts[ind : ind + glen, 0]
        sel_igs = igs[ind : ind + glen]

        for i, index in zip(sel_igs, indices):
            item = (
                f"{device_id},{start_time},{index},{label},{i[0]:.6f},{i[1]:.6f},"
                f"{i[2]:.6f},{i[3]:.6f}\n"
            )
            file.write(item)
        file.flush()
            
# def write_j_data(
#     json_file, save_file, database_url, new2old_labels, ignored_labels, glen=20
# ):
#     """
#     read json data and append indices

#     input:  device, time, index, label, imux, imuy, imuz, gps
#     e.g. row: 757,2014-05-18 06:58:26,20,0,-0.09648467,-0.04426107,0.45049885,8.89139205
#     """

#     all_measurements, ldts = bd.combine_all_data(json_file)

#     items = []
#     for meas, ldt in tqdm(zip(all_measurements, ldts)):  # N x {10,20} x 4
#         # labels read 0-based
#         if ldt[0] in ignored_labels:
#             continue
#         label = new2old_labels[ldt[0]]
#         device_id = ldt[1]
#         timestamp = ldt[2]
#         start_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
#             "%Y-%m-%d %H:%M:%S"
#         )
#         igs, idts, _ = bd.get_data(
#             database_url, device_id, start_time, start_time, glen
#         )
#         if len(igs) == 0:
#             print(device_id, start_time)
#             continue
#         ind = find_matching_index(igs[:, 0:3], meas[0, :3], 1)
#         indices = idts[ind : ind + glen, 0]
#         sel_igs = igs[ind : ind + glen]
#         if ind + glen > len(igs):
#             print("not in database; use json data", device_id, start_time, indices[0])
#             if len(meas) != glen:
#                 continue
#             sel_igs = meas
#             indices = np.concatenate((indices, -np.ones(glen - len(indices))))

#         for i, index in zip(sel_igs, indices):
#             item = (
#                 f"{device_id},{start_time},{index},{label},{i[0]:.8f},{i[1]:.8f},"
#                 f"{i[2]:.8f},{i[3]:.8f}\n"
#             )
#             items.append(item)

#     with open(save_file, "w") as f:
#         for item in items:
#             f.write(item)


def write_m_info(mat_file, save_file, new2old_labels, ignored_labels):
    items = []
    dd = loadmat(mat_file)["outputStruct"]
    n_data = dd["nOfSamples"][0][0][0][0]

    for i in range(n_data):
        year, month, day, hour, min, sec = (
            dd["year"][0][0][0, i],
            dd["month"][0][0][0, i],
            dd["day"][0][0][0, i],
            dd["hour"][0][0][0, i],
            dd["min"][0][0][0, i],
            dd["sec"][0][0][i, 0],
        )
        t = datetime(year, month, day, hour, min, sec).strftime("%Y-%m-%d %H:%M:%S")

        device_id = dd["sampleID"][0][0][0, i]
        imu_x = dd["accX"][0][0][i][0][0]
        tags = dd["tags"][0][0][0][i]

        labels = tags[tags[:, 1] == 1][:, 0] - 1  # 0-based
        labels = list(set(labels))
        if len(labels) == 0:
            continue

        len_labels = []
        nlabels = []
        for label in labels:
            # 0-based
            inds = np.where((tags[:, 1] == 1) & (tags[:, 0] - 1 == label))[0]
            if label in ignored_labels:
                continue
            nlabels.append(new2old_labels[label])
            len_labels.append(len(inds))
        if len(nlabels) == 0:
            continue

        ulabels = "_".join(map(str, nlabels))
        len_labels = "_".join(map(str, len_labels))

        item = f"{device_id},{t},{len_labels},{ulabels},{len(imu_x)}\n"
        items.append(item)

    with open(save_file, "a") as f:
        for item in items:
            f.write(item)


def write_m_data(mat_file, save_file, new2old_labels, ignored_labels, database_url):
    """
    index is -1: not available and not getting from database
    """
    items = []
    dd = loadmat(mat_file)["outputStruct"]
    n_data = dd["nOfSamples"][0][0][0][0]

    for i in range(n_data):
        year, month, day, hour, min, sec = (
            dd["year"][0][0][0, i],
            dd["month"][0][0][0, i],
            dd["day"][0][0][0, i],
            dd["hour"][0][0][0, i],
            dd["min"][0][0][0, i],
            dd["sec"][0][0][i, 0],
        )
        t = datetime(year, month, day, hour, min, sec).strftime("%Y-%m-%d %H:%M:%S")

        device_id = dd["sampleID"][0][0][0, i]
        imu_x = dd["accX"][0][0][i][0][0]
        imu_y = dd["accY"][0][0][i][0][0]
        imu_z = dd["accZ"][0][0][i][0][0]
        gps_single = dd["gpsSpd"][0][0][i, 0]
        tags = dd["tags"][0][0][0][i]

        labels = tags[tags[:, 1] == 1][:, 0] - 1  # 0-based
        labels = list(set(labels))
        if len(labels) == 0:
            continue

        for label in labels:
            inds = np.where((tags[:, 1] == 1) & (tags[:, 0] - 1 == label))[0]  # 0-based
            if label in ignored_labels:
                continue
            nlabel = new2old_labels[label]

            if len(inds) < 14:
                continue
            max_len = (
                (len(inds) // 20 + 1) * 20
                if 14 <= len(inds) < 20
                else (len(inds) // 20) * 20
            )

            if 14 <= len(inds) < 20:  # TODO
                igs, idts, _ = bd.get_data(database_url, device_id, t, t)
                if len(igs) < 20:  # TODO
                    continue
                ig = np.array([imu_x[0], imu_y[0], imu_z[0]])
                ind = find_matching_index(igs[:, :3], ig, step=1)
                if ind == -1:
                    continue
                if (ind + max_len) > len(igs):
                    continue
                sel_igs = igs[ind : ind + max_len]
                sel_idts = idts[ind : ind + max_len]
                for ig, idt in zip(sel_igs, sel_idts):
                    item = f"{device_id},{t},{idt[0]},{nlabel},{ig[0]:.8f},{ig[1]:.8f},{ig[2]:.8f},{ig[3]:.8f}\n"
                    items.append(item)

            if len(inds) >= 20:  # TODO
                s_imu_x = imu_x[inds[0] : inds[0] + max_len]
                s_imu_y = imu_y[inds[0] : inds[0] + max_len]
                s_imu_z = imu_z[inds[0] : inds[0] + max_len]
                for x, y, z in zip(s_imu_x, s_imu_y, s_imu_z):
                    item = f"{device_id},{t},{-1},{nlabel},{x:.8f},{y:.8f},{z:.8f},{gps_single:.8f}\n"
                    items.append(item)

    with open(save_file, "a") as f:
        for item in items:
            f.write(item)


def write_w_info(csv_file, save_file):
    info = defaultdict(list)
    with open(csv_file, "r") as f:
        for r in f:
            r = r.strip().split(", ")
            device_id = r[0]
            t = datetime.strptime(r[1], "%m/%d/%Y %H:%M:%S").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            label, conf = r[-1].split(" ")
            if conf == "0":
                continue
            if r[3] == "NaN" or r[4] == "NaN" or r[5] == "NaN" or r[6] == "NaN":
                continue
            # key = (device_id, t, str(int(label) - 1))  # zero-based
            key = (device_id, t)
            info[key].extend([str(int(label) - 1)])  # zero-based
    with open(save_file, "a") as f:
        for k, v in info.items():
            count_labels = dict(Counter(v))
            counts = "_".join([str(i) for i in count_labels.values()])
            labels = "_".join([i for i in count_labels.keys()])
            item = f"{k[0]},{k[1]},{counts},{labels},{0}\n"
            f.write(item)


def write_w_data(csv_file, save_file, database_url):
    info = defaultdict(list)
    with open(csv_file, "r") as f:
        for r in f:
            r = r.strip().split(", ")
            device_id = r[0]
            t = datetime.strptime(r[1], "%m/%d/%Y %H:%M:%S").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            label, conf = r[-1].split(" ")
            if conf == "0":
                continue
            if r[3] == "NaN" or r[4] == "NaN" or r[5] == "NaN" or r[6] == "NaN":
                continue
            item = [r[2], r[3], r[4], r[5], r[6]]
            key = (device_id, t, str(int(label) - 1))  # zero-based
            info[key].append(item)
    items = []

    for k, values in info.items():
        device_id = int(k[0])
        start_time = k[1]
        label = k[2]
        if len(values) >= 20:  # TODO
            max_seq_len = (len(values) // 20) * 20
            values = values[:max_seq_len]
            for value in values:
                v = [float(i) for i in value]
                item = f"{device_id},{start_time},{int(v[0])},{label},{v[1]:.8f},{v[2]:.8f},{v[3]:.8f},{v[4]:.8f}\n"
                items.append(item)
        if len(values) < 14:  # TODO
            continue
        if 14 <= len(values) < 20:  # TODO
            igs, idts, _ = bd.get_data(database_url, device_id, start_time, start_time)
            if len(igs) < 20:  # TODO
                continue
            ig = np.array(list(map(float, values[0][1:4])))
            ind = find_matching_index(igs[:, :3], ig, step=1)
            if ind == -1:
                continue
            max_seq_len = (len(values) // 20 + 1) * 20  # TODO
            if (ind + max_seq_len) > len(igs):
                continue
            sel_igs = igs[ind : ind + max_seq_len]
            sel_idts = idts[ind : ind + max_seq_len]
            for ig, idt in zip(sel_igs, sel_idts):
                item = f"{device_id},{start_time},{idt[0]},{label},{ig[0]:.8f},{ig[1]:.8f},{ig[2]:.8f},{ig[3]:.8f}\n"
                items.append(item)

    with open(save_file, "a") as f:
        for item in items:
            f.write(item)


def combine(csv_files):
    combined = defaultdict(list)
    for csv_file in csv_files:
        current = defaultdict(list)
        with open(csv_file, "r") as f:
            for r in f:
                r = r.strip().split(",")
                device_id = r[0]
                date_time = r[1]
                key = (device_id, date_time)
                item = [r[2], r[3], r[4], r[5], r[6], r[7]]
                current[key].append(item)
        if len(combined) == 0:
            combined = deepcopy(current)
        # Replace with the largest value
        for key, value in current.items():
            if key in combined:
                print(key, len(value), len(combined[key]))
                if len(value) > len(combined[key]):
                    combined[key] = value
            if key not in combined:
                combined[key] = value
    return combined


def save_dict_or_list_as_csv(save_file, data):
    with open(save_file, "w") as f:
        for key, values in data.items():
            for value in values:
                item = (*key, *value)
                f.write(",".join(item) + "\n")


def load_any_csv(csv_file, header=False):
    data = []
    with open(csv_file, "r") as f:
        if header:
            next(f)
        for r in f:
            item = r.strip().split(",")
            data.append(item)
    return data


def save_anything_as_csv(save_file, data):
    with open(save_file, "w") as f:
        if isinstance(data, dict):
            for key, value in data.items():
                item = (*key, *value)
                f.write(",".join(item) + "\n")
        if isinstance(data, list):
            for item in data:
                f.write(",".join(item) + "\n")


def data1_diff_data2(sdata1, sdata2):
    data1 = {(row[0], row[1]) for row in sdata1}
    data2 = {(row[0], row[1]) for row in sdata2}
    return data1.difference(data2)


def data1_common_data2(sdata1, sdata2):
    data1 = {(row[0], row[1]) for row in sdata1}
    data2 = {(row[0], row[1]) for row in sdata2}
    return data1.intersection(data2)


def data1_common_data2_labels_all(data1, data2):
    data2_dict = {(row[0], row[1]): row[3] for row in data2}

    common = defaultdict(list)
    for row in data1:
        key = (row[0], row[1])
        if key in data2_dict:
            common[key] = [row[3], data2_dict[key]]

    return common


def change_time_string(time):
    # '2012-05-15T12:14:14.000Z' -> '2012-05-15 12:14:14'
    return datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ").strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# json_file = Path("/home/fatemeh/Downloads/bird/data/set1/data/combined.json")
# save_file = Path("/home/fatemeh/Downloads/bird/data/final/s_data.csv")
# write_j_data(json_file, save_file, database_url, new2old_labels = {k:k for k in ind2name}, ignored_labels=[], glen=20)
new2old_labels = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 9: 9, 10: 9}
ignored_labels = [8, 14, 15, 16, 17]
json_file = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne/combined.json")
save_file = Path("/home/fatemeh/Downloads/bird/data/final/j_data.csv")
write_j_data(json_file, save_file, database_url, new2old_labels, ignored_labels, glen=20)


"""
# Step1: create all the data separately
# ======

# data
# ====
database_url = "postgresql://username:password@host:port/database_name"
save_path = Path("/home/fatemeh/Downloads/bird/data")

# set1 (s_data, s_info)
# ====================
dpath = Path("/home/fatemeh/Downloads/bird/data/set1/data")
json_file = dpath / "combined.json"
bd.combine_jsons_to_one_json(list(dpath.glob("*json")), json_file)
json_file = Path("/home/fatemeh/Downloads/bird/data/set1/data/combined.json")
save_file = Path("/home/fatemeh/Downloads/bird/data/final/s_data.csv")
write_j_data(json_file, save_file, database_url, new2old_labels = {k:k for k in ind2name}, ignored_labels=[], glen=20)
# write_info(save_path/"s_data.csv", save_path /"s_info.csv")


# Sus (j_data, j_info)
# ====================
dpath = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne")
# bd.combine_jsons_to_one_json(list(dpath.glob("*json")), json_file)
json_file = dpath / "combined.json"
new2old_labels = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 9: 9, 10: 9}
ignored_labels = [8, 14, 15, 16, 17]
json_file = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne/combined.json")
save_file = Path("/home/fatemeh/Downloads/bird/data/final/j_data.csv")
write_j_data(json_file, save_file, database_url, new2old_labels, ignored_labels, glen=20)
# write_info(save_file, save_path /"j_info.csv")


# Sus (m_data, m_info)
# ====================
dpath = Path("/home/fatemeh/Downloads/bird/data_from_Susanne")
new2old_labels = {0: 0, 5: 5, 6: 6, 11: 9, 13: 9}
ignored_labels = [10, 14, 15, 16, 17]
for mat_file in tqdm(dpath.glob("An*mat")):
    print(mat_file.name)
    write_m_info(
        mat_file, save_path / "m_info_orig_length.csv", new2old_labels, ignored_labels
    )
    write_m_data(mat_file, save_path / "m_data.csv", new2old_labels, ignored_labels, database_url)
write_info(save_path/"m_data.csv", save_path /"m_info.csv")

# Willem (w_data, w_info)
# =======================
dpath = Path("/home/fatemeh/Downloads/bird/data_from_Willem")
for p in dpath.glob("*csv"):
    write_w_data(p, save_path/"w_data.csv", database_url)
    write_w_info(p, save_path/"w_info_orig_length.csv")
write_info(save_path/"w_data.csv", save_path /"w_info.csv")
"""

"""
# !!! > This part is older version only based on id and date, result in less data.
# !!! > This part is not used anymore. Go to step 2.

# Combine all
# ===========
combined = combine([save_path/i for i in ["s_data.csv", "w_data.csv", "m_data.csv", "j_data.csv"]])
save_dict_or_list_as_csv(save_path/"combined_s_w_m_j.csv", combined)
write_info(save_path/"combined_s_w_m_j.csv", save_path/"combined_s_w_m_j_info.csv")
# combined = combine([save_path/i for i in ["w_data.csv", "s_data.csv"]])
# save_dict_or_list_as_csv(save_path/"combined_w_s.csv", combined)
# write_info(save_path/"combined_w_s.csv", save_path/"combined_w_s_info.csv")

# Not used:
# ===========
# Set2, Set3
dpath = Path("/home/fatemeh/Downloads/bird/bird/set3/data")
bd.combine_jsons_to_one_json(list(dpath.glob("*json")), dpath/"combined.json")
write_j_info(dpath/"combined.json", save_path/"set3_json_info_orig.csv")

dpath = Path("/home/fatemeh/Downloads/bird/bird/set2/data")
write_j_info(dpath/"combined.json", save_path/"set2_json_info_orig.csv")

# judy annotations
dpath = Path("/home/fatemeh/Downloads/bird/judy_annotations")
data = []
for p in dpath.glob("*csv"):
    data.extend(load_any_csv(p, True))
data = [[i[0],change_time_string(i[1]),'0',str(name2ind[i[2]]),'0'] for i in data]
common = data1_common_data2_labels_all(sdata, data)
common = [[i[0],i[1],'0',i[3],'0'] for i in common]
[data.remove(i) for i in common]
save_anything_as_csv(save_path/"judy_info.csv", data)

# Diff and common
# ===============
save_path = Path("/home/fatemeh/Downloads/bird/data")

# set2
# sinfo = load_any_csv(save_path / "s_info.csv")
# sinfo2 = load_any_csv(save_path/"set2_json_info_orig.csv")

# common = data1_common_data2_labels_all(sinfo, sinfo2)
# save_anything_as_csv(save_path/"s1_s2_common.csv", common)
# diff = data1_diff_data2(sinfo2, sinfo) # empty

# willem
sinfo = load_any_csv(save_path / "s_info.csv")
winfo = load_any_csv(save_path / "w_info.csv")

diff = data1_diff_data2(winfo, sinfo)
data_dict = {(i[0], i[1]): [i[2], i[3], i[4]] for i in winfo}
diff_dict = {key: data_dict[key] for key in diff}
save_anything_as_csv(save_path / "w_s_diff.csv", diff_dict)

diff = data1_diff_data2(sinfo, winfo)
data_dict = {(i[0], i[1]): [i[2], i[3], i[4]] for i in sinfo}
diff_dict = {key: data_dict[key] for key in diff}
save_anything_as_csv(save_path / "s_w_diff.csv", diff_dict)

common = data1_common_data2_labels_all(sinfo, winfo)
save_anything_as_csv(save_path / "w_s_common.csv", common)

# jinfo = load_any_csv(save_path / "j_info.csv")
# minfo = load_any_csv(save_path / "m_info.csv")
# common = data1_common_data2_labels_all(jinfo, minfo)
# save_anything_as_csv(save_path / "j_m_common.csv", common)

# Stats
# =====
sdata = load_any_csv(save_path / "s_data.csv")
wdata = load_any_csv(save_path / "w_data.csv")
jdata = load_any_csv(save_path / "j_data.csv")
mdata = load_any_csv(save_path / "m_data.csv")
data = load_any_csv(save_path / "combined_s_w_m_j.csv")

for item in [sdata, wdata, jdata, mdata, data]:
    labels = [int(i[3]) for i in item]
    unique_labels = set(labels)
    label_counts = dict(sorted(dict(Counter(labels)).items(), key=lambda x:x[0]))
    label_counts = {k:v//20 for k,v in label_counts.items()}
    print(len(item)//20)
    print(unique_labels)
    print(label_counts)

# sdata
# 3505
# {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
# {0: 634, 1: 38, 2: 501, 3: 176, 4: 558, 5: 894, 6: 318, 7: 25, 8: 151, 9: 210}
# wdata
# 3320
# {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
# {0: 639, 1: 42, 2: 496, 3: 176, 4: 558, 5: 802, 6: 293, 7: 26, 8: 141, 9: 147}
# jdata
# 1472
# {0, 1, 2, 4, 5, 6, 7, 8, 9}
# {0: 258, 1: 24, 2: 229, 4: 284, 5: 413, 6: 130, 7: 12, 8: 74, 9: 48}
# mdata
# 757
# {0, 9, 5, 6}
# {0: 5, 5: 610, 6: 18, 9: 124}
# combined
# 4394
# {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
# {0: 643, 1: 41, 2: 541, 3: 176, 4: 623, 5: 1507, 6: 342, 7: 29, 8: 150, 9: 342}

# example with NULL at last item
device_id, start_time = 6208, '2015-07-04 10:46:22' # 18
device_id, start_time = 6073, '2016-06-07 12:43:47' # 17

# plot data
# 33,2012-05-27 03:05:00 (18), 533,2012-05-27 03:05:00(2)
database_url = "postgresql://username:password@host:port/database_name"
device_id = 533
start_time = "2012-05-15 09:47:29"
igs, idts, llat = bd.get_data(database_url, device_id, start_time, start_time)
bu.plot_one(igs[:20])
bu.plot_one(igs[20:40])
bu.plot_one(igs[40:])
"""


"""
# Duplicates identified
df = pd.read_csv("/home/fatemeh/Downloads/bird/data/data/combined_s_w_m_j.csv", header=None)

# removed: 
# 6011,2015-04-30 09:10:31 (label 5, 9-> 9 correct)  (120) 71760, 71780, 71860
(71760, 71780, 71860), (71900, 71920), (72380, 72400)
bu.plot_one(np.array(df[[4,5,6,7]].iloc[71760:71760+20]))
bu.plot_one(np.array(df[[4,5,6,7]].iloc[71780:71780+20]))

# maybe for figures
for i in range(10):
    sel_by_label = df_20[df_20[3]==i]
    sel_by_label[[0,1,4,5,6,7]].to_csv(f"/home/fatemeh/Downloads/{i}.csv", header=None, index=False)
"""


"""
# Before Step2:
# =====
# Since s_data.csv generated before. The indices are only the starting indices. Here they are changed as
# they are in the database (zero-based).


# def modify_index(group):
#     if len(group) == 20:
#         # Get the starting value from the third column (df[2])
#         start_value = int(group.iloc[0, 2])
#         # Replace with sequential numbers
#         group[2] = range(start_value, start_value + 20)
#     return group


# df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/s_data.csv", header=None)
# grouped = df.groupby([0, 1, 2, 3], sort=False)  # otherwise group is sorting
# modified_df2 = (
#     grouped[[0, 1, 2, 3, 4, 5, 6, 7]].apply(modify_index).reset_index(drop=True)
# )
# modified_df2.to_csv(
#     "/home/fatemeh/Downloads/bird/data/final/s_data_modified.csv",
#     index=False,
#     header=None,
#     float_format="%.8f",
# )

# Step2: combine data
# =====

# Combine csv files
# save_path = Path("/home/fatemeh/Downloads/bird/data/final")
# csv_files = [
#     save_path / i
#     for i in ["s_data_modified.csv", "w_data.csv", "m_data.csv", "j_data.csv"]
# ]
# df_list = []
# for csv_file in csv_files:
#     df = pd.read_csv(csv_file, header=None)
#     df[[4, 5, 6, 7]] = df[[4, 5, 6, 7]].round(6)
#     df_list.append(df)
# combined_df = pd.concat(df_list, ignore_index=True)
# combined_df.to_csv(save_path / "combined.csv", index=False, header=None)


# Find duplicates
def group_equal_elements(df, subset, indices, equal_func):
    groups = []  # List to store groups of equal elements
    visited = set()  # Set to track which indices we have already grouped

    for i in tqdm(range(len(indices))):
        if indices[i] in visited:
            continue  # Skip if this index is already part of a group

        # Start a new group with the current index
        current_group = [indices[i]]
        visited.add(indices[i])

        for j in range(i + 1, len(indices)):
            if indices[j] not in visited:
                IS_EQUAL = equal_func(df, subset, indices[i], indices[j])
                if IS_EQUAL:
                    current_group.append(indices[j])
                    visited.add(indices[j])

        # Add the group to the list of groups
        groups.append(current_group)

    return groups

def equal_func(df, subset, ind1, ind2):
    set1 = df[subset].iloc[ind1 : ind1 + 20].reset_index(drop=True)
    set2 = df[subset].iloc[ind2 : ind2 + 20].reset_index(drop=True)
    return set1.equals(set2)


def group_equal_elements_optimized(df, subset, indices):
    hashes = {}
    for idx in tqdm(indices):
        # Round the float columns to avoid precision issues
        set1 = df[subset].iloc[idx:idx + 20].reset_index(drop=True)
        set1_rounded = set1.round(6)
        group_rows = pd.util.hash_pandas_object(set1_rounded).values.tobytes()

        if group_rows not in hashes:
            hashes[group_rows] = []
        hashes[group_rows].append(idx)

    # Extract groups of duplicates from the hash map
    groups = [group for group in hashes.values() if len(group) > 1]
    
    return groups

subset = [0, 1, 4, 5, 6, 7]
save_path = Path("/home/fatemeh/Downloads/bird/data/final")
df = pd.read_csv(save_path / "combined_unique.csv", header=None)
df_20 = df.iloc[::20]
sel_df_20 = df_20[subset]
inds = sel_df_20[sel_df_20.duplicated(keep=False)].index
print(len(inds))

groups = group_equal_elements(df, subset, inds, equal_func)
# groups = group_equal_elements_optimized(df, subset, inds)
groups = [list(map(int, g)) for g in groups]
print(groups)

# Collect all the indices to be dropped
to_drop = []
for g in groups:
    if len(g) > 1:
        # Collect indices to drop (ignoring the first one)
        to_drop.extend(range(i, i + 20) for i in g[1:])

# Flatten the list of indices to drop
to_drop = [item for sublist in to_drop for item in sublist]

# Drop all the collected rows at once
df = df.drop(to_drop)

df.to_csv(
    save_path / "combined_unique2.csv", index=False, header=None, float_format="%.6f"
)
"""

"""
# Sanity check: check if each group of 20 elemetns has the same device id, date and label
def validate_group_id_date_label(filename):
    # This function checks if each group has the same device id, date and label

    df = pd.read_csv(filename, header=None)

    # Define the group size
    group_size = 20

    # Add a group number column based on row index
    df["group_number"] = df.index // group_size

    # Iterate over each group
    for group_num, group in df.groupby("group_number"):
        # Reset index for clarity
        group = group.reset_index(drop=True)

        # Extract the reference values from the first row
        first_col = group.iloc[0, 0]
        second_col = group.iloc[0, 1]
        fourth_col = group.iloc[0, 3]

        # Check if all values in the columns match the reference values
        first_col_match = (group[0] == first_col).all()
        second_col_match = (group[1] == second_col).all()
        fourth_col_match = (group[3] == fourth_col).all()

        # Report mismatches
        if not (first_col_match and second_col_match and fourth_col_match):
            print(f"Mismatch found in group {group_num + 1}:")
            if not first_col_match:
                print(" - First column values are not the same.")
            if not second_col_match:
                print(" - Second column values are not the same.")
            if not fourth_col_match:
                print(" - Fourth column values are not the same.")

            # Optionally, display differing rows
            differing_rows = group[
                (group[0] != first_col)
                | (group[1] != second_col)
                | (group[3] != fourth_col)
            ]
            print("Differing rows:")
            print(differing_rows)

    # Check for incomplete final group
    total_rows = len(df)
    if total_rows % group_size != 0:
        print(f"Warning: The last group contains less than {group_size} rows.")


validate_group_id_date_label(
    "/home/fatemeh/Downloads/bird/data/final/combined_unique.csv"
)

# Grouped is the number of unique values for each group of 20 elements.
df = pd.read_csv(
    "/home/fatemeh/Downloads/bird/data/final/combined_unique.csv", header=None
)
df["group_number"] = df.index // 20
grouped = df.groupby("group_number")[[0, 1, 3]].nunique()
"""

"""
# df = pd.read_csv("/home/fatemeh/Downloads/combined_unique.csv", header=None)
# df2 = df[df[3]==2].copy()
# group[[4,5,6,7]] = group[[4,5,6,7]].round(6)

df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/combined_unique.csv", header=None)
df8 = df[df[3]==8].copy()

grouped = df8.groupby([0,1], sort=False)

for name in grouped.groups.keys():
    print(name)
    print(grouped.get_group(name).sort_values(by=2))

group = grouped.get_group((805, '2014-06-07 09:34:03')).sort_values(by=2)

# 176100, 22840, 6700
"""
