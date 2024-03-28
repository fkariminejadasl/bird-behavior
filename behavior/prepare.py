from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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


def find_matching_index(array, target, step=20, tolerance=1e-5):
    for i in range(0, len(array), step):
        if all(np.isclose(array[i], target, atol=tolerance)):
            return i
    return -1


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
        t = datetime.utcfromtimestamp(ldt[2]).strftime("%Y-%m-%d %H:%M:%S")
        item = f"{ldt[1]},{t},{len(ig)},{ldt[0]},0\n"
        items.append(item)
    with open(csv_file, "w") as f:
        for item in items:
            f.write(item)


def write_j_data(
    json_file, save_file, database_url, new2old_labels, ignored_labels, glen=20
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
        start_time = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        igs, idts, _ = bd.get_data(
            database_url, device_id, start_time, start_time, glen
        )
        if len(igs) == 0:
            print(device_id, start_time)
            continue
        ind = find_matching_index(igs[:, 0:3], meas[0, :3], 1)
        indices = idts[ind : ind + glen, 0]
        sel_igs = igs[ind : ind + glen]
        if ind + glen > len(igs):
            print("not in database; use json data", device_id, start_time, indices[0])
            if len(meas) != glen:
                continue
            sel_igs = meas
            indices = np.concatenate((indices, -np.ones(glen - len(indices))))

        for i, index in zip(sel_igs, indices):
            item = (
                f"{device_id},{start_time},{index},{label},{i[0]:.8f},{i[1]:.8f},"
                f"{i[2]:.8f},{i[3]:.8f}\n"
            )
            items.append(item)

    with open(save_file, "w") as f:
        for item in items:
            f.write(item)


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


"""
# data
# ====
database_url = "postgresql://username:password@host:port/database_name"
save_path = Path("/home/fatemeh/Downloads/bird/data")

# set1 (s_data, s_info)
# ====================
dpath = Path("/home/fatemeh/Downloads/bird/bird/set1/data")
save_file = save_path / "s_data.csv"
json_file = dpath / "combined.json"
bd.combine_jsons_to_one_json(list(dpath.glob("*json")), json_file)
write_j_data(json_file, save_file, database_url, new2old_labels = {k:k for k in ind2name}, ignored_labels=[], glen=20)
write_info(save_path/"s_data.csv", save_path /"s_info.csv")

# Sus (j_data, j_info)
# ====================
dpath = Path("/home/fatemeh/Downloads/bird/data_from_Susanne")
save_file = save_path / "j_data.csv"
# bd.combine_jsons_to_one_json(list(dpath.glob("*json")), json_file)
json_file = dpath / "combined.json"
new2old_labels = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 9: 9, 10: 9}
ignored_labels = [8, 14, 15, 16, 17]
write_j_data(json_file, save_file, database_url, new2old_labels, ignored_labels, glen=20)
write_info(save_file, save_path /"j_info.csv")


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
"""


"""
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
"""

"""
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
"""

"""
# example with NULL at last item
device_id, start_time = 6208, '2015-07-04 10:46:22' # 18
device_id, start_time = 6073, '2016-06-07 12:43:47' # 17
"""

""" 
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
