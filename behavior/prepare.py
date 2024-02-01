from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from behavior import data as bd
from behavior import utils as bu

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


def find_matching_index(array, target, step=20, tolerance=1e-5):
    for i in range(0, len(array), step):
        if all(np.isclose(array[i], target, atol=tolerance)):
            return i
    return -1


def write_as_csv(save_file, device_id, date, index, label, igs):
    """
    input:  device, time, index, label, imux, imuy, imuz, gps
    e.g. row: 757,2014-05-18 06:58:26,20,0,-0.09648467,-0.04426107,0.45049885,8.89139205
    """
    with open(save_file, "a") as file:
        for ig in igs:
            text = (
                f"{device_id},{date},{index},{label},{ig[0]:.8f},{ig[1]:.8f},"
                f"{ig[2]:.8f},{ig[3]:.8f}\n"
            )
            file.write(text)


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


def convert_csv_files(csv_file, output_file):
    """
    input:  device, time, index, label, imux, imuy, imuz, gps
    output: device, time, count, label, ind
    e.g.
    input:  757,2014-05-18 06:58:26,20,0,-0.09648467,-0.04426107,0.45049885,8.89139205
    output: 757,2014-05-18 06:58:26,20,0,40
    """

    counts = defaultdict(int)
    with open(csv_file, "r") as rfile:
        for row in rfile:
            row = row.strip().split(",")
            device_id = row[0]
            t = row[1]
            ind = row[2]
            label = row[3]
            key = (device_id, t, ind, label)
            counts[key] += 1
    with open(output_file, "w") as wfile:
        for key, count in counts.items():
            item = f"{key[0]},{key[1]},{count},{key[3]},{key[2]}\n"
            wfile.write(item)


def write_mat_info(mat_file, save_file):
    dd = loadmat(mat_file)["outputStruct"]
    n_data = dd["nOfSamples"][0][0][0][0]
    with open(save_file, "a") as f:
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

            tags = dd["tags"][0][0][0][i]
            ids = tags[tags[:, 1] == 1][:, 0] - 1  # zero-based
            if len(ids) == 0:
                continue
            uids = list(np.unique(ids))
            len_ids = [len(ids[ids == i]) for i in uids]
            uids = "_".join(map(str, uids))
            len_ids = "_".join(map(str, len_ids))

            device_id = dd["sampleID"][0][0][0, i]
            imu_x = dd["accX"][0][0][i][0][0]
            imu_y = dd["accY"][0][0][i][0][0]
            imu_z = dd["accZ"][0][0][i][0][0]
            gps_single = dd["gpsSpd"][0][0][i, 0]

            item = f"{device_id},{t},{len(imu_x)},{uids},{len_ids}\n"
            f.write(item)


def write_csv_info(csv_file, save_file):
    info = defaultdict(int)
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
            key = (device_id, t, str(int(label) - 1))  # zero-based
            info[key] += 1
    with open(save_file, "a") as f:
        for k, v in info.items():
            item = f"{k[0]},{k[1]},{v},{k[2]},{0}\n"
            f.write(item)


def save_csv_info_from_json_zero_ind(json_file, csv_file):
    # output: device, time, count, label, ind
    igs, ldts = bd.combine_all_data(json_file)
    with open(csv_file, "w") as f:
        for ig, ldt in zip(igs, ldts):
            t = datetime.utcfromtimestamp(ldt[2]).strftime("%Y-%m-%d %H:%M:%S")
            item = f"{ldt[1]},{t},{len(ig)},{ldt[0]},0\n"
            f.write(item)


def append_indices(json_file, save_file, database_url, glen=20):
    # input:  device, time, index, label, imux, imuy, imuz, gps

    all_measurements, ldts = bd.combine_all_data(json_file)

    for meas, ldt in tqdm(zip(all_measurements, ldts)):  # N x {10,20} x 4
        label = ldt[0]
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
        write_as_csv(save_file, device_id, start_time, ind, label, meas)


def load_csv_info(csv_file):
    data = []
    with open(csv_file, "r") as f:
        for r in f:
            item = r.strip().split(",")
            if item[3] == "":
                continue
            data.append(item)
    return data


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


def data1_common_data2_labels(data1, data2):
    """
    data1,data2: list[list[str]]
    returns: device,time,label1,label2
    """
    data = dict()
    for row in data1:
        data[(row[0], row[1])] = row[3]
    found = []
    for row in data2:
        key = (row[0], row[1])
        if key in data:
            item = [row[0], row[1], data[key], row[3]]
            found.append(item)
    return found


def data1_common_data2_labels_all(data1, data2, id_new_old):
    data2_dict = {(row[0], row[1]): str(id_new_old[int(row[3])]) for row in data2}

    common = defaultdict(list)
    for row in data1:
        key = (row[0], row[1])
        if key in data2_dict:
            if len(common[key]) == 0:
                common[key] = [data2_dict[key], row[3]]
            else:
                if common[key][-1] != row[3]:
                    common[key].append(row[3])
    return common


def data1_common_data2_labels_inds_all(
    data1, data2, id_new_old, ignored_labels=["14", "15", "16", "17"]
):
    data2_dict = {
        (row[0], row[1]): [row[4], str(id_new_old[int(row[3])])] for row in data2
    }
    common = []
    for row in data1:
        key = (row[0], row[1])
        if key in data2_dict:
            if row[3] not in ignored_labels:
                item = (
                    row[0],
                    row[1],
                    data2_dict[key][0],
                    row[4],
                    data2_dict[key][1],
                    row[3],
                )
                common.append(item)
    return common


def change_time_string(time):
    # '2012-05-15T12:14:14.000Z' -> '2012-05-15 12:14:14'
    return datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ").strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def count_labels_in_data(data):
    counts = defaultdict(int)
    for item in data:
        labels = item[3].split("_")
        for label in labels:
            counts[int(label)] += 1
    return counts
    # {ind2name[k]:v for k, v in counts.items()}


"""
ind2name = {
    6, 0: "Flap",
    5, 1: "ExFlap",
    4, 2: "Soar",
       3: "Boat",
    3, 4: "Float",
    2, 5: "SitStand",
    1, 6: "TerLoco",
    8, 7: "Other",
    7, 8: "Manouvre",
       9: "Pecking",
}
# Looking_food: 9, Handling_mussel: 10, StandForage: 11
# values from json files. subtracted from one to become zero-based. 
# Then some new assignment not to be the same with old scheme.
id_new_old = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 8:10 , 9:11 , 10:13}
# Flap:2, XflapL:6, XflapS:8, Soar:3, Float:7, stand:1, sit: 5, TerLoco/walk: 4, other: 9
id_new_old = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 8:10 , 9:11 , 10:13}
"""

"""
# in jsus data
# device_id, start_time = 6208, '2015-07-04 10:46:22' # 18
# device_id, start_time = 6073, '2016-06-07 12:43:47' # 17
"""

"""
# json: 1856 (10 items; 0-10 labels)
# [(i, len(np.where(ldts[:,1]==i)[0])) for i in np.unique(ldts[:,1])]
# [(533, 17), (534, 146), (537, 32), (541, 56), (606, 147), (608, 59), (754, 15), (757, 118), (781, 7), (782, 269), (798, 57), 
#  (805, 302), (806, 43), (871, 32), (1600, 54), (6011, 67), (6016, 24), (6073, 39), (6080, 63), (6206, 16), (6208, 113), (6210, 180)]
# [(int(p.stem.split("AnnAcc")[1].split("_")[0]), loadmat(p)["outputStruct"]['annotations'][0][0].shape[0]) for p in dpath.glob("*mat")]
# [(6208, 113), (6073, 73), (6011, 158), (6210, 271), (6016, 52), (1600, 57), (6206, 54), (6080, 141)]
"""

"""
database_url = "postgresql://username:password@host:port/database_name"

# for json files
dpath = Path("/home/fatemeh/Downloads/bird/bird/set1/data")
save_path = Path("/home/fatemeh/Downloads/bird/data")
save_file = save_path / "set1.csv"
json_file = dpath / "combined.json"
append_indices(json_file, save_file, database_url, glen=20)
convert_csv_files(save_file, save_path /"set1_info.csv")

# for json files
dpath = Path("/home/fatemeh/Downloads/bird/data_from_Susanne")
save_path = Path("/home/fatemeh/Downloads/bird/data")
save_file = save_path /"sus_json.csv"
json_file = dpath / "combined.json"
# bd.combine_jsons_to_one_json(list(dpath.glob("*json")), json_file)
append_indices(json_file, save_file, database_url, glen=10)
convert_csv_files(save_file, save_path /"sus_json_info.csv")
# save_csv_info_from_json_zero_ind(dpath/"combined.json", save_path/"sus_json_info_ind0.csv")

# for mat files
for mat_file in tqdm(dpath.glob("An*mat")):
    print(mat_file.name)
    write_mat_info(mat_file, save_path / "sus_mat_info.csv")

dpath = Path("/home/fatemeh/Downloads/bird/bird/set3/data/matfiles")
bd.combine_jsons_to_one_json(list(dpath.glob("*json")), dpath/"combined.json")
save_csv_info_from_json_zero_ind(dpath/"combined.json", save_path/"set3_json_info_ind0.csv")

dpath = Path("/home/fatemeh/Downloads/bird/bird/set2/data")

# judy annotations
dpath = Path("/home/fatemeh/Downloads/bird/judy_annotations")
data = []
for p in dpath.glob("*csv"):
    data.extend(load_any_csv(p, True))
data = [[i[0],change_time_string(i[1]),'0',str(name2ind[i[2]]),'0'] for i in data]
common = data1_common_data2_labels(sdata, data)
common = [[i[0],i[1],'0',i[3],'0'] for i in common]
[data.remove(i) for i in common]
save_anything_as_csv(save_path/"judy_info.csv", data)

# Willem
dpath = Path("/home/fatemeh/Downloads/bird/data_from_Willem")
for p in dpath.glob("*csv"):
    write_csv_info(p, save_path/"will_info_ind0.csv")
"""

"""
# extras
# set2
save_path = Path("/home/fatemeh/Downloads/bird/data")
sdata1 = load_csv_info(save_path/"set1_info.csv")
sdata2 = load_csv_info(save_path/"set2_json_info_ind0.csv")

common = data1_common_data2_labels(sdata1, sdata2)
save_anything_as_csv(save_path/"set1_jset2_common2.csv", common)

# willem
sdata = load_csv_info(save_path / "set1_info.csv")
wdata = load_csv_info(save_path / "will_info_ind0.csv")

diff = data1_diff_data2(wdata, sdata)
wdata_dict = {(i[0], i[1]): [i[2], i[3], i[4]] for i in wdata}
diff_dict = {key: wdata_dict[key] for key in diff}
save_anything_as_csv(save_path / "will_set1_diff.csv", diff_dict)

common =  data1_common_data2_labels_all(sdata, wdata, {k:k for k in ind2name.keys()})
save_anything_as_csv(save_path/"will_set1_common.csv", common)
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

save_path = Path("/home/fatemeh/Downloads/bird/data")


jdata = load_csv_info(save_path / "sus_json_info_ind0.csv")
mdata = load_csv_info(save_path / "sus_mat_info.csv")
sdata = load_csv_info(save_path / "set1_info.csv")
wdata = load_csv_info(save_path / "will_info_ind0.csv")

count_labels_in_data(mdata)

len(sdata), len(jdata), len(mdata)
len(data1_diff_data2(mdata, sdata))
len(data1_common_data2(jdata, mdata))
ids = np.unique([int(i[0]) for i in mdata])
",".join([str(i) for i in ids])

"""
# jdata: 1856, sdata: 3505, mdata: 706
save_path = Path("/home/fatemeh/Downloads/bird/data")

jdata = load_csv_info(save_path/"sus_json_info_ind0.csv")
mdata = load_csv_info(save_path/"sus_mat_info.csv")
sdata = load_csv_info(save_path/"set1_info.csv")

# fmt: no
id_new_old = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 8:10 , 9:11 , 10:13}
# fmt: yes


common =  data1_common_data2_labels_all(data1, data2, id_new_old)
save_anything_as_csv(save_path / "set1_jsus_common_labels.csv", common)

with open(save_path / "set1_jsus_common_labels_different.csv", 'w') as f:
    for key, value in common.items():
        if value[0]!=value[1]:
            item = (*key, *value)
            f.write(','.join(item) + "\n")

common = data1_common_data2_labels_inds_all(mdata, jdata, id_new_old, ignored_labels=["14", "15", "16", "17"])
save_anything_as_csv(save_path / "m_jsus_common_labels.csv", common)            
"""

"""

# no common 
unique_rows = set(tuple(row) for row in mdata)
u_mdata = [list(row) for row in unique_rows]


# TODO
# csv reader for willem/set3
# convert labels
# set3: csv1, csv760, json 
# willem: csv
# check common data has the same id; 
# put all data together 
# remove commond part
# download them
"""
