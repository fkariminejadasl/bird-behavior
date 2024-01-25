from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from behavior import data as bd


def find_matching_index(array, target, step=20, tolerance=1e-5):
    for i in range(0, len(array), step):
        if all(np.isclose(array[i], target, atol=tolerance)):
            return i
    return -1


def write_as_csv(save_file, device_id, date, index, label, igs):
    """
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


"""
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

target_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # no Other
target_labels_names = [ind2name[t] for t in target_labels]

data_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data")
combined_file = data_path / "combined.json" # "combined.json"

all_measurements, ldts = bd.combine_all_data(combined_file)

save_file = Path("/home/fatemeh/Downloads/bird/result/failed/exp1/set1.csv")
database_url = "postgresql://username:password@host:port/database_name"

for meas, ldt in tqdm(zip(all_measurements, ldts)): # N x 20 x 4
    label = ldt[0]
    device_id = ldt[1]
    timestamp = ldt[2]
    start_time = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    igs, idts, _ = bd.get_data(database_url, device_id, start_time, start_time)
    ind = find_matching_index(igs[:,0:3], meas[0,:3], 1)
    write_as_csv(save_file, device_id, start_time, ind, label, meas)


write_csv_info("/home/fatemeh/Downloads/bird/data/set1.csv", "/home/fatemeh/Downloads/bird/data/set1_info.csv")
"""


def write_csv_info(csv_file, output_file):
    """
    e.g.
    input:  757,2014-05-18 06:58:26,20,0,-0.09648467,-0.04426107,0.45049885,8.89139205
    output: 757,2014-05-18 06:58:26,20,0,-0.09648467,-0.04426107,0.45049885,8.89139205
    """
    t_pre = ""
    d_pre = ""
    count = 0
    with open(csv_file, "r") as rfile, open(output_file, "w") as wfile:
        for row in rfile:
            row = row.strip().split(",")
            device_id = row[0]
            t = row[1]
            ind = row[2]
            label = row[3]
            imu = row[4]
            count += 1
            if (t != t_pre) and (device_id != d_pre):
                item = f"{device_id},{t},{count},{label},{ind}\n"
                wfile.write(item)
                count = 0
            t_pre = t
            d_pre = device_id


def write_mat_info(mat_file):
    dpath = mat_file.parent
    dd = loadmat(mat_file)["outputStruct"]
    n_data = len(dd["tags"][0][0][0])
    with open(dpath / "mat.csv", "a") as f:
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
            ids = tags[tags[:, 1] == 1][:, 0]
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


"""
dpath = Path("/home/fatemeh/Downloads/bird/data_from_Susanne")
# bd.combine_jsons_to_one_json(list(dpath.glob("json")))
igs, ldts = bd.combine_all_data(dpath/"combined.json")
# write
with open(dpath/"json.csv", 'w') as f:
    for ig, ldt in zip(igs, ldts):
        t = datetime.utcfromtimestamp(ldt[2]).strftime("%Y-%m-%d %H:%M:%S")
        item = f"{ldt[1]},{t},{len(ig)},{ldt[0]}\n"
        f.write(item)
# read
jdata = []
with open(dpath/"json.csv", 'r') as f:
    for r in f:
        r = r.strip().split(',')
        item = [int(r[0]), r[1], int(r[2]), int(r[3])]
        jdata.append(item)    
# json: 1856 (10 items; 0-10 labels)
# [(i, len(np.where(ldts[:,1]==i)[0])) for i in np.unique(ldts[:,1])]
# [(533, 17), (534, 146), (537, 32), (541, 56), (606, 147), (608, 59), (754, 15), (757, 118), (781, 7), (782, 269), (798, 57), 
#  (805, 302), (806, 43), (871, 32), (1600, 54), (6011, 67), (6016, 24), (6073, 39), (6080, 63), (6206, 16), (6208, 113), (6210, 180)]
# [(int(p.stem.split("AnnAcc")[1].split("_")[0]), loadmat(p)["outputStruct"]['annotations'][0][0].shape[0]) for p in dpath.glob("*mat")]
# [(6208, 113), (6073, 73), (6011, 158), (6210, 271), (6016, 52), (1600, 57), (6206, 54), (6080, 141)]

# for mat files
for mat_file in tqdm(dpath.glob("*mat")):
    print(mat_file.name)
    write_mat_info(mat_file)
"""
