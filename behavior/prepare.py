from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
combined_file = data_path / "tmp.json" # "combined.json"

all_measurements, ldts = bd.combine_all_data(combined_file)

save_file = Path("/home/fatemeh/Downloads/bird/result/failed/exp1/tmp.csv")
database_url = "postgresql://username:password@host:port/database_name"

for meas, ldt in tqdm(zip(all_measurements, ldts)): # N x 20 x 4
    label = ldt[0]
    device_id = ldt[1]
    timestamp = ldt[2]
    start_time = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    igs, idts, _ = bd.get_data(database_url, device_id, start_time, start_time)
    ind = find_matching_index(igs[:,0:3], meas[0,:3], 1)
    write_as_csv(save_file, device_id, start_time, ind, label, meas)


# repeate for all datasets and combine them.
"""
