from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

import behavior.data as bd
import behavior.utils as bu


def write_j_data_orig(json_file, save_file, new2old_labels, ignored_labels):
    """
    read json data and save it as formatted csv file

    input:  device, time, index, label, imux, imuy, imuz, gps
    e.g. row: 757,2014-05-18 06:58:26,20,0,-0.09648467,-0.04426107,0.45049885,8.89139205
    """

    all_measurements, ldts = bd.load_all_data_from_json(json_file)

    items = []
    for meas, ldt in tqdm(
        zip(all_measurements, ldts), total=len(ldts)
    ):  # N x {10,20} x 4
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
            i = np.round(i, 6)
            item = (
                f"{device_id},{start_time},{-1},{label},{i[0]:.6f},{i[1]:.6f},"
                f"{i[2]:.6f},{i[3]:.6f}\n"
            )
            items.append(item)

    with open(save_file, "w") as f:
        for item in items:
            f.write(item)


def write_m_data_orig(mat_file, save_file, new2old_labels, ignored_labels):
    """
    Becareful: save_file appending contents
    """
    dd = loadmat(mat_file)["outputStruct"]
    n_data = dd["nOfSamples"][0][0][0][0]

    file = open(save_file, "a")
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
        if len(labels) == 0:
            continue

        for label, x, y, z in zip(labels, imu_x, imu_y, imu_z):
            if label in ignored_labels:
                continue
            nlabel = new2old_labels[label]
            if any([np.isnan(x), np.isnan(y), np.isnan(z), np.isnan(gps_single)]):
                continue
            ig = np.round(np.array([x, y, z, gps_single]), 6)
            item = f"{device_id},{t},-1,{nlabel},{ig[0]:.6f},{ig[1]:.6f},{ig[2]:.6f},{ig[3]:.6f}\n"
            file.write(item)
            file.flush()
    file.close()


def write_w_data_orig(csv_file, save_file):
    """
    Becareful: save_file appending contents
    """
    file = open(save_file, "a")
    with open(csv_file, "r") as f:
        for r in f:
            r = r.strip().split(", ")
            device_id = r[0]
            start_time = datetime.strptime(r[1], "%m/%d/%Y %H:%M:%S").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            label, conf = r[-1].split(" ")
            label = int(label) - 1  # zero-based
            if conf == "0":
                continue
            if r[3] == "NaN" or r[4] == "NaN" or r[5] == "NaN" or r[6] == "NaN":
                continue
            item = (
                f"{device_id},{start_time},{r[2]},{label},{r[3]},{r[4]},{r[5]},{r[6]}\n"
            )
            file.write(item)
        file.flush()
    file.close()


def build_index(keys_rounded, n_rows=3):
    """
    Create a dictionary where key = tuple of n_rows consecutive tuples, value = starting index
    """
    index_map = {}
    for i in range(len(keys_rounded) - n_rows + 1):
        key = tuple(tuple(keys_rounded[i + j]) for j in range(n_rows))
        if key in index_map:
            index_map[key].append(i)
        else:
            index_map[key] = [i]
    return index_map


def match_forward_backward(index_maps, keys_rounded, i):
    """
    Try to find the best match at index i by checking forward and backward
    for n=3, 2, 1 in that order. Returns (match_index_in_db, start_row_in_df, n_rows)
    """
    max_n = 3
    L = len(keys_rounded)

    for n_rows in reversed(range(1, max_n + 1)):
        # Forward match
        if i + n_rows <= L:
            key_fwd = tuple(tuple(keys_rounded[i + j]) for j in range(n_rows))
            indices_fwd = index_maps[n_rows - 1].get(key_fwd)
            if indices_fwd and len(indices_fwd) == 1:
                return indices_fwd[0], i, n_rows

        # Backward match
        if i - (n_rows - 1) >= 0:
            key_bwd = tuple(tuple(keys_rounded[i - j]) for j in reversed(range(n_rows)))
            indices_bwd = index_maps[n_rows - 1].get(key_bwd)
            if indices_bwd and len(indices_bwd) == 1:
                return indices_bwd[0], i - (n_rows - 1), n_rows

    return None, None, None


def add_index(df_db, df, save_file):
    # Settings
    tol = 1e-4  # precision 6 is issue in original data
    precision = -int(np.log10(tol))

    # Preprocess keys
    df_db.iloc[:, [4, 5, 6, 7]] = np.round(df_db.iloc[:, [4, 5, 6, 7]], precision)
    keys_rounded = df_db.iloc[:, [0, 1, 4, 5, 6, 7]].values
    index_map1 = build_index(keys_rounded, 1)
    index_map2 = build_index(keys_rounded, 2)
    index_map3 = build_index(keys_rounded, 3)
    index_maps = [index_map1, index_map2, index_map3]

    # Pre-round df once
    df_values = df.copy()
    df_values.iloc[:, [4, 5, 6, 7]] = np.round(df.iloc[:, [4, 5, 6, 7]], precision)
    df_values = df_values.iloc[:, [0, 1, 4, 5, 6, 7]].values

    # Output buffer
    output_lines = []
    ind = 0
    j = 0
    pbar = tqdm(total=len(df))
    while j < len(df):
        sel_ind, start_j, n_row = match_forward_backward(index_maps, df_values, j)
        # sel_ind, n_row = match_best_sequence(index_maps, df_values, j)
        if sel_ind is None:
            j += 1
            pbar.update(1)
            continue  # No match found

        # Match found
        match_offset = j - start_j  # Position of current j in the matched window
        ind = sel_ind + match_offset
        i = df.iloc[j]
        imu_ind = df_db.iloc[ind, 2]
        item = (
            f"{i[0]},{i[1]},{imu_ind},{i[3]},{i[4]:.6f},{i[5]:.6f},"
            f"{i[6]:.6f},{i[7]:.6f}\n"
        )
        output_lines.append(item)

        j += 1
        pbar.update(1)

    # Write all at once
    with open(save_file, "w") as file:
        file.writelines(output_lines)


def map_new_labels(df, new2old_labels, save_file):
    """
    Map new labels to old labels
    """
    df[3] = df[3].map(new2old_labels)
    df.to_csv(save_file, index=False, header=None, float_format="%.6f")


def get_rules():
    """
    Get rules for label selection

    Example:
    rule["Flap"]["ExFlap"] is 2, which mean the second label ExFlap is selected.
    """
    # fmt: off
    labels = [
        "Flap", "ExFlap", "Soar", "Manouvre", "Boat", "Float", "Float_groyne",  "SitStand", "TerLoco",
        "Pecking", "Handling_mussel"
    ]
    ind2name = {0: 'Flap', 1: 'ExFlap', 2: 'Soar', 3: 'Boat', 4: 'Float', 5: 'SitStand', 6: 'TerLoco', 7: 'Other', 8: 'Manouvre', 9: 'Pecking', 
    10: 'Looking_food', 11: 'Handling_mussel', 13: 'StandForage', 14: 'xtraShake', 15: 'xtraCall', 16: 'xtra', 17: 'Float_groyne'}

    discard={7: 'Other', 10: 'Looking_food', 13: 'StandForage', 14: 'xtraShake', 15: 'xtraCall', 16: 'xtra'}
    upper_triangle = np.array([
        [1, 2, 1, 2, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 2, 0, 0, 0, 1, 2, 2, 2],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2],
        [0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ])
    # fmt: on

    rule = upper_triangle + upper_triangle.T - np.diag(upper_triangle.diagonal())
    rule_df = pd.DataFrame(rule, index=labels, columns=labels)

    return ind2name, discard, rule_df


def find_matching_index(keys, query, tol=1e-4):
    """
    find matching two rows
    """
    precision = -int(np.log10(tol))
    keys = np.round(keys, precision)
    query = np.round(query, precision)
    len_keys = len(keys)
    for i in range(0, len_keys - 1):
        cond1 = all(np.isclose(keys[i], query[0], rtol=tol))
        cond2 = all(np.isclose(keys[i + 1], query[1], rtol=tol))
        if cond1 & cond2:
            return i
    return -1


def map_to_nearest_divisible_20(start, end):
    return [int(round(i / 20) * 20) for i in [start, end]]


def get_label_range(slice):
    all_labels = slice[3].squeeze().values
    idxs = np.where(np.diff(all_labels) != 0)[0] + 1  # len=0 or more
    start_idxs = np.concatenate(([0], idxs), axis=0)
    end_idxs = np.concatenate((idxs, [len(all_labels)]), axis=0)
    label_ranges = []
    for idx1, idx2 in zip(start_idxs, end_idxs):
        assert len(np.unique(slice.iloc[idx1:idx2][3])) == 1
        label = slice.iloc[idx1:idx2].iloc[0, 3]
        index_range = [idx1, idx2]
        label_range = [label] + index_range
        label_range = list(map(int, label_range))
        label_ranges.append(label_range)
    return label_ranges


def write_unsorted_data(all_data_file, w_file, save_file, len_labeled_data):
    df = pd.read_csv(all_data_file, header=None)
    df_w = pd.read_csv(w_file, header=None)
    file = open(save_file, "w")

    # Sort by device, time
    df_w = df_w.sort_values(by=[0, 1], ignore_index=True)
    # Unique device, time
    uniq_device_times = df_w[[0, 1]].drop_duplicates(ignore_index=True).values
    for device_id, start_time in tqdm(uniq_device_times):
        # device_id, start_time = 534,"2012-06-02 12:38:01"
        slice_df = df[(df[0] == device_id) & (df[1] == start_time)]
        keys = slice_df[[4, 5, 6, 7]].values
        if len(keys) == 0:
            print("Not in database", device_id, start_time)
            continue
        # Get label ranges
        slice = df_w[(df_w[0] == device_id) & (df_w[1] == start_time)]
        for i in range(0, len(slice), len_labeled_data):
            label = slice.iloc[i, 3]
            st_idx, en_idx = i, i + len_labeled_data
            # Get index: Query on IMU&GPS first two rows of label range
            query = slice.iloc[st_idx : st_idx + 2][[4, 5, 6, 7]].values
            ind = find_matching_index(keys, query)
            if ind == -1:
                print("No matching in database", device_id, start_time, query[0])
                continue
            # Map index range
            st_idx, en_idx = ind, ind + len_labeled_data
            # st_idx, en_idx = map_to_nearest_divisible_20(st_idx, en_idx)
            # Get data from database
            s_slice_df = slice_df.iloc[st_idx:en_idx]
            # Data from database is not complete
            if len(s_slice_df) != len_labeled_data:
                continue
            # Write data
            for _, i in s_slice_df.iterrows():
                item = (
                    f"{device_id},{start_time},{i[2]},{label},{i[4]:.6f},{i[5]:.6f},"
                    f"{i[6]:.6f},{i[7]:.6f}\n"
                )
                file.write(item)
            file.flush()
    file.close()


def write_sorted_data(all_data_file, w_file, save_file, min_thr):
    df = pd.read_csv(all_data_file, header=None)
    df_w = pd.read_csv(w_file, header=None)
    file = open(save_file, "w")

    # Sort by device, time
    df_w = df_w.sort_values(by=[0, 1], ignore_index=True)
    # Unique device, time
    uniq_device_times = df_w[[0, 1]].drop_duplicates(ignore_index=True).values
    for device_id, start_time in tqdm(uniq_device_times):
        slice_df = df[(df[0] == device_id) & (df[1] == start_time)]
        keys = slice_df[[4, 5, 6, 7]].values
        if len(keys) == 0:
            print("Not in database", device_id, start_time)
            continue
        # Get label ranges
        slice = df_w[(df_w[0] == device_id) & (df_w[1] == start_time)]
        label_ranges = get_label_range(slice)
        for label_range in label_ranges:
            label = label_range[0]
            st_idx, en_idx = label_range[1:]
            # Drop data from database
            len_labeled_data = en_idx - st_idx
            if len_labeled_data < min_thr:
                continue
            # Get index: Query on IMU&GPS first two rows of label range
            query = slice.iloc[st_idx : st_idx + 2][[4, 5, 6, 7]].values
            ind = find_matching_index(keys, query)
            if ind == -1:
                print("No matching in database", device_id, start_time, query[0])
                continue
            # Map index range
            st_idx, en_idx = ind, ind + len_labeled_data
            st_idx, en_idx = map_to_nearest_divisible_20(st_idx, en_idx)
            # Get data from database
            s_slice_df = slice_df.iloc[st_idx:en_idx]
            # Data from database is not complete
            if len(s_slice_df) != len_labeled_data:
                continue
            # Write data
            for _, i in s_slice_df.iterrows():
                item = (
                    f"{device_id},{start_time},{i[2]},{label},{i[4]:.6f},{i[5]:.6f},"
                    f"{i[6]:.6f},{i[7]:.6f}\n"
                )
                file.write(item)
            file.flush()
    file.close()
