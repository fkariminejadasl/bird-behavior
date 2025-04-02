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


def round_array(arr, precision):
    return np.round(arr, precision)


def build_index(keys_rounded):
    """
    Create a dictionary where key = (row1_tuple, row2_tuple), value = index
    """
    index_map = {}
    for i in range(len(keys_rounded) - 2):
        k1 = tuple(keys_rounded[i])
        k2 = tuple(keys_rounded[i + 1])
        k3 = tuple(keys_rounded[i + 2])
        index_map[(k1, k2, k3)] = i
    return index_map


def add_index(df_db, df, save_file):
    # Settings
    tol = 1e-4  # precision 6 is issue in original data
    precision = -int(np.log10(tol))

    # Preprocess keys
    # keys = df_db.iloc[:, [4, 5, 6, 7]].values
    # keys_rounded = round_array(keys, precision)
    df_db.iloc[:, [4, 5, 6, 7]] = np.round(df_db.iloc[:, [4, 5, 6, 7]], precision)
    keys_rounded = df_db.iloc[:, [0, 1, 4, 5, 6, 7]].values
    index_map = build_index(keys_rounded)

    # Pre-round df once
    # df_values = round_array(df.iloc[:, [4, 5, 6, 7]].values, precision)
    df_values = df.copy()
    df_values.iloc[:, [4, 5, 6, 7]] = np.round(df.iloc[:, [4, 5, 6, 7]], precision)
    df_values = df_values.iloc[:, [0, 1, 4, 5, 6, 7]].values

    # Output buffer
    output_lines = []
    ind = 0
    last_ind = 0
    j = 0
    pbar = tqdm(total=len(df))
    while j < len(df):
        if j == len(df) - 2:
            sel_ind = -1
        else:
            q1 = tuple(df_values[j])
            q2 = tuple(df_values[j + 1])
            q3 = tuple(df_values[j + 2])
            sel_ind = index_map.get((q1, q2, q3), -1)

        if sel_ind == -1:
            inds = [last_ind + 1, last_ind + 2]
            js = [j, j + 1]
            last_ind = last_ind + 2
            j = j + 2
            pbar.update(2)
        else:
            inds = [sel_ind]
            js = [j]
            last_ind = sel_ind
            j = j + 1
            pbar.update(1)
        for jj, ind in zip(js, inds):
            i = df.iloc[jj]
            imu_ind = df_db.iloc[ind, 2]  # if ind != -1 else -1
            item = (
                f"{i[0]},{i[1]},{imu_ind},{i[3]},{i[4]:.6f},{i[5]:.6f},"
                f"{i[6]:.6f},{i[7]:.6f}\n"
            )
            output_lines.append(item)

    # Write all at once
    with open(save_file, "w") as file:
        file.writelines(output_lines)


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
