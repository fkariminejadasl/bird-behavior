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
from behavior.data_processing import get_label_range, map_to_nearest_divisible_20
from behavior.utils import ind2name

# j: json (set1, sus json)
# m: matlab (sus mat)
# w: csv (willem)


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


"""
# Step1: Unify all data in csv format
# ======

# s_data (json set1)
# ====================
# Code write_j_data_orig is changed to change_format_json_file and it is not changing labels anymore.
dpath = Path("/home/fatemeh/Downloads/bird/data/set1/data")
json_file = Path("/home/fatemeh/Downloads/bird/data/set1/data/combined.json")
save_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/s_data_orig.csv")
bd.combine_jsons_to_one_json(list(dpath.glob("*json")), json_file)
change_format_json_file(json_file, save_file, new2old_labels = {k:k for k in ind2name}, ignored_labels=[])

# j_data (Json Suzzane)
# ====================
# Code write_j_data_orig is changed to change_format_json_file and it is not changing labels anymore.
dpath = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne")
bd.combine_jsons_to_one_json(list(dpath.glob("*json")), json_file)
json_file = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne/combined.json")
save_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/j_data_orig.csv")
new2old_labels = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 9: 9, 10: 9}
ignored_labels = [8, 14, 15, 16, 17]
change_format_json_file(json_file, save_file, new2old_labels, ignored_labels)

# m_data (matlab Suzzane)
# ====================
# Code write_m_data_orig change to change_format_mat_file. There was a bug in labels. 
# The labels were shifted. Now the function only does change format and saving 
# is done outside. The mapping labels are done outside as well.
dpath = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne")
save_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/m_data_orig.csv")
new2old_labels = {0: 0, 5: 5, 6: 6, 11: 9, 13: 9}
ignored_labels = [10, 14, 15, 16, 17]
for mat_file in tqdm(dpath.glob("An*mat")):
    print(mat_file.name)
    write_m_data_orig(mat_file, save_file, new2old_labels, ignored_labels)

# w_data (csv Willem)
# =======================
dpath = Path("/home/fatemeh/Downloads/bird/data/data_from_Willem")
save_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/w_data_orig.csv")
for p in dpath.glob("*csv"):
    print(p.name)
    change_format_csv_file(p, save_file)

# Step 2. Get all the data from database
==================
database_url = "postgresql://username:password@host:port/database_name"
save_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv")
df_s = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/s_data_orig.csv", header=None)
df_j = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/j_data_orig.csv", header=None)
df_w = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/w_data_orig.csv", header=None)
df_m = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/m_data_orig.csv", header=None)
data = pd.concat((df_s, df_j, df_w, df_m), axis=0, ignore_index=True)
get_s_j_w_m_data_from_database(data, save_file, database_url, glen=1)
# e.g. 782,2013-06-07 15:33:49 contains 59 rows in the database. So with glen=1 we get all the data. 
# With glen=20, we get 40 rows. # all_database_final.csv glen=1, old: all_database.csv glen=20.

# Step 3. Add index to the data
==================
df_db = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv", header=None)
df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/w_data_orig.csv", header=None)
save_file = "/home/fatemeh/Downloads/bird/data/final/orig/w_data_orig_with_index.csv"
add_index(df_db, df, save_file)

# Step3: get indices data
==================
# This part has an small issue with map_to_nearest_divisible_20. The numpy and python round function, map the
# border value such as .5, 1.5, 2.5, 3.5 and so on to the nearest even number 0., 2., 2., 4..
# So border values such as 10, 30, 50, 70, 90 map differently 0, 40, 40, 80, 80.
# 2. There is small issue in the getting data from database. There was few cases where two consecutive rows 
# were equal. The current version only compare two rows. This issue properly resolved in the add_index. 
all_data_file = "/home/fatemeh/Downloads/bird/data/final/orig/all_database.csv"
orig_path = Path("/home/fatemeh/Downloads/bird/data/final/orig")
save_path = Path("/home/fatemeh/Downloads/bird/data/final")
file_names = ["s_data", "j_data"]
glens = [20, 10]
for name, glen in zip(file_names, glens):
    orig_file = orig_path / f"{name}_orig.csv"
    save_file = save_path / f"{name}.csv"
    print(orig_file)
    write_unsorted_data(all_data_file, orig_file, save_file, glen)

all_data_file = "/home/fatemeh/Downloads/bird/data/final/orig/all_database.csv"
orig_path = Path("/home/fatemeh/Downloads/bird/data/final/orig")
save_path = Path("/home/fatemeh/Downloads/bird/data/final")
file_names = ["m_data", "w_data"]
for name in file_names:
    orig_file = orig_path / f"{name}_orig.csv"
    save_file = save_path / f"{name}.csv"
    print(orig_file)
    write_sorted_data(all_data_file, orig_file, save_file, min_thr=10)


# Step4: Combine data
# =====
save_path = Path("/home/fatemeh/Downloads/bird/data/final")
combined_name = "combined.csv"
csvs_to_combine = ["s_data.csv", "w_data.csv", "m_data.csv", "j_data.csv"]
csv_files = [save_path / i for i in csvs_to_combine]
df_list = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file, header=None)
    df[[4, 5, 6, 7]] = df[[4, 5, 6, 7]].round(6)
    df_list.append(df)
combined_df = pd.concat(df_list, ignore_index=True)
combined_df.to_csv(save_path / combined_name, index=False, header=None)


# Step5: Find duplicates
# ===============
subset = [0, 1, 4, 5, 6, 7]
save_path = Path("/home/fatemeh/Downloads/bird/data/final")
df = pd.read_csv(save_path / combined_name, header=None)
df_20 = df.iloc[::20]
sel_df_20 = df_20[subset]
inds = sel_df_20[sel_df_20.duplicated(keep=False)].index
print(len(inds))

# groups = group_equal_elements(df, subset, inds, equal_func)
groups = group_equal_elements_optimized(df, subset, inds)
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

save_name = combined_name[:-4] + "_unique.csv"
df.to_csv(
    save_path / save_name, index=False, header=None, float_format="%.6f"
)
"""


"""
# Analyses and stats
# ====================
# group_sizes = a.groupby([0, 1, 2]).size()
# groups_with_size_2 = group_sizes[group_sizes >2]
# df_s[df_s.duplicated(df_s)].index

# a = pd.concat((df_w, df_j), axis=0, ignore_index=True)
# b = a[a.groupby([0, 1, 2, 3, 4, 5, 6, 7]).transform('size') ==2]
# b = b.sort_values(by=[0, 1, 2], ascending=[True, True, True])
# b.to_csv("/home/fatemeh/Downloads/bird/data/final/dup_j_s.csv", index=False, header=None, float_format="%.6f")
# 533,2012-05-15 03:38:59

# Debug: discover duplicates
# >>> b = df_c[df_c.duplicated()].sort_values(by=[0,1,2]).groupby([0,1]).size().reset_index(name='s')
# >>> b.to_csv("/home/fatemeh/Downloads/bird/data/final/combine_dup_size.csv", index=False, header=None, float_format="%.6f")
# >>> b = df_c[df_c.duplicated()].sort_values(by=[0,1,2])
# >>> b.to_csv("/home/fatemeh/Downloads/bird/data/final/combine_dup.csv", index=False, header=None, float_format="%.6f")
# # per data or a = pd.concat((df_w, df_j), axis=0, ignore_index=True)
#  df_j[df_j[[0,1,2,4,5,6,7]].duplicated()].sort_values(by=[0,1,2])
# Debug: start from middle of burst
# c = df_j[::20][2].values
# c[np.where(c%20!=0)[0]]


>>> df_s = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/s_data.csv", header=None)
>>> df_j = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/j_data.csv", header=None)
>>> df_w = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/w_data.csv", header=None)
>>> df_m = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/m_data.csv", header=None)
>>> df_c = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/combined_unique.csv", header=None)
>>> df_s[df_s.duplicated()]
>>> for i in [df_s, df_j, df_m, df_w, df_c]:
        print(i[i.duplicated()])
>>> group_sizes = df_c.groupby([0,1,2]).size()
>>> group_sizes[group_sizes>1]
>>> for i in [df_s, df_j, df_m, df_w, df_c]:
...     cut_i = i[i[3]!=7].copy()
...     print(len(cut_i)//20)
...     print(len(i)//20)
>>> dict([(i, len(df_c[df_c[3]==i])//20) for i in range(10)])
s_data: {0: 633, 1: 38, 2: 500, 3: 176, 4: 558, 5: 894,  6: 318, 7: 25, 8: 151, 9: 210}
j_data: {0: 216, 1: 19, 2: 146, 3: 0,   4: 460, 5: 375,  6: 127, 7: 10, 8: 47,  9: 66}
m_data: {0: 5,   1: 0,  2: 0,   3: 0,   4: 0,   5: 642,  6: 23,  7: 0,  8: 0,   9: 187}
w_data: {0: 652, 1: 45, 2: 504, 3: 176, 4: 558, 5: 806,  6: 304, 7: 30, 8: 143, 9: 153}
c_data: {0: 655, 1: 42, 2: 549, 3: 176, 4: 823, 5: 1544, 6: 359, 7: 33, 8: 154, 9: 411}

3503, 1466, 857, 3371, 4746 # all
3478, 1456, 857, 3341, 4713 # no 7 (other)
"""

"""
There are checked: by file, by sizes and by diff
>>> b = df_m[df_m.duplicated()].sort_values(by=[0,1,2])
>>> b.to_csv("/home/fatemeh/Downloads/bird/data/final/dup_m.csv", index=False, header=None, float_format="%.6f")
>>> df_m[df_m.duplicated()].sort_values(by=[0,1,2]).groupby([0,1]).size()
>>> np.diff(np.array(df_m[df_m.duplicated()].index.tolist()))
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


def test_find_matching_index():
    df = pd.read_csv(
        Path(__file__).parent.parent / "data/data_from_db.csv", header=None
    )
    keys = df[(df[0] == 533) & (df[1] == "2012-05-15 05:41:52")][[4, 5, 6, 7]].values
    query = np.array(
        [
            [0.225012, -0.433472, 1.318443, 9.072514],
            [0.281927, 1.049555, 0.661937, 9.072514],
        ]
    )
    assert find_matching_index(keys, query) == 19


def write_j_data_orig(json_file, save_file, new2old_labels, ignored_labels):
    """
    read json data and save it as formatted csv file

    input:  device, time, index, label, imux, imuy, imuz, gps
    e.g. row: 757,2014-05-18 06:58:26,20,0,-0.09648467,-0.04426107,0.45049885,8.89139205
    """

    all_measurements, ldts = bd.load_all_data_from_json(json_file)

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
            st_idx, en_idx = map_to_nearest_divisible_20(st_idx, en_idx)
            # Get data from database
            s_slice_df = slice_df.iloc[st_idx:en_idx]
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
            # Write data
            for _, i in s_slice_df.iterrows():
                item = (
                    f"{device_id},{start_time},{i[2]},{label},{i[4]:.6f},{i[5]:.6f},"
                    f"{i[6]:.6f},{i[7]:.6f}\n"
                )
                file.write(item)
            file.flush()
    file.close()
