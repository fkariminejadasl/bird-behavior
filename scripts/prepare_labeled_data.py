from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behavior import data_processing as bdp
from behavior import utils as bu
from behavior.data_processing import (
    change_format_csv_files,
    change_format_json_files,
    change_format_mat_files,
    make_combined_data_pipeline,
    make_data_pipeline,
)

"""
'''
Get the data from the database

"/home/fatemeh/Downloads/bird/data/final/orig/{}_data_orig.csv" or
first make {}_format.csv (The first operation in the data pipeline).
/home/fatemeh/Downloads/bird/data/final/proc2/{}_format.csv

Since get_s_j_w_m_data_from_database is very slow, we previously downloaded the data from the database. But some keys 
wwere missing, so we need to download them again. Below is the code to download the missing keys.
'''

'''
# Find missing keys
# ======
# Since, some labels are removed from both j_data, m_data, previous getting data from database doesn't contain all the data.
# Here the missing keys are downloaded from database and them manually added to the all_database_final.csv.
m0 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/m_data_orig_no_mapping.csv", header=None)
j0 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/j_data_orig_no_mapping_with_index.csv", header=None)
a0 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final_no_missing_keys.csv", header=None)
mk = list(m0.groupby([0,1]).groups.keys())
jk = list(j0.groupby([0,1]).groups.keys())
ak = list(a0.groupby([0,1]).groups.keys())
a = set(jk).difference(ak)
b = set(mk).difference(ak)
# sum these two since a.difference(b) empty
n = pd.DataFrame(list(a.intersection(b)) + list(b.difference(a)))
n.to_csv("/home/fatemeh/Downloads/bird/data/final/orig/to_be_download.csv", header=None)
database_url = "postgresql://username:password@host:port/database_name"
save_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final_missing_keys.csv")
data = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/missing_keys.csv", header=None)
get_s_j_w_m_data_from_database(data, save_file, database_url, glen=1)
'''

database_url = "postgresql://username:password@host:port/database_name"
save_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv")
df_s = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/s_data_orig.csv", header=None)
df_j = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/j_data_orig.csv", header=None)
df_w = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/w_data_orig.csv", header=None)
df_m = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/m_data_orig.csv", header=None)
data = pd.concat((df_s, df_j, df_w, df_m), axis=0, ignore_index=True)
bdp.get_s_j_w_m_data_from_database(data, save_file, database_url, glen=1)
# e.g. 782,2013-06-07 15:33:49 contains 59 rows in the database. So with glen=1 we get all the data.
# With glen=20, we get 40 rows. # all_database_final.csv glen=1, old: all_database.csv glen=20.
"""

"""
# Prepare data: complete pipeline
change_format = {
    "s": change_format_json_files,
    "j": change_format_json_files,
    "m": change_format_mat_files,
    "w": change_format_csv_files,
}
save_path = Path("/home/fatemeh/Downloads/bird/data/final/proc2")
database_file = Path(
    "/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv"
)
name_input_files = [
    ("s", Path("/home/fatemeh/Downloads/bird/data/set1/data")),
    ("j", Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne")),
    ("m", Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne")),
    ("w", Path("/home/fatemeh/Downloads/bird/data/data_from_Willem")),
]
[
    make_data_pipeline(name, input_file, save_path, database_file, change_format)
    for name, input_file in name_input_files
]
filenames = [f"{i}_complete.csv" for i in ["s", "j", "m", "w"]]
make_combined_data_pipeline(save_path, save_path, filenames)
print("Done")
"""

"""
# Train, valid, test split
seed = 42
np.random.seed(seed)
save_path = Path("/home/fatemeh/Downloads/bird/data/final/proc2")
data_file = save_path / "shift.csv"
bdp.make_train_valid_test_split(data_file, save_path)
"""

"""
# Stats: write the start and end indices. Format: devic,time,label:start-end,...
# test_labels_comes_together: count which classes comes together
for i in ["s", "j", "m", "w"]:  # ["combined"]:
    name = f"{i}_complete"  # "combined"
    df = pd.read_csv(
        f"/home/fatemeh/Downloads/bird/data/final/proc2/{name}.csv", header=None
    )
    save_file = Path(
        f"/home/fatemeh/Downloads/bird/data/final/proc2/{name}_label_range.txt"
    )
    bdp.write_all_start_end_inds(df, save_file)
"""

"""
# Visualize shift data
df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc2/shift.csv", header=None)
save_path = Path("/home/fatemeh/Downloads/bird/result/shift")
glen = 20
dt = 6011, "2015-04-30 09:10:31"
dt = 6210, "2016-05-09 11:09:55"
dt = 6210, "2016-05-09 11:09:15"
dt = 6011, "2015-04-30 09:10:18"
ind2name = bdp.get_rules().ind2name
bu.generate_per_glen_figures_for_dt(save_path, df, dt, ind2name, glen=20)
"""

"""
# Create five random balanced and unbalanced datasets from the given data file and save them to the specified path.

# data_file = "/home/fatemeh/Downloads/bird/data/final/orig/s_data_orig_with_index.csv"
# save_path = Path("/home/fatemeh/Downloads/bird/data/final/s_data")
data_file = "/home/fatemeh/Downloads/bird/data/final/s_data_shift.csv"
save_path = Path("/home/fatemeh/Downloads/bird/data/final/s_data_shift")
create_five_balanced_data_and_unbalanced(data_file, save_path)
"""


def get_stats(df: pd.DataFrame, glen=20):
    # dfs = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc2/s_map.csv", header=None)
    # dfj = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc2/j_map.csv", header=None)
    # dfm = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc2/m_map.csv", header=None)
    # dfw = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc2/w_map.csv", header=None)
    # dts = dfs[[0,1]].drop_duplicates().reset_index(drop=True)
    # dtw = dfw[[0,1]].drop_duplicates().reset_index(drop=True)
    # dtj = dfj[[0,1]].drop_duplicates().reset_index(drop=True)
    # msw = pd.merge(dts, dtw).reset_index(drop=True)
    # msj = pd.merge(dts, dtj).reset_index(drop=True)
    # mwj = pd.merge(dtw, dtj).reset_index(drop=True)
    # mswj = pd.merge(msw, msj).reset_index(drop=True)
    # dfswjm = bdp.merge_prefer_valid(dfs, dfj, dfm, dfw)
    # >>> len(dts), len(dtj), len(dtm), len(dtw)
    # (1526, 1854, 706, 1426)
    # >>> len(dfs), len(dfj), len(dfm), len(dfw)
    # (71842, 90668, 32790, 67807)
    # >>> len(dfswjm)
    # 105692
    save_path = Path("/home/fatemeh/Downloads/bird/data/final/proc2")
    a = [
        pd.read_csv(save_path / f"{i}_map.csv", header=None)
        for i in ["s", "j", "m", "w"]
    ]
    [len(a[i][[0, 1]].drop_duplicates()) for i in range(len(a))]
    [len(a[i]) for i in range(len(a))]
    sorted({k: v // glen for k, v in Counter(df[3].values).items()}.items())
    len(df[[0, 1]].drop_duplicates())
    # train, valid, test: 832020/20, 163940/20, 166520/20


# >>> df1 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc2/s_map0.csv", header=None)
# >>> df2 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc2/w_map0.csv", header=None)
# >>> dts1 = set(df1.groupby([0,1]).groups.keys())
# >>> dts2 = set(df2.groupby([0,1]).groups.keys())
# dts1.difference(dts2)
# >>> len(dts1), len(dts2), len(df1), len(df2)
# (1536, 1439, 70100, 67689)

# path = Path("/home/fatemeh/Downloads/bird/data/final/proc2")
# for n in ["s","j","m","w"]:
#     df = pd.read_csv(path/f"{n}_map0.csv",header=None)
#     df = df.sort_values([0,1]).reset_index(drop=True)
#     print(f"{n}: {len(df)}, {np.unique(df[0])}")

# for n in ["s"]:#,"j","m","w"]:
#     df = pd.read_csv(path/f"{n}_map0.csv",header=None)
#     df = df.sort_values([0,1]).reset_index(drop=True)
#     groups = df.groupby([0])
#     for name, group in groups:
#         print(f"{name[0]}_{min(group[1])}_{max(group[1])}")
