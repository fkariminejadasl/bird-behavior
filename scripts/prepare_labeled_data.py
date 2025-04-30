from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from behavior import data_processing as bdp
from behavior.data_processing import (
    change_format_csv_files,
    change_format_json_files,
    change_format_mat_files,
    make_combined_data_pipeline,
    make_data_pipeline,
)

"""
Get the data from the database

"/home/fatemeh/Downloads/bird/data/final/orig/{}_data_orig.csv" or
first make {}_format.csv (The first operation in the data pipeline).
/home/fatemeh/Downloads/bird/data/final/proc2/{}_format.csv
"""
# database_url = "postgresql://username:password@host:port/database_name"
# save_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv")
# df_s = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/s_data_orig.csv", header=None)
# df_j = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/j_data_orig.csv", header=None)
# df_w = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/w_data_orig.csv", header=None)
# df_m = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/m_data_orig.csv", header=None)
# data = pd.concat((df_s, df_j, df_w, df_m), axis=0, ignore_index=True)
# get_s_j_w_m_data_from_database(data, save_file, database_url, glen=1)
# # e.g. 782,2013-06-07 15:33:49 contains 59 rows in the database. So with glen=1 we get all the data.
# # With glen=20, we get 40 rows. # all_database_final.csv glen=1, old: all_database.csv glen=20.


# seed = 42
# np.random.seed(seed)
# save_path = Path("/home/fatemeh/Downloads/bird/data/final/proc2")
# data_file = save_path / "shift.csv"
# make_train_valid_test_split(data_file, save_path)

""""
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
