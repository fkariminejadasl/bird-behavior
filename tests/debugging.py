from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from behavior import data_processing as bdp
from behavior.data import create_balanced_data, save_specific_labels

"""
# Create balanced data
# ====================
# data_file = "/home/fatemeh/Downloads/bird/data/final/orig/s_data_orig_with_index.csv"
# save_path = Path("/home/fatemeh/Downloads/bird/data/final/s_data")
data_file = "/home/fatemeh/Downloads/bird/data/final/s_data_shift.csv"
save_path = Path("/home/fatemeh/Downloads/bird/data/final/s_data_shift")
save_path.mkdir(parents=True, exist_ok=True)
keep_labels = [0, 2, 4, 5, 6, 9]
np.random.seed(123)
for i in range(5):
    df = pd.read_csv(data_file, header=None)
    sel_df = create_balanced_data(df, keep_labels, glen=20)
    sel_df.to_csv(save_path / f"s_data_balanced_{i}.csv", index=False, header=None, float_format="%.6f")
    print(f"Done {i}")
save_specific_labels(data_file, save_path / "s_data_unbalanced.csv", keep_labels)
"""


# TODO
# new2old_labels: old2new
# df = df.sort_values([0, 1]).reset_index(drop=True) = df.sort_values(by=[0,1], ignore_index=True) # maybe for everything.
# check original data preserved.
# check tests


"""
# Debugging: finding mapping
# ======
# First original data only format and then index, sort and for each label
# from the top of the list find the common dates and look for the label in the other data. 

df1 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/s_data_index.csv", header=None)
df2 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/j_data_index.csv", header=None)
df3 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/m_data_index.csv", header=None)
df4 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/w_data_index.csv", header=None)
df2 = df2.sort_values([0,1,2])
df1 = df1.sort_values([0,1,2])
df3 = df3.sort_values([0,1,2])
df4 = df4.sort_values([0,1,2])
# df2[df2[3]==3]

>>> np.unique(df1[3])
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.unique(df2[3])
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
>>> np.unique(df3[3])
array([ 0,  5,  6, 10, 11, 13, 14, 15, 16, 17])

# Discovered mapping
# ======
j:s
5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8
j:m
0: 6, 1: 5, 5: 0, 
8: 10, 9: 11, 10: 13
7: 15
ind2name = {0: 'Flap', 1: 'ExFlap', 2: 'Soar', 3: 'Boat', 4: 'Float', 5: 'SitStand', 6: 'TerLoco', 7: 'Other', 8: 'Manouvre', 9: 'Pecking', 
10: 'Looking_food', 11: 'Handling_mussel', 13: 'StandForage', 14: 'xtraShake', 15: 'xtraCall', 16: 'xtra', 17: 'Float_groyne'}

# j:s
new2old_labels = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 9: 9, 10: 9}
ignored_labels = [8, 14, 15, 16, 17]

# s:m
new2old_labels = {0: 0, 5: 5, 6: 6, 11: 9, 13: 9}
ignored_labels = [10, 14, 15, 16, 17]
"""

"""
# Debugging: Missing keys
# ======
# Since, some labels are removed from both j_data, s_data, previous getting data from database doesn't contain all the data.
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
"""
