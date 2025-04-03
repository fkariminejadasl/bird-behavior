from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from behavior.data import create_balanced_data, save_specific_labels
from behavior.data_processing import (
    add_index,
    map_new_labels,
    write_j_data_orig,
    write_m_data_orig,
    write_unsorted_data,
    write_w_data_orig,
)
from behavior.utils import ind2name

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

"""
# Debugging s_data
# ======
# dpath = Path("/home/fatemeh/Downloads/bird/data/set1/data")
# json_file = Path("/home/fatemeh/Downloads/bird/data/set1/data/combined.json")
# save_file = Path("/home/fatemeh/Downloads/bird/data/final/proc/s_data_format.csv")
# # bd.combine_jsons_to_one_json(list(dpath.glob("*json")), json_file)
# write_j_data_orig(
#     json_file, save_file, new2old_labels={k: k for k in ind2name}, ignored_labels=[]
# )

# all_data_file = "/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv"
# orig_file = Path("/home/fatemeh/Downloads/bird/data/final/proc/s_data_format.csv")
# save_file = Path("/home/fatemeh/Downloads/bird/data/final/proc/s_data_no_shift.csv")
# write_unsorted_data(all_data_file, orig_file, save_file, 20)

df_db = pd.read_csv(
    "/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv", header=None
)
df = pd.read_csv(
    "/home/fatemeh/Downloads/bird/data/final/proc/s_data_format.csv", header=None
)
save_file = "/home/fatemeh/Downloads/bird/data/final/proc/s_data_index.csv"
add_index(df_db, df, save_file)
"""

# # correct the code for add_index for missing items
# # remove the to be removed
# # visualize
# # label mapping
# >>> df3 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/j_data_no_shift.csv", header=None)
# >>> df1 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/s_data_index.csv", header=None)
# >>> df2 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/j_data_index.csv", header=None)
# >>> df2.iloc[:,4:] = np.round(df2.iloc[:,4:], 4)
# >>> df3.iloc[:,4:] = np.round(df3.iloc[:,4:], 4)
# >>> df2 = df2.sort_values([0,1,2])
# >>> df1 = df1.sort_values([0,1,2])
# >>> df3 = df3.sort_values([0,1,2])
# >>> common = pd.merge(df2,df3)
# >>> pd.concat([df3, common]).drop_duplicates(keep=False) # 534  2012-06-08 08:32:46 issue in new code


"""
# Debugging: j_data (Suzzane)
# ======
# dpath = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne")
# json_file = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne/combined.json")
# save_file = Path("/home/fatemeh/Downloads/bird/data/final/proc/j_data_format.csv")
# new2old_labels = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 9: 9, 10: 9}
# ignored_labels = [8, 14, 15, 16, 17]
# # bd.combine_jsons_to_one_json(list(dpath.glob("*json")), json_file)
# write_j_data_orig(json_file, save_file, new2old_labels = {k:k for k in range(30)}, ignored_labels=[])

# all_data_file = "/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv"
# orig_file = Path("/home/fatemeh/Downloads/bird/data/final/proc/j_data_format.csv")
# save_file = Path("/home/fatemeh/Downloads/bird/data/final/proc/j_data_no_shift.csv")
# write_unsorted_data(all_data_file, orig_file, save_file, 10)

# df_db = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv", header=None)
# df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/j_data_format.csv", header=None)
# save_file = "/home/fatemeh/Downloads/bird/data/final/proc/j_data_index.csv"
# add_index(df_db, df, save_file)

df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/j_data_index.csv", header=None)
save_file = "/home/fatemeh/Downloads/bird/data/final/proc/j_data_map0.csv"
new2old_labels = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 8: 10, 9: 11, 10: 13}
map_new_labels(df, new2old_labels, save_file)
"""

"""
# Debugging: M_data (Matlab Suzzane)
# ======
dpath = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne")
save_file = Path("/home/fatemeh/Downloads/bird/data/final/proc/m_data_format.csv")
new2old_labels = {0: 0, 5: 5, 6: 6, 11: 9, 13: 9}
ignored_labels = [10, 14, 15, 16, 17]
for mat_file in tqdm(dpath.glob("An*mat")):
    print(mat_file.name)
    write_m_data_orig(mat_file, save_file, new2old_labels = {k:k for k in range(30)}, ignored_labels=[])

df_db = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv", header=None)
df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/m_data_format.csv", header=None)
save_file = "/home/fatemeh/Downloads/bird/data/final/proc/m_data_index.csv"
add_index(df_db, df, save_file)
"""

""""
# Debugging: w_data (willem)
# ======
dpath = Path("/home/fatemeh/Downloads/bird/data/data_from_Willem")
save_file = Path("/home/fatemeh/Downloads/bird/data/final/proc/w_data_format.csv")
for p in dpath.glob("*csv"):
    print(p.name)
    write_w_data_orig(p, save_file)

df_db = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv", header=None)
df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/w_data_format.csv", header=None)
save_file = "/home/fatemeh/Downloads/bird/data/final/proc/w_data_index.csv"
add_index(df_db, df, save_file)
"""

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

# df2 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/j_data_index.csv", header=None)
# df3 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/m_data_index.csv", header=None)
# df3_cands = df3[df3[3]==17]
# for _, r3 in df3_cands.iterrows():
#     r2 = df2[(df2[0]==r3[0]) & (df2[1]==r3[1]) & (df2[2]==r3[2])]
#     if len(r2) !=0:
#         print(r3, r2)
#         break

# s:w 5: 9 and 9: 2 mistakes
# ts = np.unique(df4[df4[3]==5][1])
# for t in ts:
#     sel = df1[df1[1]==t]
#     if len(sel) != 0:
#         label = sel.iloc[0,3]
#         print(t, label)

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
