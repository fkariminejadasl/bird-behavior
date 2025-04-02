from pathlib import Path

import numpy as np
import pandas as pd

from behavior.data import create_balanced_data, save_specific_labels
from behavior.data_processing import add_index, write_j_data_orig, write_unsorted_data
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
# Debugging
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


"""
# Debugging
# ======
dpath = Path("/home/fatemeh/Downloads/bird/data/set1/data")
json_file = Path("/home/fatemeh/Downloads/bird/data/set1/data/combined.json")
save_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/j_data_orig2.csv")
# bd.combine_jsons_to_one_json(list(dpath.glob("*json")), json_file)
write_j_data_orig(json_file, save_file, new2old_labels = {k:k for k in range(11)}, ignored_labels=[])

all_data_file = "/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv"
orig_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/j_data_orig2.csv")
save_file = Path("/home/fatemeh/Downloads/bird/data/final/j_data2.csv")
write_unsorted_data(all_data_file, orig_file, save_file, 10)

df_db = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv", header=None)
df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/j_data_orig2.csv", header=None)
save_file = "/home/fatemeh/Downloads/bird/data/final/orig/j_data_orig_with_index2.csv"
add_index(df_db, df, save_file)
"""


"""
# Debugging: finding mapping
# ======
# For debugging: First original j_data without mapping only json to csv (j_data_orig_no_mapping). 
# Then add index to the data (j_data_orig_no_mapping_with_index)
# Then check the mapping values. label 9, 10 seems wrong

# J_data (Json Suzzane)
dpath = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne")
json_file = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne/combined.json")
save_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/j_data_orig_no_mapping.csv")
new2old_labels = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 9: 9, 10: 9}
ignored_labels = [8, 14, 15, 16, 17]
write_j_data_orig(json_file, save_file, new2old_labels = {k:k for k in range(11)}, ignored_labels=[])

df_db = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv", header=None)
df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/j_data_orig_no_mapping.csv", header=None)
save_file = "/home/fatemeh/Downloads/bird/data/final/orig/j_data_orig_no_mapping_with_index.csv"
add_index(df_db, df, save_file)

s0 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/s_data_orig_with_index.csv", header=None)
j0 = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/j_data_orig_no_mapping_with_index.csv", header=None)
j0=j0.sort_values([0,1])
s0=s0.sort_values([0,1])
j0[j0[3]==3]
"""

"""
dpath = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne")
save_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/m_data_orig_no_mapping.csv")
new2old_labels = {0: 0, 5: 5, 6: 6, 11: 9, 13: 9}
ignored_labels = [10, 14, 15, 16, 17]
# for mat_file in tqdm(dpath.glob("An*mat")):
#     print(mat_file.name)
#     write_m_data_orig(mat_file, save_file, new2old_labels, ignored_labels=[])

df_db = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv", header=None)
df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/orig/m_data_orig_no_mapping.csv", header=None)
save_file = "/home/fatemeh/Downloads/bird/data/final/orig/m_data_orig_no_mapping_with_index.csv"
add_index(df_db, df, save_file)
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
