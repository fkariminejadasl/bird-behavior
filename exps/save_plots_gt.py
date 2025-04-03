from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from behavior import utils as bu
from behavior.utils import ind2name

# gt2_combined, gt2_sdata
gt_path = Path("/home/fatemeh/Downloads/bird/result/gt2_j_data_index")
# combined_unique_sorted012, s_data_orig_index
df = pd.read_csv(
    "/home/fatemeh/Downloads/bird/data/final/proc/j_data_index.csv",
    header=None,
)
df_db = pd.read_csv(
    "/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv", header=None
)
glen = 10  # 20

# # Since balance.csv labels are 0, 1, 2, 3, 4, 5 and orig labels are 0, 2, 4, 5, 6, 9
# mapping = {0: 0, 1: 2, 2: 4, 3: 5, 4: 6, 5: 9}
# df[3] = df[3].map(mapping)

# Unique device and starting times
save_all_path = gt_path / "all"
save_all_path.mkdir(parents=True, exist_ok=True)
df = df[df[3] != 7]
unique_dt = df.groupby(by=[0, 1])
for i, dt in tqdm(enumerate(unique_dt.groups.keys()), total=len(unique_dt)):
    dataframe = unique_dt.get_group(dt).sort_values(by=[2])
    dataframe_db = df_db[(df_db[0] == dt[0]) & (df_db[1] == dt[1])].sort_values(by=[2])
    # print(dt[0], dt[1], len(dataframe))
    fig = bu.plot_all(dataframe, dataframe_db, glen=glen)
    name = f"{i},{dt[0]},{dt[1]}"
    # Take the first label (it can also be max or any other thing)
    label = ind2name[dataframe.iloc[0, 3]]
    save_path = gt_path / f"{label}"
    save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path / f"{name}.png", bbox_inches="tight")
    fig.savefig(save_all_path / f"{name}.png", bbox_inches="tight")
    plt.close()

# To test the for loop for case with 20 elements labeled and 40 in db
# [[0, (782,"2014-05-26 11:03:55")]]:
