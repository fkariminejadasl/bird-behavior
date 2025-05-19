from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from behavior import data_processing as bdp
from behavior import utils as bu

# combined.csv -> gt2_combined
save_path = Path("/home/fatemeh/Downloads/bird/result")
data_file = Path("/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv")
database_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv")

df = pd.read_csv(data_file, header=None) # combined.csv
df_db = pd.read_csv(database_file, header=None)
ind2name = bdp.get_rules().ind2name
glen = 10  # 20

# # Since balance.csv labels are 0, 1, 2, 3, 4, 5 and orig labels are 0, 2, 4, 5, 6, 9
# mapping = {0: 0, 1: 2, 2: 4, 3: 5, 4: 6, 5: 9}
# df[3] = df[3].map(mapping)

save_path = save_path / f"gt2_{data_file.stem}" 
save_all_path = save_path / "all"
save_all_path.mkdir(parents=True, exist_ok=True)

# Unique device and starting times
df_db = df_db.sort_values([0, 1, 2])
df = df.sort_values([0, 1, 2])
# df = df[df[3] != 7]
unique_dt = df.groupby(by=[0, 1])
for i, dt in tqdm(enumerate(unique_dt.groups.keys()), total=len(unique_dt)):
    # all_keys = list(unique_dt.groups.keys())[1354:]
    # for i, dt in tqdm(enumerate(all_keys), total=len(all_keys)):
    dataframe = unique_dt.get_group(dt)
    dataframe_db = df_db[(df_db[0] == dt[0]) & (df_db[1] == dt[1])]
    # print(dt[0], dt[1], len(dataframe))
    # fig = bu.plot_all(dataframe, dataframe_db, glen=glen)
    fig = bu.plot_labeled_data(dataframe, dataframe_db, ind2name)
    name = f"{i},{dt[0]},{dt[1]}"
    # Take the first label (it can also be max or any other thing)
    lable_ids = dataframe.iloc[:, 3].values
    lable_ids = [int(i) for i in lable_ids]
    label_id = next(i for i in lable_ids if i != -1)
    label = ind2name[label_id]
    save_label_path = save_path / f"{label}"
    save_label_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_label_path / f"{name}.png", bbox_inches="tight")
    fig.savefig(save_all_path / f"{name}.png", bbox_inches="tight")
    plt.close()

# To test the for loop for case with 20 elements labeled and 40 in db
# [[0, (782,"2014-05-26 11:03:55")]]:
