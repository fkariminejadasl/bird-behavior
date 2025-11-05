from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from omegaconf import OmegaConf
from tqdm import tqdm

from behavior import model_utils as bmu


def write_parquet():
    # Read CSV with numeric headers
    df = pd.read_csv("/home/fatemeh/Downloads/bird/data/ssl/5390_7.csv", header=None)
    df.to_parquet("/home/fatemeh/Downloads/bird/data/ssl/5390_7.parquet", index=False)
    dfp = pd.read_parquet("/home/fatemeh/Downloads/bird/data/ssl/5390_7.parquet")
    # columns by default Index(['0', '1', '2', '3', ...], dtype='object')
    dfp.columns = dfp.columns.astype(int)

    # Group every 20 rows together
    df["group"] = df.index // 20

    # Flatten each group of 20 rows into a single wide row
    flat = (
        df.groupby([0, 1, 3, "group"])
        .apply(lambda g: g[[2, 4, 5, 6, 7]].to_numpy().flatten())
        .reset_index(name="flat")
    )

    # Expand the flattened arrays into columns
    wide = pd.DataFrame(flat["flat"].to_list())
    wide.columns = [f"v{i}" for i in range(wide.shape[1])]
    wide.insert(0, 3, flat[3])  # category
    wide.insert(0, 1, flat[1])  # date_time
    wide.insert(0, 0, flat[0])  # device_id
    wide.columns = range(wide.shape[1])
    int_cols = list(range(3, wide.shape[1], 5))  # 3, 8, 13, ...
    wide[int_cols] = wide[int_cols].astype("int16")
    wide.iloc[:, 3::5] = wide.iloc[:, 3::5].astype("int32")

    # Save as Parquet
    wide.to_csv(
        "/home/fatemeh/Downloads/bird/data/ssl/5390_7_row.csv",
        index=False,
        header=False,
        float_format="%.8f",
    )

    arr_device = pa.array(flat[0])
    arr_dt = pa.array(flat[1])
    arr_cat = pa.array(flat[3])
    list_arr = pa.array(flat["flat"].to_list(), type=pa.list_(pa.float64()))
    table = pa.Table.from_arrays(
        [arr_device, arr_dt, arr_cat, list_arr],
        names=["device_id", "date_time", "category", "gimu"],
    )
    pq.write_table(table, "/home/fatemeh/Downloads/bird/data/ssl/5390_7_row.parquet")
    # b = pq.read_table("/home/fatemeh/Downloads/bird/data/ssl/5390_7_row.parquet").to_pandas()
    b = pd.read_parquet("/home/fatemeh/Downloads/bird/data/ssl/5390_7_row.parquet")
    c3 = b.loc[0, "gimu"].reshape(-1, 5)[:, 1:]


def write_only_gimu_float32_norm_gps(csv_file, parquet_file):
    """
    save only gimu (4,5,6,7) in float32 and normalize gps speed
    """
    df = pd.read_csv(csv_file, header=None)

    # Normalize GPS speed by gps_scale
    gps_scale = 22.3012351755624
    df[7] = df[7] / gps_scale

    # Extract only gimu columns
    gimus = df[[4, 5, 6, 7]].to_numpy(dtype=np.float32)
    # Reshape into (num_chunks, 20 * 4)
    n_full_chunks = gimus.shape[0] // 20
    gimus_flat = gimus.reshape(n_full_chunks, 20 * 4)

    # Convert to Arrow array of float32 lists
    list_arr = pa.array(gimus_flat.tolist(), type=pa.list_(pa.float32()))
    table = pa.Table.from_arrays([list_arr], names=["gimu"])
    # df.to_parquet generates larger file compare to pq.write_table
    pq.write_table(table, parquet_file)

    # a = flat["flat"].apply(lambda x: ','.join(str(i) for i in x)) # with ""
    # a = flat["flat"].apply(lambda x: " ".join(f"{v:.8f}" for v in x))
    # a.to_csv("/home/fatemeh/Downloads/bird/data/ssl/5390_7_only_gimu.csv", index=False, header=False)
    # # df.to_parquet generates larger file compare to pq.write_table
    # pd.DataFrame(a).to_parquet("/home/fatemeh/Downloads/bird/data/ssl/5390_7_only_gimu.parquet", index=False)
    # b = pd.read_csv("/home/fatemeh/Downloads/bird/data/ssl/5390_7_only_gimu.csv", header=None)
    # c1 = np.array([np.float64(i) for i in b.iloc[0,0].split()]).reshape(-1,4)
    # b = pd.read_parquet("/home/fatemeh/Downloads/bird/data/ssl/5390_7_only_gimu.parquet")
    # c3 = b.iloc[0,0].reshape(-1,4)


def write_only_gimu_float32_norm_gps_batch(csv_files, parquet_file):
    """
    save only gimu (4,5,6,7) in float32 and normalize gps speed
    """
    count = 0
    arr = pa.array([], type=pa.list_(pa.float32()))
    for csv_file in tqdm(csv_files):
        df = pd.read_csv(csv_file, header=None)

        # Normalize GPS speed by gps_scale
        gps_scale = 22.3012351755624
        df[7] = df[7] / gps_scale

        # Extract only gimu columns
        gimus = df[[4, 5, 6, 7]].to_numpy(dtype=np.float32)
        # Reshape into (num_chunks, 20 * 4)
        n_full_chunks = gimus.shape[0] // 20
        gimus_flat = gimus.reshape(n_full_chunks, 20 * 4)

        # Convert to Arrow array of float32 lists
        new_arr = pa.array(gimus_flat.tolist(), type=pa.list_(pa.float32()))
        arr = pa.concat_arrays((arr, new_arr))
        print(len(new_arr))
        count += len(new_arr)
    print(len(arr), count)
    table = pa.Table.from_arrays([arr], names=["gimu"])
    # df.to_parquet generates larger file compare to pq.write_table
    pq.write_table(table, parquet_file)


# max_vals = dict()
# min_vals = dict()
# parquet_path = Path("/home/fatemeh/Downloads/bird/data/ssl/ssl20parquet")
# parquet_files = list(parquet_path.glob("*.parquet"))
# for parquet_file in tqdm(parquet_files):
#     df = pd.read_parquet(parquet_file)
#     gimus = np.vstack(df["gimu"].apply(lambda x: x.reshape(-1, 4)))
#     device = int(parquet_file.stem)
#     min_vals[device] = gimus.min(axis=0)
#     max_vals[device] = gimus.max(axis=0)
# min_val = np.vstack(list(min_vals.values())).min(axis=0)
# max_val = np.vstack(list(max_vals.values())).max(axis=0)
# print(min_val)
# print(max_val)
# print("done")


# Plot histograms and scatter plots and save min/max stats
max_vals = dict()
min_vals = dict()
gps_scale = 22.3012351755624
save_path = Path("/home/fatemeh/Downloads/bird/data/ssl/hist_ssl20")
save_path.mkdir(parents=True, exist_ok=True)
parquet_path = Path("/home/fatemeh/Downloads/bird/data/ssl/ssl20parquet")
parquet_files = list(parquet_path.glob("*.parquet"))
parquet_files = sorted(parquet_files, key=lambda x: int(x.stem))
for parquet_file in tqdm(parquet_files):
    df = pd.read_parquet(parquet_file)
    gimus = np.vstack(df["gimu"].apply(lambda x: x.reshape(-1, 4)))
    gimus[:, 3] *= gps_scale
    device = int(parquet_file.stem)
    min_vals[device] = gimus.min(axis=0)
    max_vals[device] = gimus.max(axis=0)

    # Save figure
    fig, axs = plt.subplots(1, 4)
    for i in range(4):
        axs[i].hist(gimus[:, i])
    plt.savefig(save_path / f"{device}.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2)
    sns.scatterplot(x=gimus[:, 0], y=gimus[:, 3], ax=axs[0])
    axs[0].set_xlabel("imu_x")
    axs[0].set_ylabel("gps_speed")
    axs[0].set_title("imu_x vs gps_speed")
    sns.scatterplot(x=gimus[:, 1], y=gimus[:, 2], ax=axs[1])
    axs[1].set_xlabel("imu_y")
    axs[1].set_ylabel("imu_z")
    axs[1].set_title("imu_y vs imu_z")
    plt.tight_layout()
    plt.savefig(save_path / f"{device}_scatter.png")
    plt.close(fig)
    # # Extreme slow and get all memory
    # df = pd.DataFrame(gimus)
    # plt.figure()
    # sns.kdeplot(data=df, x=3)
    # sns.rugplot(data=df, x=3)
    # plt.savefig(save_path/f"{device}_gps.png")
    # plt.close(fig)
    # sns.scatterplot(data=df, x=0, y=3)
    # sns.rugplot(data=df, x=0, y=3)
    # plt.savefig(save_path/f"{device}_imu_gps.png")
    # plt.close(fig)
    # plt.figure()
    # sns.scatterplot(data=df, x=1, y=2)
    # sns.rugplot(data=df, x=1, y=2)
    # plt.savefig(save_path/f"{device}_imu_y_z.png")
    # plt.close(fig)
    """
    p99 = np.percentile(gimus, q=99, axis=0)
    gimus[np.any(gimus>p99, axis=1)].shape
    """
min_val = np.vstack(list(min_vals.values())).min(axis=0)
max_val = np.vstack(list(max_vals.values())).max(axis=0)
print(min_val)
print(max_val)


# Save stats
def write_stats(save_path, max_vals, max_val):
    with open(save_path / "stats.txt", "a") as wfile:
        wfile.write("device, imu_x, imu_y, imu_z, gps_speed\n")
        for k, v in max_vals.items():
            wfile.write(f"{k:4d}, {v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}, {v[3]:.2f}\n")
        wfile.write(
            f"{0:4d}, {max_val[0]:.2f}, {max_val[1]:.2f}, {max_val[2]:.2f}, {max_val[3]:.2f}\n"
        )


write_stats(save_path, max_vals, max_val)
write_stats(save_path, min_vals, min_val)

print("done")

# parquet_path = Path("/home/fatemeh/Downloads/bird/data/ssl/parquetmini")
# csv_files = list(Path("/home/fatemeh/Downloads/bird/data/ssl/csvmini").glob("*csv"))
# # csv_files = Path("/home/fatemeh/Downloads/bird/data/ssl/6210").glob("*.csv")
# # csv_files = sorted(csv_files, key=lambda x: int(x.stem.split("_")[1]))
# for csv_file in csv_files:
#     device = int(csv_file.stem)
#     parquet_file = parquet_path / f"{device}.parquet"
#     write_only_gimu_float32_norm_gps(csv_file, parquet_file)

# if "__main__" == __name__:
#     parquet_path = Path("/home/fatemeh/Downloads/bird/data/ssl/ssl20parquet")
#     parquet_path.mkdir(parents=True, exist_ok=True)

#     csv_path = Path("/home/fatemeh/Downloads/bird/data/ssl/ssl20")
#     devices = np.unique([int(p.stem.split("_")[0]) for p in csv_path.glob("*.csv")])

#     for device in tqdm(devices):
#         csv_files = csv_path.glob(f"{device}_*.csv")
#         csv_files = sorted(csv_files, key=lambda x: int(x.stem.split("_")[1]))
#         parquet_file = parquet_path / f"{device}.parquet"
#         write_only_gimu_float32_norm_gps_batch(csv_files, parquet_file)

# TODO
# - put them in a data processing script
# - pretrain_memory make the parquet more clear


"""
parquet_file = Path("/home/fatemeh/Downloads/bird/data/ssl/6210.parquet")
# csv_files = Path("/home/fatemeh/Downloads/bird/data/ssl/6210").glob("*.csv")
# csv_files = sorted(csv_files, key=lambda x: int(x.stem.split("_")[1]))
# write_only_gimu_float32_norm_gps_batch(csv_files, parquet_file)

df = pd.read_parquet(parquet_file)
data = np.vstack(df["gimu"].apply(lambda x: x.reshape(-1, 20, 4)))
print(data.shape)

# I want to read multiple parquet files and combine them into one big numpy array
combine_data = []
parquet_files = Path("/home/fatemeh/Downloads/bird/data/ssl/").glob("*.parquet")
for parquet_file in parquet_files:
    df = pd.read_parquet(parquet_file)
    data = np.vstack(df["gimu"].apply(lambda x: x.reshape(-1, 20, 4)))
    print(parquet_file.stem, data.shape)
    combine_data.append(data)
combine_data = np.vstack(combine_data)
print("combined data shape:", combine_data.shape)
"""

"""
I have:
- file:  20K rows
- files: 25M rows (1075 files)
25M * 20*4 * 4 ~= 8 GB

karpathy nanochat fineweb-edu-100b-shuffle data:
- file:  53K rows, 100M file size
- files: 97M rows, 171G file size  (1822 files)

e.g 6210 contains 75 files each abot 20K rows combining gets 1510999 rows. If only take gimu float32 save as parquet, file size is 300M.
1510999*80*4/(1024**2)=461M data.
"""
print("done")

# TODO: This part should be in the analyses or debugging part
"""
# Get the number of lines per device shard
cfg = dict(
    main_path=Path("/home/fatemeh/Downloads/bird/data/ssl/tmp"),
    save_path=Path("/home/fatemeh/Downloads/bird/data/ssl/stats"),
    database_url="",
    # database_url = "postgresql://username:password@host:port/database_name"
)
cfg = OmegaConf.create(cfg)  # megaConf.to_container for reverse

if not cfg.save_path.exists():
    cfg.save_path.mkdir(parents=True, exist_ok=True)

save_file = cfg.save_path / "device_num_lines.csv"
num_files = len(list(cfg.main_path.glob("*.csv")))
device_num_lines = dict()
with open(save_file, "w") as out_f:
    for p in tqdm(cfg.main_path.glob("*.csv"), total=num_files):
        with open(p) as f:
            device_id = int(p.stem)
            num_lines = len(f.readlines())
            device_num_lines[device_id] = num_lines
            out_f.write(f"{device_id},{num_lines}\n")
            if num_lines % 20 != 0:
                print(device_id, num_lines)


num_all_lines = sum(device_num_lines.values())
print(f"{num_all_lines:,}")


# Get per class counts per device shard
cfg = dict(
    glen=20,
    exp=125,
    labels_to_use=[0, 1, 2, 3, 4, 5, 6, 8, 9],
    in_channe=4,
    width=30,
    n_classes=None,
    checkpoint_file=Path(f"/home/fatemeh/Downloads/bird/result"),
    main_path=Path("/home/fatemeh/Downloads/bird/data/ssl/tmp"),
    save_path=Path("/home/fatemeh/Downloads/bird/data/ssl/stats"),
)
cfg = OmegaConf.create(cfg)
cfg.n_classes = len(cfg.labels_to_use)
cfg.checkpoint_file = cfg.checkpoint_file / f"{cfg.exp}_best.pth"

if not cfg.save_path.exists():
    cfg.save_path.mkdir(parents=True, exist_ok=True)

save_file = cfg.save_path / "device_classes.csv"
num_files = len(list(cfg.main_path.glob("*.csv")))
device_class_counts = dict()
with open(save_file, "w") as out_f:
    class_counts = defaultdict(int)
    _ = [class_counts[label] for label in cfg.labels_to_use]
    for p in tqdm(cfg.main_path.glob("*.csv"), total=num_files):
        device_id = int(p.stem)
        df = pd.read_csv(p, header=None).sort_values([0, 1, 2])
        df = bmu.infer_update_classes(
            df, cfg.glen, cfg.labels_to_use, cfg.checkpoint_file, cfg.n_classes
        )
        class_counts = dict(sorted(Counter(df[8]).items()))
        device_class_counts[device_id] = class_counts
        class_counts_one_line = ",".join([f"{k},{v}" for k, v in class_counts.items()])
        out_f.write(f"{device_id},{class_counts_one_line}\n")


all_classes_counts = sum((Counter(d) for d in device_class_counts.values()), Counter())
print(f"{all_classes_counts}")
"""

"""
{0: 1132, 1: 52, 2: 403, 3: 36, 4: 565, 5: 6654, 6: 240, 8: 54, 9: 162}  unlabel
{0: 888, 1: 70, 2: 223, 3: 105, 4: 172, 5: 5174, 6: 168, 8: 83, 9: 252}  unlabel

{0: 2020, 1: 122, 2: 626, 3: 141, 4: 737, 5: 11828, 6: 408, 8: 137, 9: 414} unlabel
{0: 643, 1: 38, 2: 537, 3: 176, 4: 729, 5: 1502, 6: 337, 8: 151, 9: 225} labed
"""
