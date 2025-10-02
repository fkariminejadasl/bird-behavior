from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from behavior import model_utils as bmu

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
{0: 1132, 1: 52, 2: 403, 3: 36, 4: 565, 5: 6654, 6: 240, 8: 54, 9: 162}  unlabel
{0: 888, 1: 70, 2: 223, 3: 105, 4: 172, 5: 5174, 6: 168, 8: 83, 9: 252}  unlabel

{0: 2020, 1: 122, 2: 626, 3: 141, 4: 737, 5: 11828, 6: 408, 8: 137, 9: 414} unlabel
{0: 643, 1: 38, 2: 537, 3: 176, 4: 729, 5: 1502, 6: 337, 8: 151, 9: 225} labed
"""
