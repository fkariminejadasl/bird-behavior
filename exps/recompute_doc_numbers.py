"""Recompute numbers the docs quote that no other script prints.

Each check prints one group of numbers, and the docs name this script beside
them. Results of the last run, 2026-11-16:

1. `split_shift`: dropping the 26 label-noise bursts reshuffles the split. 177
   of exp197's 438 valid bursts (`starts.csv`) fall in exp198's train split
   (`starts_clean.csv`, the same seed 32984).
2. `class_speeds`: GPS speed per burst, in m/s. ExFlap median 1.39 (90th
   percentile 8.90), Flap median 9.63 (10th percentile 4.92), in both files.
   The fastest land or water burst (SitStand, TerLoco, Pecking, Float, Boat) is
   17.78 m/s in `starts.csv` and 4.72 in `starts_clean.csv`.
3. `exflap_flips`: the 4 ExFlap bursts of the 437 valid bursts that exp199-204
   share, read at epoch 4000. Only the two at 15.84 and 4.99 m/s are ever called
   Flap, and only by clean-label runs: 15.84 by exp200 and exp203, 4.99 by
   exp200. The two at 0.06 m/s are always right.
4. `ornitela_nearest_fix`: the GPS fix each burst gets from
   `scripts/data/bird_behavior_app_data.py::prepare_calibrated_app_data`, before
   commit a71729f (the GPS row above the burst) and now (the nearest fix within
   2 s), on `all_devices_calibrated.csv`. Before: 659,409 bursts of 20 samples.
   247 bursts come back, 659,656 now. 287 bursts (160 fixes) get another fix;
   for the 113 fixes whose speed changed, the change has a median of 0.28 m/s
   and a maximum of 7.78.
5. `app_file_limits`: what `infer_save_app_data` drops and clips. In
   `rose_data_old.csv`, the app file of June, 6 bursts of 20 samples are at 30
   m/s or faster, and 4,621 rows have an IMU value beyond 2 g. In
   `rose_data.csv`, now: 6 and 4,622.

Checks 1 and 3 need the GPU: the split is drawn by a generator on the device,
and a CPU split is a different one (`exps/eval_labeled.py::build_splits`).
Check 4 reads a 2.7 GB file and writes the new app file to a temporary
directory, deleted at the end; it takes about 2 minutes. Nothing else is
written.

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/recompute_doc_numbers.py
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "exps"))
sys.path.insert(0, str(root / "scripts" / "data"))
import bird_behavior_app_data as app  # noqa: E402
import eval_labeled as el  # noqa: E402

from behavior import utils as bu  # noqa: E402

LAND_OR_WATER = ["SitStand", "TerLoco", "Pecking", "Float", "Boat"]


def gpu():
    assert torch.cuda.is_available(), "the training split was drawn on the GPU"
    return torch.device("cuda")


def split_shift(cfg):
    """Valid bursts of the raw split that the clean split puts in train."""
    raw, clean = cfg.data_dir + "starts.csv", cfg.data_dir + "starts_clean.csv"
    valid_raw = el.valid_keys(cfg, raw, gpu())
    train_clean = set(el.burst_keys(cfg, clean)) - el.valid_keys(cfg, clean, gpu())
    print(
        f"{len(valid_raw & train_clean)} of the {len(valid_raw)} valid bursts of "
        f"starts.csv are in the train split of starts_clean.csv"
    )


def class_speeds(cfg):
    """GPS speed per burst and class, in m/s, in both data files."""
    for name in ["starts.csv", "starts_clean.csv"]:
        df = pd.read_csv(cfg.data_dir + name, header=None)
        first = df.iloc[:: cfg.glen]
        speed = pd.Series(first[7].values, index=[bu.ind2name[lab] for lab in first[3]])
        exflap, flap = speed["ExFlap"], speed["Flap"]
        print(
            f"{name}, {len(first)} bursts: ExFlap median {exflap.median():.2f} "
            f"(p90 {exflap.quantile(0.9):.2f}), Flap median {flap.median():.2f} "
            f"(p10 {flap.quantile(0.1):.2f}), fastest land or water burst "
            f"{speed[speed.index.isin(LAND_OR_WATER)].max():.2f}"
        )


def exflap_flips(cfg):
    """Each ExFlap valid burst, its speed and the class every run gives it."""
    run_cfg = OmegaConf.merge(
        cfg,
        {
            "data_file": cfg.data_dir + "starts_clean.csv",
            "split_file": cfg.data_dir + "starts.csv",
            "in_channels": {str(e): 7 for e in cfg.runs},
            "checkpoints": {str(e): f"{e}_4000.pth" for e in cfg.runs},
        },
    )
    x, y = el.build_splits(run_cfg, gpu(), in_channels=7)["valid"]
    borrowed = el.valid_keys(run_cfg, run_cfg.split_file, gpu())
    keys = [k for k in el.burst_keys(run_cfg, run_cfg.data_file) if k in borrowed]
    assert len(keys) == len(y)
    names = [bu.ind2name[lab] for lab in cfg.labels_to_use]
    preds = {e: el.predict(el.load_model(run_cfg, e, gpu()), x).cpu() for e in cfg.runs}
    exflap = names.index("ExFlap")
    print(f"{len(y)} valid bursts; class per run, raw {cfg.runs[:3]} then clean:")
    for i in torch.nonzero(y == exflap).squeeze(1).tolist():
        speed = x[i, 3, 0].item() * cfg.gps_norm
        calls = ", ".join(f"exp{e} {names[preds[e][i]]}" for e in cfg.runs)
        print(f"  {keys[i]}, {speed:.2f} m/s: {calls}")


def fixes_row_above(df):
    """The GPS fix of each burst before commit a71729f, one row per fix.

    A copy of the matching in `prepare_calibrated_app_data` before that commit:
    each SENSOR row takes the last valid GPS row above it, and a GPS row is
    valid only if the row right after it is a SENSOR row of the same device at
    most 2 s away.
    """
    is_gps = df["datatype"].eq("GPS")
    is_sensor = df["datatype"].eq("SENSORS")
    valid_gps = (
        is_gps
        & df["datatype"].shift(-1).eq("SENSORS")
        & df["device_id"].eq(df["device_id"].shift(-1))
        & ~(df["Latitude"].eq(0) & df["Longitude"].eq(0))
        & ((df["UTC_datetime"].shift(-1) - df["UTC_datetime"]).abs()).le(
            pd.Timedelta(seconds=2)
        )
    )
    df = df.copy()
    df["gps_time"] = df["UTC_datetime"].where(is_gps)
    df["gps_time"] = df.groupby("device_id")["gps_time"].ffill()
    df["valid_time"] = df["UTC_datetime"].where(valid_gps)
    df["valid_time"] = df.groupby("device_id")["valid_time"].ffill()
    for col in ["Latitude", "Longitude", "speed_km_h"]:
        df[col] = df[col].where(valid_gps)
        df[col] = df.groupby("device_id")[col].ffill()
    kept = df[
        is_sensor
        & df["gps_time"].eq(df["valid_time"])
        & df["Latitude"].notna()
        & df["Longitude"].notna()
    ]
    fixes = kept.groupby(["device_id", "UTC_datetime"]).agg(
        n=("x_g", "size"),
        lat=("Latitude", "first"),
        lon=("Longitude", "first"),
        speed=("speed_km_h", "first"),
    )
    fixes["speed"] = fixes["speed"] / 3.6
    fixes["bursts"] = fixes["n"] // 20
    return fixes[fixes["bursts"] > 0]


def fixes_nearest(src):
    """The GPS fix of each burst now: `prepare_calibrated_app_data` itself."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "app.csv"
        app.prepare_calibrated_app_data(src, out)
        df = pd.read_csv(out, header=None, engine="pyarrow")
    df.columns = range(df.shape[1])
    df[1] = pd.to_datetime(df[1])
    fixes = df.groupby([0, 1]).agg(
        n=(2, "size"), lat=(10, "first"), lon=(11, "first"), speed=(7, "first")
    )
    fixes.index.names = ["device_id", "UTC_datetime"]
    fixes["bursts"] = fixes["n"] // 20
    return fixes


def ornitela_nearest_fix(cfg):
    """Bursts that get another GPS fix, or come back, with the nearest fix."""
    src = Path(cfg.simon_dir) / "all_devices_calibrated.csv"
    cols = ["device_id", "UTC_datetime", "datatype", "Latitude", "Longitude"]
    cols += ["speed_km_h", "x_g"]
    df = pd.read_csv(src, usecols=cols, engine="pyarrow")
    df["UTC_datetime"] = pd.to_datetime(df["UTC_datetime"])
    old = fixes_row_above(df)
    new = fixes_nearest(src)
    both = old.join(new, rsuffix="_new", how="inner")
    speed_change = (both["speed"] - both["speed_new"]).abs()
    other = (
        (both["lat"] - both["lat_new"]).abs().gt(1e-9)
        | (both["lon"] - both["lon_new"]).abs().gt(1e-9)
        | speed_change.gt(1e-6)
    )
    back = new.index.difference(old.index)
    lost = old.index.difference(new.index)
    moved = speed_change[speed_change.gt(1e-6)]
    print(
        f"bursts of 20 samples: {old.bursts.sum():,} before, {new.bursts.sum():,} "
        f"now; {new.loc[back, 'bursts'].sum()} come back, "
        f"{old.loc[lost, 'bursts'].sum()} are lost"
    )
    print(
        f"{new.loc[other[other].index, 'bursts'].sum()} bursts "
        f"({int(other.sum())} fixes) get another fix; for the {len(moved)} fixes "
        f"whose speed changed, median {moved.median():.2f} m/s, max {moved.max():.2f}"
    )


def app_file_limits(cfg):
    """Bursts at 30 m/s or faster, and rows beyond 2 g, in the app files."""
    for name in ["rose_data_old.csv", "rose_data.csv"]:
        df = pd.read_csv(Path(cfg.simon_dir) / name, header=None, engine="pyarrow")
        df.columns = range(df.shape[1])
        fast = int(df[7].ge(30).sum()) // 20
        beyond = int(df[[4, 5, 6]].abs().gt(2).any(axis=1).sum())
        print(
            f"{name}: {len(df) // 20:,} bursts of 20 samples, {fast} at 30 m/s or "
            f"faster, {beyond:,} rows with an IMU value beyond 2 g"
        )


CHECKS = {
    "split_shift": split_shift,
    "class_speeds": class_speeds,
    "exflap_flips": exflap_flips,
    "ornitela_nearest_fix": ornitela_nearest_fix,
    "app_file_limits": app_file_limits,
}


def main(cfg):
    for name in cfg.checks:
        print(f"\n## {name}\n")
        CHECKS[name](cfg)


if __name__ == "__main__":
    config = {
        "data_dir": "/home/fatemeh/Downloads/bird/data/final/",
        "simon_dir": "/home/fatemeh/Downloads/bird/data/simon",
        "save_path": "/home/fatemeh/Downloads/bird/results",
        "checks": list(CHECKS),  # or a subset; ornitela_nearest_fix is the slow one
        "runs": [199, 201, 202, 200, 203, 204],  # raw seeds, then clean seeds
        "labels_to_use": [0, 1, 2, 3, 4, 5, 6, 8, 9],
        "glen": 20,
        "train_per": 0.9,
        "seed": 32984,
        "gps_norm": 22.3012351755624,
    }
    main(OmegaConf.create(config))
