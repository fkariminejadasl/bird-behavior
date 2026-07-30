"""Diff the per-class metrics of two experiments, to see where a change paid.

`scripts/batch_train_supervised.py` writes four CSVs per run into
`save_path/failed/<exp>_<data stem>/`:

    per_class_metrics_{train,valid}.csv           counts as they are
    per_class_metrics_balanced_{train,valid}.csv  resampled to equal class size

Each has TP / FP / FN / precision / recall / F1 per class plus a `Combined` row.
Overall accuracy hides which classes moved, and the two files can disagree in
sign for a rare class, so this prints both. Edit the config in `__main__` to
change the run.

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/compare_per_class_metrics.py
"""

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


def read(cfg, exp, kind, stage):
    """kind: '' or 'balanced_'; stage: 'train' or 'valid'."""
    path = (
        Path(cfg.save_path)
        / f"failed/{exp}_{cfg.data_stem}/per_class_metrics_{kind}{stage}.csv"
    )
    return pd.read_csv(path).set_index("Class")


def compare(cfg, kind, stage):
    a, b = cfg.exps
    da, db = read(cfg, a, kind, stage), read(cfg, b, kind, stage)
    print(f"\n=== {kind}{stage}: F1 per class, exp{a} -> exp{b} ===")
    print(f"{'class':<10}{f'exp{a}':>8}{f'exp{b}':>8}{'delta':>8}")
    for cls in da.index:
        if cls not in db.index:
            continue
        fa, fb = da.loc[cls, "F1 Score"], db.loc[cls, "F1 Score"]
        mark = "  <<" if abs(fb - fa) >= cfg.flag_delta else ""
        print(f"{cls:<10}{fa:>8.2f}{fb:>8.2f}{fb - fa:>+8.2f}{mark}")


def main(cfg):
    for stage in ("train", "valid"):
        for kind in ("", "balanced_"):
            compare(cfg, kind, stage)
    print(f"\n'<<' marks a class whose F1 moved by at least {cfg.flag_delta:.2f}.")
    print("The 'Combined' row of a balanced file is the class-balanced score.")


if __name__ == "__main__":
    config = {
        "save_path": "/home/fatemeh/Downloads/bird/results",
        "data_stem": "starts",
        # two experiment numbers: baseline then variant
        "exps": [194, 195],
        "flag_delta": 0.10,  # mark a class whose F1 moved at least this much
    }
    main(OmegaConf.create(config))
