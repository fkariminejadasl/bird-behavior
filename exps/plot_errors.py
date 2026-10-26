"""Plot the valid bursts one model gets wrong, one PNG per burst.

`exps/eval_labeled.py` says *how much* a class costs; this says *which bursts*,
so they can be looked at. Written for exp197, whose three weakest classes are
TerLoco (walking), Pecking and Manouvre.

Imports the split, the checkpoint loader and the prediction step from
`exps/eval_labeled.py`, so a burst plotted here is exactly one that the F1 table
there counts as wrong.

Only bursts whose *true* label is one of `classes` are plotted. The summary
table also counts the other direction, bursts of another class predicted as one
of these three, because F1 pays for both.

One PNG per wrong burst, from `behavior/utils.py::plot_one` (acc x red, y blue,
z green), under `<save_path>/errors_exp<exp>/<true class>/`, named
`<predicted class>,<device>,<datetime>,<start index>.png` in the style of
`/home/fatemeh/Downloads/bird/results/gt2_starts`.

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/plot_errors.py
"""

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from behavior import data as bd
from behavior import utils as bu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_labeled import (  # noqa: E402  (same directory)
    as_markdown,
    build_splits,
    channels_of,
    load_model,
    predict,
)
from find_label_noise import load_bursts  # noqa: E402


def valid_bursts(cfg, device):
    """Unnormalized valid bursts, with the device, datetime and start index.

    Repeats the seeded split of `eval_labeled.build_splits`, which returns model
    input but not burst identity. `stratified_split` depends only on the labels
    and the seed, so row i here is row i there; `main` asserts it.

    The datetime comes from `find_label_noise.load_bursts`, which reads the CSV
    column as text. The timestamp of `bd.load_csv_pandas` is not usable: under
    pandas 3 it is 1000x too small (see docs/lesson_learned.md).
    """
    bu.set_seed(cfg.seed)
    igs, ldts = bd.load_csv_pandas(cfg.data_file, cfg.labels_to_use, glen=cfg.glen)
    bursts = load_bursts(cfg.data_file, cfg.glen)
    _, valid = bu.stratified_split(
        torch.tensor(ldts[:, 0], device=device),
        split_ratios=[cfg.train_per, 1 - cfg.train_per],
        seed=cfg.seed,
    )
    valid = valid.cpu().numpy()
    # load_bursts does not filter rows, so check it lines up before indexing.
    labels = np.asarray(cfg.labels_to_use)[ldts[:, 0]]
    assert len(bursts["label"]) == len(ldts), "burst counts differ"
    assert (bursts["label"] == labels).all(), "burst order differs"
    ids = pd.DataFrame(
        {k: bursts[k][valid] for k in ("device_id", "datetime", "start_index")}
    )
    ids["label"] = ldts[valid, 0]
    return igs[valid], ids


def summary_table(true, pred, names, class_ids):
    """Per class: how many valid bursts, how many wrong, and where they went."""
    rows = []
    for c in class_ids:
        missed = pred[true == c]
        missed = missed[missed != c]
        false = true[pred == c]
        false = false[false != c]
        rows.append(
            {
                "true": names[c],
                "valid": int((true == c).sum()),
                "wrong": len(missed),
                "predicted as": ", ".join(
                    f"{names[p]} {n}" for p, n in Counter(missed).most_common()
                ),
                "wrongly predicted": len(false),
                "coming from": ", ".join(
                    f"{names[t]} {n}" for t, n in Counter(false).most_common()
                ),
            }
        )
    return pd.DataFrame(rows)


def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    names = [bu.ind2name[i] for i in cfg.labels_to_use]
    class_ids = [names.index(n) for n in cfg.classes]

    model = load_model(cfg, cfg.exp, device)
    x, y = build_splits(cfg, device, channels_of(cfg, cfg.exp))["valid"]
    igs, ids = valid_bursts(cfg, device)
    true, pred = y.cpu().numpy(), predict(model, x).cpu().numpy()
    assert (ids["label"].values == true).all(), "the two splits are not aligned"

    save_path = Path(cfg.save_path) / f"errors_exp{cfg.exp}"
    wrong = np.where((pred != true) & np.isin(true, class_ids))[0]
    for i in wrong:
        dev, time, ind = ids.iloc[i][["device_id", "datetime", "start_index"]]
        bu.plot_one(igs[i], SHOW=False)
        plt.title(
            f"{names[true[i]]} -> {names[pred[i]]}, {dev}, {time}, "
            f"ind {ind}, gps {igs[i, 0, 3]:.2f}"
        )
        label_path = save_path / names[true[i]]
        label_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            label_path / f"{names[pred[i]]},{dev},{time},{ind}.png",
            bbox_inches="tight",
        )
        plt.close()

    print(
        f"exp{cfg.exp}, {channels_of(cfg, cfg.exp)} channels, {len(true)} valid "
        f"bursts, accuracy {(pred == true).mean() * 100:.2f}\n"
        f"{len(wrong)} plots in {save_path}\n"
    )
    print(as_markdown(summary_table(true, pred, names, class_ids)))


if __name__ == "__main__":
    config = {
        "data_file": "/home/fatemeh/Downloads/bird/data/final/starts.csv",
        "save_path": "/home/fatemeh/Downloads/bird/results",
        "exp": 197,
        # Input width of the checkpoint; 7 means trained with add_magnitudes.
        "in_channels": {"197": 7},
        "classes": ["TerLoco", "Pecking", "Manouvre"],
        "labels_to_use": [0, 1, 2, 3, 4, 5, 6, 8, 9],
        "seed": 32984,
        "train_per": 0.9,
        "glen": 20,
    }
    main(OmegaConf.create(config))
