"""How much does a model rely on GPS speed? Models on raw vs clean labels.

`starts_clean.csv` drops the 26 bursts whose GPS speed contradicts their label.
They were the only fast-but-not-flying bursts in training, so a model trained
without them might trust speed more. This measures that. The acceleration is
never changed, only the speed, so any change in the prediction comes from
speed alone.

Four tests, on all bursts of `data_file`:

1. **One burst, speed swept.** `sweep_burst` keeps its acceleration while its
   speed goes through `sweep_speeds`; prints P(`sweep_class`) at each speed.
2. **One class, slow vs fast.** Every burst of `sweep_class`, mean
   P(`sweep_class`) with the speed set to 0 and to the last of `sweep_speeds`.
3. **Speeds shuffled.** Each burst gets the speed of another burst; the share of
   bursts whose predicted class changes, mean of `n_shuffles` shuffles.
4. **Speed set to 0.** The share of bursts whose predicted class changes.

A higher number in 3 and 4, or a steeper fall in 1 and 2, means the model leans
more on speed. Compare several seeds per side: one seed pair can mislead (see
docs/lesson_learned.md).

Imports the checkpoint loader from `exps/eval_labeled.py`.

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/speed_reliance.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from behavior import data as bd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_labeled import as_markdown, channels_of, load_model  # noqa: E402


@torch.no_grad()
def probabilities(model, igs, speed, in_channels, device):
    """Class probabilities, with the speed channel replaced by `speed`.

    Goes through `bd.BirdDataset`, so the magnitude channels and the speed
    scaling are exactly those of training.
    """
    igs = igs.copy()
    igs[:, :, 3] = speed[:, None]
    dataset = bd.BirdDataset(igs, None, None, add_magnitudes=in_channels == 7)
    x = torch.from_numpy(dataset.data.transpose(0, 2, 1).copy()).to(device)
    return torch.softmax(model(x), dim=1).cpu().numpy()


def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    igs, ldts = bd.load_csv_pandas(cfg.data_file, cfg.labels_to_use, glen=cfg.glen)
    igs = igs.astype(np.float32)
    speed = igs[:, 0, 3].copy()
    keys = bd.burst_keys(cfg.data_file, cfg.labels_to_use, cfg.glen)
    burst = keys.index(tuple(cfg.sweep_burst))
    cls = list(cfg.labels_to_use).index(cfg.sweep_class)
    of_class = np.where(ldts[:, 0] == cls)[0]
    sweep = np.asarray(cfg.sweep_speeds, dtype=np.float32)
    rng = np.random.default_rng(cfg.seed)
    shuffles = [rng.permutation(len(igs)) for _ in range(cfg.n_shuffles)]
    print(
        f"{cfg.data_file}: {len(igs)} bursts\n"
        f"sweep burst {tuple(cfg.sweep_burst)}, speed {speed[burst]:.2f} m/s; "
        f"{len(of_class)} bursts of class {cfg.sweep_class}\n"
    )

    sweep_column = "one burst, P at " + "/".join(f"{s:g}" for s in sweep) + " m/s"
    class_column = f"class mean P, 0 vs {sweep[-1]:g} m/s"
    rows = []
    for exp in cfg.model_exps:
        model = load_model(cfg, exp, device)
        ch = channels_of(cfg, exp)

        def prob(idx, s):
            return probabilities(model, igs[idx], s, ch, device)

        one = prob(np.repeat(burst, len(sweep)), sweep)[:, cls]
        n = len(of_class)
        slow = prob(of_class, np.zeros(n, np.float32))[:, cls].mean()
        fast = prob(of_class, np.full(n, sweep[-1], np.float32))[:, cls].mean()
        everyone = np.arange(len(igs))
        base = prob(everyone, speed).argmax(1)
        shuffled = np.mean(
            [(prob(everyone, speed[p]).argmax(1) != base).mean() for p in shuffles]
        )
        zero = (prob(everyone, np.zeros(len(igs), np.float32)).argmax(1) != base).mean()
        rows.append(
            {
                "model": f"exp{exp}",
                "labels": cfg.trained_on[str(exp)],
                sweep_column: " ".join(f"{p:.2f}" for p in one),
                class_column: f"{slow:.2f} vs {fast:.2f}",
                "changed, shuffled %": 100 * shuffled,
                "changed, speed 0 %": 100 * zero,
            }
        )
    print(as_markdown(pd.DataFrame(rows).round(1)))


if __name__ == "__main__":
    config = {
        "data_file": "/home/fatemeh/Downloads/bird/data/final/starts.csv",
        "save_path": "/home/fatemeh/Downloads/bird/results",
        "model_exps": [197, 199, 201, 202, 200, 203, 204],
        # Which labels each run was trained on, only for the printed table.
        "trained_on": {
            "197": "raw",
            "199": "raw",
            "201": "raw",
            "202": "raw",
            "200": "clean",
            "203": "clean",
            "204": "clean",
        },
        # Input width per checkpoint; 7 means it was trained with add_magnitudes.
        "in_channels": {str(e): 7 for e in [197, 199, 200, 201, 202, 203, 204]},
        # Checkpoint per run; the default is <exp>_best.pth. exp197 is the
        # working model, so its best checkpoint is the one to test.
        "checkpoints": {
            str(e): f"{e}_4000.pth" for e in [199, 200, 201, 202, 203, 204]
        },
        "labels_to_use": [0, 1, 2, 3, 4, 5, 6, 8, 9],
        "glen": 20,
        # An ExFlap burst at a Flap-like speed (4.99 m/s); one of the two
        # ExFlap valid bursts that flip between runs (see experiments_log.md).
        "sweep_burst": [782, "2014-05-26 10:41:11", 14],
        "sweep_class": 1,  # ExFlap
        "sweep_speeds": [0, 5, 10, 15, 20],
        "n_shuffles": 5,
        "seed": 0,
    }
    main(OmegaConf.create(config))
