"""Orientation robustness of two trained checkpoints on the same valid split.

An A/B on the standard split only shows what an orientation augmentation costs,
not what it buys: the labeled data is one logger type mounted much the same way
on every bird, so the split holds almost no orientation variation. This
re-evaluates both checkpoints on the same validation bursts with the
accelerometer frame perturbed (IMU x, y, z only; GPS 2D speed untouched):

  - clean       : no perturbation, reproduces failed/<exp>_<stem>/app_loss_acc.txt
  - xy_swap     : x <-> y, a reflection (det -1) and so outside SO(3)
  - pitch_<deg> : rotation about the x axis, a sensitivity curve in tilt
  - so3         : full random SO(3), averaged over cfg.n_so3 draws — the realistic
                  case for a logger whose tilt and position on the bird are both
                  unknown

Inference only, no training. Edit the config in `__main__` to change the run.

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/eval_rotation_robustness.py
"""

import math

import numpy as np
import torch
from omegaconf import OmegaConf

from behavior import data as bd
from behavior import model as bm
from behavior import utils as bu
from behavior.data_augmentation import random_rotation_matrix


def build_valid_split(cfg, device):
    """The stratified valid split of scripts/batch_train_supervised.py."""
    bu.set_seed(cfg.seed)
    igs, ldts = bd.load_csv_pandas(cfg.data_file, cfg.labels_to_use, glen=20)
    igs = torch.tensor(igs, device=device)
    ldts = torch.tensor(ldts, device=device)
    splits = bu.stratified_split(
        ldts[:, 0], split_ratios=[cfg.train_per, 1 - cfg.train_per], seed=cfg.seed
    )
    idx = splits[1]
    return bd.BirdDataset(
        igs[idx].cpu().numpy(), ldts[idx].cpu().numpy(), None, channel_first=True
    )


def rot_x(deg, device):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], device=device)


def swap_xy(device):
    return torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device
    )


def apply_rot(x, R):
    """x: (N, C, T). Rotate the acc channels 0..2 only: v' = R @ v."""
    out = x.clone()
    out[:, :3, :] = torch.einsum("ij,njt->nit", R, x[:, :3, :])
    return out


@torch.no_grad()
def accuracy(model, x, y):
    return (model(x).argmax(1) == y).float().mean().item() * 100


def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = build_valid_split(cfg, device)
    x = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    # load_csv_pandas already remaps the labels to contiguous 0..n_classes-1
    y = torch.tensor(
        [int(dataset[i][1][0]) for i in range(len(dataset))], device=device
    )
    print(f"valid samples: {len(dataset)}, device: {device}")

    models = dict()
    for exp in cfg.exps:
        model = bm.BirdModelSmallDilated(**cfg.model.parameters).to(device)
        bm.load_model(f"{cfg.save_path}/{exp}_best.pth", model, device)
        model.eval()
        models[exp] = model

    rows = [
        ("clean", {e: accuracy(m, x, y) for e, m in models.items()}),
        (
            "xy_swap",
            {
                e: accuracy(m, apply_rot(x, swap_xy(device)), y)
                for e, m in models.items()
            },
        ),
    ]
    for deg in cfg.pitches:
        R = rot_x(deg, device)
        rows.append(
            (
                f"pitch_{deg}",
                {e: accuracy(m, apply_rot(x, R), y) for e, m in models.items()},
            )
        )

    torch.manual_seed(0)
    Rs = [
        random_rotation_matrix(device=device, dtype=x.dtype) for _ in range(cfg.n_so3)
    ]
    so3 = {
        e: np.array([accuracy(m, apply_rot(x, R), y) for R in Rs])
        for e, m in models.items()
    }
    rows.append(("so3_mean", {e: a.mean() for e, a in so3.items()}))

    a, b = cfg.exps
    print(f"\n{'perturbation':<14}{f'exp{a}':>12}{f'exp{b}':>12}{'delta':>9}")
    for name, vals in rows:
        print(f"{name:<14}{vals[a]:>12.2f}{vals[b]:>12.2f}{vals[b] - vals[a]:>+9.2f}")
    for exp, acc in so3.items():
        print(
            f"exp{exp} so3 over {cfg.n_so3} orientations: mean {acc.mean():.2f}, "
            f"std {acc.std():.2f}, min {acc.min():.2f}, max {acc.max():.2f}"
        )


if __name__ == "__main__":
    config = {
        "save_path": "/home/fatemeh/Downloads/bird/results",
        "data_file": "/home/fatemeh/Downloads/bird/data/final/starts.csv",
        # two experiment numbers: baseline then augmented
        "exps": [194, 195],
        "labels_to_use": [0, 1, 2, 3, 4, 5, 6, 8, 9],
        "seed": 32984,
        "train_per": 0.9,
        "n_so3": 20,  # random orientations averaged over
        "pitches": [20, 45, 70, 90],
        # model of both checkpoints
        "model": {
            "name": "BirdModelSmallDilated",
            "parameters": {
                "in_channels": 4,
                "mid_channels": 20,
                "out_channels": 9,
                "dropout": 0.15,
            },
        },
    }
    main(OmegaConf.create(config))
