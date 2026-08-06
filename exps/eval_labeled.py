"""Compare models on the **labeled** split, in one table per question.

The labeled counterpart of `exps/eval_unlabeled.py`. Everything here needs
ground truth; everything there needs none. Together they are the whole picture,
and both print markdown ready to paste into docs/lesson_learned.md.

Three questions, three tables:

1. **How accurate?** accuracy, average precision and loss on train and valid,
   plus valid F1 plain and class-balanced. AP and loss match `app_loss_acc.txt`.
   The plain and balanced numbers can disagree in sign on a rare class -- with 16
   Manouvre and 4 ExFlap bursts in the valid split, one sample moves an F1 by
   0.06 -- so both are always shown.
2. **Which classes pay?** per-class valid F1 side by side, flagging any class
   that moves by `flag_delta` between models.
3. **Does it survive a different tag?** the same valid bursts with the
   accelerometer frame perturbed. The labeled split is one logger type mounted
   much the same way on every bird, so plain accuracy cannot see this at all.

Reuses `behavior.utils.per_class_statistics{,_balanced}`, the same functions
`scripts/batch_train_supervised.py` writes its CSVs with, so every number matches
`failed/<exp>_<stem>/per_class_metrics_*.csv` and `app_loss_acc.txt` exactly. It
recomputes from the checkpoints rather than reading those files, so it works for
any model and any perturbation. It replaces the former
`compare_per_class_metrics.py` and `eval_rotation_robustness.py`.

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/eval_labeled.py
"""

import math

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score

from behavior import data as bd
from behavior import model as bm
from behavior import utils as bu
from behavior.data_augmentation import random_rotation_matrix


def rot_x(deg, device):
    """Rotation about the x axis: a change in the tag's mounting pitch."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], device=device)


def swap_xy(device):
    """Swap x and y: the Ornitela vs UvA-BiTS column convention.

    A reflection (det -1), so outside SO(3) and outside what the rotation
    augmentation samples -- which makes it a genuine out-of-distribution test.
    """
    return torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device
    )


def apply_rot(x, mat):
    """x: (N, C, T). Rotate the acc channels 0..2 only: v' = R @ v."""
    out = x.clone()
    out[:, :3, :] = torch.einsum("ij,njt->nit", mat, x[:, :3, :])
    return out


def build_splits(cfg, device):
    """The exact train/valid split of scripts/batch_train_supervised.py.

    The label tensor must be on `device` before `stratified_split`: it seeds a
    `torch.Generator(device=...)`, so a CPU tensor gives a different split and
    silently mixes training bursts into the validation set.
    """
    bu.set_seed(cfg.seed)
    igs, ldts = bd.load_csv_pandas(cfg.data_file, cfg.labels_to_use, glen=cfg.glen)
    igs = torch.tensor(igs, device=device)
    ldts = torch.tensor(ldts, device=device)
    splits = bu.stratified_split(
        ldts[:, 0], split_ratios=[cfg.train_per, 1 - cfg.train_per], seed=cfg.seed
    )
    out = {}
    for name, idx in zip(("train", "valid"), splits):
        dataset = bd.BirdDataset(
            igs[idx].cpu().numpy(), ldts[idx].cpu().numpy(), None, channel_first=True
        )
        x = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
        y = torch.tensor(
            [int(dataset[i][1][0]) for i in range(len(dataset))], device=device
        )
        out[name] = (x, y)
    return out


def load_model(cfg, exp, device):
    model = bm.BirdModelSmallDilated(4, 20, len(cfg.labels_to_use)).to(device)
    bm.load_model(f"{cfg.save_path}/{exp}_best.pth", model, device)
    model.eval()
    return model


@torch.no_grad()
def predict(model, x):
    return model(x).argmax(1)


@torch.no_grad()
def scores(model, x, y):
    """Accuracy, average precision and cross-entropy loss for one split.

    AP and loss are computed exactly as `behavior.utils` does when training
    writes `app_loss_acc.txt`, so the values line up with that file.
    """
    logits = model(x)
    prob = torch.softmax(logits, dim=1).cpu().numpy()
    labels = y.cpu().numpy()
    loss = torch.nn.functional.cross_entropy(logits, y).item()
    accuracy = (logits.argmax(1) == y).float().mean().item() * 100
    ap = average_precision_score(labels, prob)
    return accuracy, ap, loss


def confusion(pred, y, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y.cpu().numpy(), pred.cpu().numpy()), 1)
    return cm  # rows = true, cols = predicted, as behavior.utils expects


def accuracy_table(cfg, splits, models, names):
    rows = []
    for exp, model in models.items():
        row = {"model": f"exp{exp}"}
        for stage, (x, y) in splits.items():
            acc, ap, loss = scores(model, x, y)
            row[f"{stage}_acc"] = acc
            row[f"{stage}_AP"] = ap
            row[f"{stage}_loss"] = loss
        cm = confusion(
            predict(model, splits["valid"][0]), splits["valid"][1], len(names)
        )
        plain = bu.per_class_statistics(cm, names)
        balanced = bu.per_class_statistics_balanced(cm, names)
        row["valid_F1"] = plain.set_index("Class").loc["Combined", "F1 Score"]
        row["valid_F1_bal"] = balanced.set_index("Class").loc["Combined", "F1 Score"]
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def per_class_table(cfg, splits, models, names, kind):
    """kind: 'plain' or 'balanced'."""
    stats = {}
    for exp, model in models.items():
        cm = confusion(
            predict(model, splits["valid"][0]), splits["valid"][1], len(names)
        )
        fn = (
            bu.per_class_statistics
            if kind == "plain"
            else bu.per_class_statistics_balanced
        )
        stats[f"exp{exp}"] = fn(cm, names).set_index("Class")["F1 Score"]
    out = pd.DataFrame(stats)
    if len(models) == 2:
        a, b = out.columns
        out["delta"] = (out[b] - out[a]).round(2)
        out["moved"] = np.where(out["delta"].abs() >= cfg.flag_delta, "<<", "")
    return out


def robustness_table(cfg, splits, models, device):
    x, y = splits["valid"]
    perturbations = [("clean", None), ("xy_swap", swap_xy(device))]
    perturbations += [(f"pitch_{d}", rot_x(d, device)) for d in cfg.pitches]

    rows = []
    for name, mat in perturbations:
        row = {"valid under": name}
        for exp, model in models.items():
            xx = x if mat is None else apply_rot(x, mat)
            row[f"exp{exp}"] = (predict(model, xx) == y).float().mean().item() * 100
        rows.append(row)

    torch.manual_seed(0)
    mats = [
        random_rotation_matrix(device=device, dtype=x.dtype) for _ in range(cfg.n_so3)
    ]
    row = {"valid under": f"so3 (mean of {cfg.n_so3})"}
    for exp, model in models.items():
        accs = [
            (predict(model, apply_rot(x, m)) == y).float().mean().item() * 100
            for m in mats
        ]
        row[f"exp{exp}"] = float(np.mean(accs))
    rows.append(row)
    return pd.DataFrame(rows).round(2)


def as_markdown(df, index=False):
    """Markdown table, without pulling in `tabulate` for four lines of work."""
    df = df.reset_index() if index else df
    cells = [
        [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
        for row in df.itertuples(index=False)
    ]
    header = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in cells]
    return "\n".join(lines)


def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    splits = build_splits(cfg, device)
    names = [bu.ind2name[i] for i in cfg.labels_to_use]
    models = {exp: load_model(cfg, exp, device) for exp in cfg.model_exps}
    n_train, n_valid = len(splits["train"][1]), len(splits["valid"][1])
    print(
        f"{cfg.data_file}\n{n_train} train / {n_valid} valid bursts, "
        f"{len(names)} classes, seed {cfg.seed}\n"
    )

    print("## Accuracy\n")
    print(as_markdown(accuracy_table(cfg, splits, models, names)))

    print("\n## Per-class valid F1\n")
    print(as_markdown(per_class_table(cfg, splits, models, names, "plain"), index=True))
    print("\n## Per-class valid F1, class-balanced\n")
    print(
        as_markdown(per_class_table(cfg, splits, models, names, "balanced"), index=True)
    )

    print("\n## Orientation robustness (valid split, accelerometer frame perturbed)\n")
    print(as_markdown(robustness_table(cfg, splits, models, device)))

    # valid bursts per class, so a per-class swing can be read for what it is
    counts = pd.Series(splits["valid"][1].cpu().numpy()).value_counts().sort_index()
    print(
        "\nvalid bursts per class: "
        + ", ".join(f"{names[i]} {c}" for i, c in counts.items())
    )


if __name__ == "__main__":
    config = {
        "data_file": "/home/fatemeh/Downloads/bird/data/final/starts.csv",
        "save_path": "/home/fatemeh/Downloads/bird/results",
        "model_exps": [194, 196],  # baseline first, then the model of interest
        "labels_to_use": [0, 1, 2, 3, 4, 5, 6, 8, 9],
        "seed": 32984,
        "train_per": 0.9,
        "glen": 20,
        "flag_delta": 0.10,  # mark a class whose F1 moved at least this much
        "pitches": [20, 45, 70, 90],
        "n_so3": 20,
    }
    main(OmegaConf.create(config))
