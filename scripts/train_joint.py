"""Train one model on gulls and spoonbills together, on joint classes.

The two label sets do not match one to one (`docs/descriptions.md`, Spoonbill
to gull labels). Gull SitStand is coarser than the spoonbill rest and alert
labels; spoonbill fly-flap is coarser than gull Flap, ExFlap and Manouvre. So
the model predicts joint classes, the finer of the two wherever they differ,
and each species' label is the set of joint classes it can be (`to_joint` in
the config).

The loss is cross-entropy on a set: minus the log of the probability summed
over the joint classes of the label's set (partial-label learning). For a set
of one class it is the plain cross-entropy. A gull SitStand burst is right as
stand-rest, sit-alert, stand-alert or drink, and the model is free to choose
among them; what it learns from the spoonbills decides which. So a set never
holds a joint class of one species only: a gull-only SitStand class in that
set would take every gull resting burst, and nothing would carry over.

The map follows the ecologist's table, with two choices made here:

- gull Pecking is for-handle or for-intake, not for-search: the table says
  gulls do not search that way.
- the preening, shaking and stand-other labels are not in gull SitStand: the
  table puts them under Other, which the gull data drops.

Each species keeps the split of its own models, so the valid sets are those of
exp197 (438 gull bursts, stratified, seed 32984) and exp206-211 (505 spoonbill
segments, by fix, split seed 32984). Before training, the best checkpoint of
`check_exp` must reproduce the valid accuracy in its `app_loss_acc.txt` on the
rebuilt split, or the script stops.

A prediction is right when it is in the label's set, so the valid accuracy of
each species compares with its single-species models on the same segments. The
best checkpoint is the epoch with the highest mean of the two valid
accuracies.

Writes, under `save_path`: `<exp>_best.pth`, `<exp>_<no_epochs>.pth`,
`tensorboard/<exp>/`, and in `failed/<exp>_joint/`: `app_loss_acc.txt`, and
per checkpoint and split `counts_<ckpt>_<split>.csv` (rows species and label,
columns joint class) and `f1_<ckpt>_<split>.csv` (F1 per species label, 0-1,
a joint class mapped back to the label whose set holds it), and each counts
CSV drawn as a confusion matrix, `counts_<ckpt>_<split>.png` (`plot_counts`).
Prints the valid counts of both checkpoints.

  /home/fatemeh/miniconda3/envs/bird/bin/python scripts/train_joint.py \\
    > /home/fatemeh/Downloads/bird/results/tensorboard/<exp>.txt 2>&1
"""

import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import batch_train_supervised as bts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf
from torch.utils import tensorboard

from behavior import data as bd
from behavior import data_augmentation as bau
from behavior import model as bm
from behavior import utils as bu

GULL_NAMES = {
    0: "Flap",
    1: "ExFlap",
    2: "Soar",
    3: "Boat",
    4: "Float",
    5: "SitStand",
    6: "TerLoco",
    8: "Manouvre",
    9: "Pecking",
}
SPOONBILL_NAMES = {
    2: "stand-rest",
    4: "stand-preen",
    6: "sit-alert",
    7: "stand-alert",
    8: "fly-flap",
    9: "fly-soar",
    10: "walk",
    11: "for-search",
    12: "for-handle",
    13: "for-intake",
    14: "stand-other",
    15: "stand-shake-feathers",
    16: "drink",
    17: "beg",
}
# Joint classes, the finer of the two species wherever they differ, and each
# label as the set of joint classes it can be. From the ecologist's table
# (exps/eval_cross_species.py::SPOONBILL_TO_GULL), read for training.
# exp212-213 were trained with these; a new map needs new names, so that
# exps/eval_joint.py still reads the old runs right.
JOINT_CLASSES = ["Flap", "ExFlap", "Manouvre", "Soar", "Float", "Boat"]
JOINT_CLASSES += ["stand-rest", "sit-alert", "stand-alert", "drink"]
JOINT_CLASSES += ["stand-preen", "stand-other", "stand-shake-feathers"]
JOINT_CLASSES += ["walk", "for-search", "for-handle", "for-intake", "beg"]
GULL_TO_JOINT = {
    "Flap": ["Flap"],
    "ExFlap": ["ExFlap"],
    "Soar": ["Soar"],
    "Boat": ["Boat"],
    "Float": ["Float"],
    "SitStand": ["stand-rest", "sit-alert", "stand-alert", "drink"],
    "TerLoco": ["walk"],
    "Manouvre": ["Manouvre"],
    "Pecking": ["for-handle", "for-intake"],
}
SPOONBILL_TO_JOINT = {n: [n] for n in SPOONBILL_NAMES.values()}
SPOONBILL_TO_JOINT["fly-flap"] = ["Flap", "ExFlap", "Manouvre"]
SPOONBILL_TO_JOINT["fly-soar"] = ["Soar"]


def species_config():
    """The two species as exp212-213 use them: data, split and joint sets.

    Each keeps the split of its own models: gull stratified per segment as
    exp197, spoonbill by fix as exp206-211. `check_exp` must reproduce its
    valid accuracy on the rebuilt split.
    """
    return {
        "gull": {
            "data_file": "/home/fatemeh/Downloads/bird/data/final/starts.csv",
            "labels": GULL_NAMES,
            "split_by": None,
            "check_exp": 197,
            "to_joint": GULL_TO_JOINT,
        },
        "spoonbill": {
            "data_file": "/home/fatemeh/Downloads/bird/data/spoonbill_final/"
            "spoonbill_starts.csv",
            "labels": SPOONBILL_NAMES,
            "split_by": "fix",
            "check_exp": 207,
            "to_joint": SPOONBILL_TO_JOINT,
        },
    }


def joint_classes(cfg):
    """Joint class names, in the order of the model's outputs."""
    return list(cfg.joint_classes)


def check_map(cfg):
    """Every set uses known joint classes, and one species' sets do not overlap.

    Without overlap, a joint class maps back to at most one label per species.
    """
    known = set(joint_classes(cfg))
    for name, sp in cfg.species.items():
        assert set(sp.labels.values()) == set(sp.to_joint), name
        seen = {}
        for label, members in sp.to_joint.items():
            for m in members:
                assert m in known, f"{name} {label}: unknown joint class {m}"
                assert m not in seen, f"{name}: {m} in {seen.get(m)} and {label}"
                seen[m] = label


def species_split(cfg, sp, labels, device):
    """Train and valid indices of one species, as its own models split it."""
    codes = sorted(sp.labels)
    if sp.split_by == "fix":
        scfg = OmegaConf.create(
            {
                "data_file": sp.data_file,
                "labels_to_use": codes,
                "train_per": cfg.train_per,
                "split_seed": cfg.split_seed,
            }
        )
        train, valid = bts.split_by_fix(scfg, labels, device)
    else:  # stratified per segment; seeded on the device, as training does
        y = torch.tensor(labels, device=device)
        ratios = [cfg.train_per, 1 - cfg.train_per]
        train, valid = bu.stratified_split(y, split_ratios=ratios, seed=cfg.split_seed)
    return train.cpu().numpy(), valid.cpu().numpy()


def file_accuracy(cfg, sp, exp):
    """Valid accuracy written by training, from `app_loss_acc.txt`."""
    stem = Path(sp.data_file).stem
    path = Path(cfg.save_path) / "failed" / f"{exp}_{stem}" / "app_loss_acc.txt"
    return float(re.search(r"valid: .*Accuracy: ([\d.]+)", path.read_text()).group(1))


@torch.no_grad()
def check_split(cfg, sp, x_valid, y_valid, device):
    """The split is the one of `check_exp` if its valid accuracy comes back."""
    model = bts.models[cfg.model.name](
        **{**cfg.model.parameters, "out_channels": len(sp.labels)}
    ).to(device)
    bm.load_model(Path(cfg.save_path) / f"{sp.check_exp}_best.pth", model, device)
    model.eval()
    acc = (model(x_valid).argmax(1).cpu().numpy() == y_valid).mean() * 100
    written = file_accuracy(cfg, sp, sp.check_exp)
    assert abs(acc - written) < 0.01, f"exp{sp.check_exp}: {acc:.2f} vs {written}"
    print(f"Split check: exp{sp.check_exp} valid accuracy {acc:.2f} = {written}")


def load_species(cfg, device):
    """Per species and split: model input, label names and set masks.

    Returns {species: {split: dict(x, names, mask)}}; x is N x C x L on
    `device`, mask N x K bool, K the joint classes.
    """
    joint = joint_classes(cfg)
    out = {}
    for name, sp in cfg.species.items():
        codes = sorted(sp.labels)
        igs, ldts = bd.load_csv_pandas(sp.data_file, codes, glen=cfg.glen)
        assert len(np.unique(ldts[:, 0])) == len(codes), sp.data_file
        y = ldts[:, 0].astype(int)
        train, valid = species_split(cfg, sp, y, device)
        names = np.array([sp.labels[codes[i]] for i in y])
        mask = np.array([[j in sp.to_joint[n] for j in joint] for n in names])
        data = bd.BirdDataset(igs, None, add_magnitudes=cfg.add_magnitudes)
        x = torch.from_numpy(data.data).permute(0, 2, 1).contiguous()
        out[name] = {}
        for split, idx in (("train", train), ("valid", valid)):
            out[name][split] = {
                "x": x[idx].to(device),
                "y": y[idx],
                "names": names[idx],
                "mask": torch.from_numpy(mask[idx]).to(device),
            }
        n_tr, n_va = len(train), len(valid)
        print(f"{name}: {n_tr} train, {n_va} valid segments, {sp.data_file}")
        check_split(cfg, sp, out[name]["valid"]["x"], y[valid], device)
    return out


def set_loss(logits, mask):
    """Minus the log of the probability summed over each segment's set."""
    allowed = logits.masked_fill(~mask, float("-inf"))
    return (torch.logsumexp(logits, 1) - torch.logsumexp(allowed, 1)).mean()


def set_correct(logits, mask):
    """How many predictions fall in their label's set."""
    pred = logits.argmax(1)
    return mask[torch.arange(len(pred), device=pred.device), pred].sum().item()


@torch.no_grad()
def score(model, part):
    """Loss and accuracy in %, the prediction right when in the label's set."""
    model.eval()
    logits = model(part["x"])
    n = len(part["y"])
    return (
        set_loss(logits, part["mask"]).item(),
        100 * set_correct(logits, part["mask"]) / n,
    )


def refuse_to_overwrite(cfg):
    """Stop if this experiment number already has results; see bts."""
    if cfg.overwrite:
        return
    save = Path(cfg.save_path)
    taken = [
        p
        for p in (
            save / f"{cfg.exp}_best.pth",
            save / "failed" / f"{cfg.exp}_joint",
            save / "tensorboard" / str(cfg.exp),
        )
        if p.exists()
    ]
    if taken:
        raise FileExistsError(f"exp {cfg.exp} already has results: {taken}")


def train(cfg, data, device):
    """Train; save the best and the last epoch. Returns the best epoch."""
    bu.set_seed(cfg.seed)
    save = Path(cfg.save_path)
    xs = torch.cat([d["train"]["x"] for d in data.values()])
    masks = torch.cat([d["train"]["mask"] for d in data.values()])
    transform = bau.BatchRandomRotation3D() if cfg.augment else None
    batch = len(xs) if cfg.batch_size is None else cfg.batch_size

    def batches():
        """Shuffled batches on the GPU, rotated per batch, as bd.GpuBatches."""
        order = torch.randperm(len(xs), device=device)
        for start in range(0, len(xs), batch):
            sel = order[start : start + batch]
            x = xs[sel] if transform is None else transform(xs[sel])
            yield x, masks[sel]

    model = bts.models[cfg.model.name](
        **{**cfg.model.parameters, "out_channels": len(cfg.joint_classes)}
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.step_size, gamma=0.1
    )
    print(
        f"{cfg.model.name}, {len(cfg.joint_classes)} joint classes, rotation "
        f"{cfg.augment}, {len(xs)} train segments, start {datetime.now()}"
    )

    best, best_epoch = -1.0, 0
    with tensorboard.SummaryWriter(save / f"tensorboard/{cfg.exp}") as writer:
        for epoch in range(1, cfg.no_epochs + 1):
            model.train()
            total, correct, n = 0.0, 0, 0
            for x, m in batches():
                optimizer.zero_grad()
                logits = model(x)
                loss = set_loss(logits, m)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(x)
                correct += set_correct(logits.detach(), m)
                n += len(x)
            scheduler.step()
            writer.add_scalar("loss/train", total / n, epoch)
            writer.add_scalar("accuracy/train", 100 * correct / n, epoch)
            accs = []
            for name, d in data.items():
                loss_v, acc_v = score(model, d["valid"])
                writer.add_scalar(f"loss/valid_{name}", loss_v, epoch)
                writer.add_scalar(f"accuracy/valid_{name}", acc_v, epoch)
                accs.append(acc_v)
            mean = float(np.mean(accs))
            writer.add_scalar("accuracy/valid_mean", mean, epoch)
            if mean > best:
                best, best_epoch = mean, epoch
                bm.save_model(save, cfg.exp, epoch, model, optimizer, scheduler, True)
            if epoch % cfg.print_every == 0 or epoch == 1:
                valid = ", ".join(f"{k} {a:.2f}" for k, a in zip(data, accs))
                print(
                    f"epoch {epoch}: train loss {total / n:.4f}, accuracy "
                    f"{100 * correct / n:.2f}; valid {valid}; best mean "
                    f"{best:.2f} at {best_epoch}"
                )
    bm.save_model(save, cfg.exp, cfg.no_epochs, model, optimizer, scheduler)
    print(f"end {datetime.now()}, best epoch {best_epoch}")
    return best_epoch


def back_to_label(cfg, species, joint_names):
    """Each joint class as the label of `species` whose set holds it, or none."""
    to_label = {m: lab for lab, ms in cfg.species[species].to_joint.items() for m in ms}
    return np.array([to_label.get(j, "none") for j in joint_names])


def f1_table(true, guess, labels):
    """F1 per label, 2TP / (2TP + FP + FN), 0-1; guess may be none."""
    rows = []
    for lab in labels:
        tp = np.sum((true == lab) & (guess == lab))
        fp = np.sum((true != lab) & (guess == lab))
        fn = np.sum((true == lab) & (guess != lab))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        rows.append({"label": lab, "segments": int(np.sum(true == lab)), "F1": f1})
    return pd.DataFrame(rows)


@torch.no_grad()
def report(cfg, data, device, best_epoch):
    """Counts, F1 and accuracy per species, split and checkpoint, to `failed/`."""
    save = Path(cfg.save_path)
    out = save / "failed" / f"{cfg.exp}_joint"
    out.mkdir(parents=True, exist_ok=True)
    joint = np.array(joint_classes(cfg))
    model = bts.models[cfg.model.name](
        **{**cfg.model.parameters, "out_channels": len(joint)}
    ).to(device)
    lines = []
    ckpts = {"best": f"{cfg.exp}_best.pth", "final": f"{cfg.exp}_{cfg.no_epochs}.pth"}
    for ckpt, file in ckpts.items():
        bm.load_model(save / file, model, device)
        model.eval()
        epoch = best_epoch if ckpt == "best" else cfg.no_epochs
        for split in ("train", "valid"):
            counts = []
            for name, d in data.items():
                part = d[split]
                loss, acc = score(model, part)
                lines.append(
                    f"{ckpt} (epoch {epoch}), {name} {split}: Loss: {loss:.4f}, "
                    f"Accuracy: {acc:.2f}, segments: {len(part['y'])}"
                )
                pred = joint[model(part["x"]).argmax(1).cpu().numpy()]
                t = pd.crosstab(
                    part["names"],
                    pd.Categorical(pred, joint),
                    dropna=False,
                    colnames=["joint class"],
                )
                order = list(cfg.species[name].labels.values())
                t = t.reindex(order).fillna(0).astype(int)
                t.index = pd.MultiIndex.from_product([[name], t.index])
                counts.append(t)
                f1 = f1_table(part["names"], back_to_label(cfg, name, pred), order)
                f1.to_csv(out / f"f1_{ckpt}_{split}_{name}.csv", index=False)
            table = pd.concat(counts).rename_axis(["species", "label"])
            table.to_csv(out / f"counts_{ckpt}_{split}.csv")
            if split == "valid":
                print(f"\n{ckpt} (epoch {epoch}), valid, segments per joint class:")
                print(table.loc[:, table.sum() > 0].to_string())
    (out / "app_loss_acc.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    plot_run(cfg, cfg.exp)


def plot_counts(cfg, csv_path, title):
    """Draw a counts CSV of `report` as a confusion matrix, a PNG beside it.

    One panel per species: rows its labels, columns the joint classes, in the
    order of `joint_classes`. The number is segments; the color is the share
    of the row, so small labels show too. An orange box marks the joint
    classes of the label's set, the ones counted right, and the panel title
    gives the accuracy that follows from them.
    """
    joint = joint_classes(cfg)
    table = pd.read_csv(csv_path, index_col=[0, 1])[joint]
    species = list(cfg.species)
    sizes = [len(table.loc[s]) for s in species]
    fig, axes = plt.subplots(
        len(species),
        1,
        figsize=(0.5 * len(joint) + 4, 0.34 * sum(sizes) + 3),
        height_ratios=sizes,
        sharex=True,
        layout="constrained",
    )
    for ax, s in zip(axes, species):
        sets = cfg.species[s].to_joint
        t = table.loc[s]
        order = sorted(t.index, key=lambda lab: min(joint.index(m) for m in sets[lab]))
        counts = t.loc[order].values
        n = counts.sum(1)
        share = 100 * counts / np.maximum(n, 1)[:, None]
        im = ax.imshow(share, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        right = 0
        for r, lab in enumerate(order):
            for c, j in enumerate(joint):
                if j in sets[lab]:
                    right += counts[r, c]
                    box = Rectangle(
                        (c - 0.5, r - 0.5), 1, 1, fill=False, ec="#eb6834", lw=2
                    )
                    ax.add_patch(box)
                if counts[r, c]:
                    ink = "white" if share[r, c] > 60 else "black"
                    ax.text(
                        c,
                        r,
                        counts[r, c],
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=ink,
                    )
        ax.set_yticks(range(len(order)), [f"{lab} ({k})" for lab, k in zip(order, n)])
        ax.set_ylabel("true label (segments)")
        ax.set_title(
            f"{s}: {n.sum()} segments, {100 * right / n.sum():.2f}% in their "
            "label's set",
            loc="left",
            fontsize=10,
        )
    axes[-1].set_xticks(
        range(len(joint)), joint, rotation=45, ha="right", rotation_mode="anchor"
    )
    axes[-1].set_xlabel("predicted joint class")
    fig.colorbar(im, ax=axes, shrink=0.5, label="share of the row's segments, %")
    fig.suptitle(
        f"{title}\nnumber: segments; orange box: the joint classes of the "
        "label's set, counted right"
    )
    fig.savefig(Path(csv_path).with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_run(cfg, exp):
    """A confusion matrix PNG for each counts CSV of one run (`plot_counts`)."""
    save = Path(cfg.save_path)
    ckpts = {"best": f"{exp}_best.pth", "final": f"{exp}_{cfg.no_epochs}.pth"}
    for ckpt, file in ckpts.items():
        epoch = torch.load(save / file, map_location="cpu", weights_only=True)["epoch"]
        for split in ("train", "valid"):
            title = f"exp{exp}, {split} split, {ckpt} checkpoint (epoch {epoch})"
            csv = save / "failed" / f"{exp}_joint" / f"counts_{ckpt}_{split}.csv"
            plot_counts(cfg, csv, title)


def main(cfg):
    refuse_to_overwrite(cfg)
    check_map(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = load_species(cfg, device)
    best_epoch = train(cfg, data, device)
    report(cfg, data, device, best_epoch)


if __name__ == "__main__":
    base_config = {
        "save_path": "/home/fatemeh/Downloads/bird/results",
        "exp": 212,
        "overwrite": False,  # True to replace the results of an experiment number
        "seed": 32984,
        "split_seed": 32984,  # both species keep the split of their own models
        "train_per": 0.9,
        "glen": 20,
        "species": species_config(),  # data, split and joint sets per species
        "joint_classes": JOINT_CLASSES,
        "add_magnitudes": True,  # 7 channels, as exp197 and exp206-211
        "augment": True,  # random 3D rotation of the acceleration
        # Training, as exp197: full batch, AdamW, StepLR
        "no_epochs": 4000,
        "batch_size": None,  # None is the whole train set in one batch
        "max_lr": 3e-4,
        "weight_decay": 1e-2,
        "step_size": 2000,
        "print_every": 100,
        "model": {
            "name": "BirdModelSmallDilated",
            "parameters": {"in_channels": 7, "mid_channels": 20, "dropout": 0.15},
        },
    }
    # exp212: rotation, exp213: no rotation; seed 32984 each.
    experiments = [{"exp": 212}, {"exp": 213, "augment": False}]

    configs = [
        OmegaConf.merge(OmegaConf.create(deepcopy(base_config)), e) for e in experiments
    ]
    for cfg in configs:  # every number free before anything trains
        refuse_to_overwrite(cfg)
    for cfg in configs:
        print(f"\n# Experiment {cfg.exp}")
        main(cfg)
