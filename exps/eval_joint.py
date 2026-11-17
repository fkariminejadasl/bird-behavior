"""The joint models beside the single-species models, on the same valid segments.

The joint models (exp212 with rotation, exp213 without) are trained on both
species at once, on joint classes (`scripts/train_joint.py`). Each species
keeps the valid split of its own models:

- gull: the 438 valid bursts of exp197. Baselines: exp199, 201, 202, exp197's
  config at seeds 32984, 1234, 5678, at epoch 4000; and exp197 at its best
  epoch, the working model. All trained with rotation.
- spoonbill: the 505 valid segments of exp206-211, at epoch 4000, with
  rotation (206, 208, 209) and without (207, 210, 211).

The joint runs are read at epoch 4000. A joint prediction is right when it is
in the label's set; it is mapped back to the label whose set holds it, or to
none. A single-species prediction is right when it equals the label.

Prints three tables:

1. Valid accuracy, %, per species.
2. Valid F1 per label, 0-1, 2TP / (2TP + FP + FN): baselines as mean (min-max)
   over seeds, joint runs one each.
3. Where the joint models put the labels whose set holds more than one joint
   class (gull SitStand and Pecking, spoonbill fly-flap): segments per joint
   class, over all segments of the label (train and valid) and over valid
   only, and the joint class most often given outside the set. The loss lets
   the model choose freely inside the set, so the split inside it is what it
   learned from the other species, not from these labels.

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/eval_joint.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts"))
sys.path.insert(0, str(root / "exps"))
import batch_train_supervised as bts  # noqa: E402
import train_joint as tj  # noqa: E402
from eval_labeled import as_markdown  # noqa: E402

from behavior import model as bm  # noqa: E402


@torch.no_grad()
def predict(cfg, checkpoint, n_out, x, device):
    """Output index per segment."""
    params = {**cfg.model.parameters, "out_channels": n_out}
    model = bts.models[cfg.model.name](**params).to(device)
    bm.load_model(Path(cfg.save_path) / checkpoint, model, device)
    model.eval()
    return model(x).argmax(1).cpu().numpy()


def species_guess(cfg, species, checkpoint, x, device):
    """Label names a single-species model gives."""
    labels = cfg.species[species].labels
    names = np.array([labels[c] for c in sorted(labels)])
    return names[predict(cfg, checkpoint, len(names), x, device)]


def joint_guess(cfg, species, checkpoint, x, device):
    """Joint classes a joint model gives, and their labels of `species`."""
    joint = np.array(tj.joint_classes(cfg))
    pred = joint[predict(cfg, checkpoint, len(joint), x, device)]
    return pred, tj.back_to_label(cfg, species, pred)


def spread(vals):
    """'mean (min-max)', or the one value."""
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{np.mean(vals):.2f} ({min(vals):.2f}-{max(vals):.2f})"


def runs(cfg, species):
    """{column name: [(exp, checkpoint, is_joint)]} for one species."""
    out = {}
    for arm, exps in cfg.baselines[species].items():
        out[arm] = [(e, f"{e}_{cfg.epoch}.pth", False) for e in exps]
    if species in cfg.best_also:
        e = cfg.best_also[species]
        out[f"exp{e}, best epoch"] = [(e, f"{e}_best.pth", False)]
    for arm, e in cfg.joint.items():
        out[f"exp{e}, joint, {arm}"] = [(e, f"{e}_{cfg.epoch}.pth", True)]
    return out


def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = tj.load_species(cfg, device)
    acc_rows, f1_tables, set_rows = [], {}, {}
    for species, d in data.items():
        valid = d["valid"]
        order = list(cfg.species[species].labels.values())
        f1_cols = {}
        for col, members in runs(cfg, species).items():
            accs, f1s = [], []
            for e, ckpt, is_joint in members:
                if is_joint:
                    _, guess = joint_guess(cfg, species, ckpt, valid["x"], device)
                else:
                    guess = species_guess(cfg, species, ckpt, valid["x"], device)
                accs.append(100 * np.mean(guess == valid["names"]))
                f1s.append(tj.f1_table(valid["names"], guess, order)["F1"].values)
            exps = ", ".join(str(m[0]) for m in members)
            acc_rows.append(
                {
                    "species": species,
                    "model": col,
                    "exp": exps,
                    "valid segments": len(valid["y"]),
                    "accuracy %": spread(accs),
                }
            )
            f1s = np.array(f1s)
            f1_cols[col] = [
                spread(list(f1s[:, i])) if not np.isnan(f1s[:, i]).all() else "nan"
                for i in range(len(order))
            ]
        counts = pd.Series(valid["names"]).value_counts().reindex(order).fillna(0)
        t = pd.DataFrame({"label": order, "valid segments": counts.astype(int)})
        for col, vals in f1_cols.items():
            t[col] = vals
        f1_tables[species] = t

        # where the joint models put the labels that are sets
        all_x = torch.cat([d["train"]["x"], d["valid"]["x"]])
        all_names = np.concatenate([d["train"]["names"], d["valid"]["names"]])
        for label, members in cfg.species[species].to_joint.items():
            if len(members) < 2:
                continue
            for arm, e in cfg.joint.items():
                ckpt = f"{e}_{cfg.epoch}.pth"
                for which, x, names in (
                    ("all", all_x, all_names),
                    ("valid", valid["x"], valid["names"]),
                ):
                    pred, _ = joint_guess(cfg, species, ckpt, x, device)
                    pred = pred[names == label]
                    row = {
                        f"{species} {label}": f"exp{e}, {arm}",
                        "segments": f"{which}, {len(pred)}",
                    }
                    for m in members:
                        row[m] = int(np.sum(pred == m))
                    outside = pred[~np.isin(pred, list(members))]
                    row["outside the set"] = len(outside)
                    top = pd.Series(outside).value_counts()
                    row["most often outside"] = (
                        f"{top.index[0]} {top.iloc[0]}" if len(top) else "-"
                    )
                    set_rows.setdefault(label, []).append(row)

    print("\n## Valid accuracy, %\n")
    print(as_markdown(pd.DataFrame(acc_rows)))
    for species, t in f1_tables.items():
        print(f"\n## {species}: valid F1 per label, 0-1\n")
        print(as_markdown(t))
    print("\n## Where the joint models put a label of several joint classes\n")
    for rows in set_rows.values():
        print(as_markdown(pd.DataFrame(rows)) + "\n")


if __name__ == "__main__":
    config = {
        "save_path": "/home/fatemeh/Downloads/bird/results",
        "glen": 20,
        "train_per": 0.9,
        "split_seed": 32984,
        "add_magnitudes": True,
        # the model of every run here, as in scripts/train_joint.py
        "model": {
            "name": "BirdModelSmallDilated",
            "parameters": {"in_channels": 7, "mid_channels": 20, "dropout": 0.15},
        },
        "species": tj.species_config(),
        "joint_classes": tj.JOINT_CLASSES,
        "joint": {"rotation": 212, "no rotation": 213},
        "epoch": 4000,  # read <exp>_4000.pth
        "baselines": {
            "gull": {"exp197 config, rotation": [199, 201, 202]},
            "spoonbill": {"rotation": [206, 208, 209], "no rotation": [207, 210, 211]},
        },
        "best_also": {"gull": 197},  # the working gull model, at its best epoch
    }
    main(OmegaConf.create(config))
