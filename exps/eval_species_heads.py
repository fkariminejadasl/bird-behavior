"""The species-heads models beside the single-species models, same valid segments.

The species-heads models (`scripts/train_species_heads.py`) have a coarse
head, a fine head per species and a species token. Each species keeps the
valid split of its own models: the 438 gull bursts of exp197 and the 505
spoonbill segments of exp206-211. All runs are read at epoch 4000, the last;
exp197 also at its best epoch, the working gull model. Baselines:

- gull: exp199, 201, 202, exp197's config at seeds 32984, 1234, 5678, all
  with rotation.
- spoonbill: exp206, 208, 209 with rotation, exp207, 210, 211 without.

Prints three parts:

1. Valid accuracy %, fine and coarse, per species. A single-species model's
   coarse accuracy maps its fine prediction to its group in `COARSE` of
   `scripts/train_species_heads.py`. The species-heads models with the token
   given and dropped.
2. Valid F1 per label, fine, 0-1, 2TP / (2TP + FP + FN), token given.
3. Each species through the other species' fine head, token dropped, over all
   segments (train and valid), for the first run of each arm: per label, the
   other species' classes it gets, most frequent first. The cross test of
   `exps/eval_cross_species.py` with features trained on both species. No
   label checks it.
4. Part 3 for the labels in `track`, every run: does the translation hold
   over seeds?

Several runs of one arm show as mean (min-max).

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/eval_species_heads.py
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
import eval_joint as ej  # noqa: E402
import train_joint as tj  # noqa: E402
import train_species_heads as ths  # noqa: E402
from eval_labeled import as_markdown  # noqa: E402

from behavior import model as bm  # noqa: E402


def baseline_cfg(cfg):
    """`cfg` with the single-species model, for `eval_joint.species_guess`."""
    out = OmegaConf.to_container(cfg)
    out["model"] = out["check_model"]
    return OmegaConf.create(out)


@torch.no_grad()
def heads_guess(cfg, species, checkpoint, part, token, device):
    """Fine class names and coarse group indices of a species-heads model."""
    model = ths.build_model(cfg, device)
    bm.load_model(Path(cfg.save_path) / checkpoint, model, device)
    coarse, fine = ths.predict(model, part, species, token)
    classes = np.array(ths.fine_classes(cfg, species))
    return classes[fine.cpu().numpy()], coarse.cpu().numpy()


def columns(cfg, species):
    """{column: [(exp, checkpoint, token)]}; token None for a baseline."""
    out = {}
    for arm, exps in cfg.baselines[species].items():
        out[arm] = [(e, f"{e}_{cfg.epoch}.pth", None) for e in exps]
    if species in cfg.best_also:
        e = cfg.best_also[species]
        out[f"exp{e}, best epoch"] = [(e, f"{e}_best.pth", None)]
    for arm, exps in cfg.heads.items():
        for token in ("given", "dropped"):
            out[f"{arm}, token {token}"] = [
                (e, f"{e}_{cfg.epoch}.pth", token) for e in exps
            ]
    return out


@torch.no_grad()
def other_head(cfg, e, data, device):
    """Part 3 for one run: per label, the other species' classes it gets."""
    model = ths.build_model(cfg, device)
    bm.load_model(Path(cfg.save_path) / f"{e}_{cfg.epoch}.pth", model, device)
    model.eval()
    rows = []
    for species, d in data.items():
        other = next(s for s in data if s != species)
        classes = np.array(ths.fine_classes(cfg, other))
        x = torch.cat([d["train"]["x"], d["valid"]["x"]])
        names = np.concatenate([d["train"]["names"], d["valid"]["names"]])
        unknown = torch.full((len(names),), -1, device=device)
        pick = classes[model(x, unknown)[other].argmax(1).cpu().numpy()]
        for label in ths.by_group(cfg, ths.fine_classes(cfg, species)):
            top = pd.Series(pick[names == label]).value_counts()
            rows.append(
                {
                    "species": species,
                    "label": label,
                    "segments": int(np.sum(names == label)),
                    "classes of the other species' head": ", ".join(
                        f"{k} {v}" for k, v in top.items()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = ths.load(cfg, device)
    base = baseline_cfg(cfg)
    groups = list(cfg.coarse)
    to_group = {n: groups.index(g) for g, names in cfg.coarse.items() for n in names}
    acc_rows, f1_tables = [], {}
    for species, d in data.items():
        valid = d["valid"]
        order = ths.by_group(cfg, ths.fine_classes(cfg, species))
        true_coarse = valid["coarse"].cpu().numpy()
        f1_cols = {}
        for col, members in columns(cfg, species).items():
            fine_acc, coarse_acc, f1s = [], [], []
            for e, ckpt, token in members:
                if token is None:
                    guess = ej.species_guess(base, species, ckpt, valid["x"], device)
                    coarse = np.array([to_group.get(g, -1) for g in guess])
                else:
                    guess, coarse = heads_guess(
                        cfg, species, ckpt, valid, token, device
                    )
                fine_acc.append(100 * np.mean(guess == valid["names"]))
                coarse_acc.append(100 * np.mean(coarse == true_coarse))
                f1s.append(tj.f1_table(valid["names"], guess, order)["F1"].values)
            acc_rows.append(
                {
                    "species": species,
                    "model": col,
                    "exp": ", ".join(str(m[0]) for m in members),
                    "valid segments": len(valid["names"]),
                    "fine accuracy %": ej.spread(fine_acc),
                    "coarse accuracy %": ej.spread(coarse_acc),
                }
            )
            if token in (None, "given"):
                f1s = np.array(f1s)
                f1_cols[col.replace(", token given", "")] = [
                    (
                        ej.spread(list(f1s[:, i]))
                        if not np.isnan(f1s[:, i]).all()
                        else "nan"
                    )
                    for i in range(len(order))
                ]
        counts = pd.Series(valid["names"]).value_counts().reindex(order).fillna(0)
        t = pd.DataFrame({"label": order, "valid segments": counts.astype(int)})
        for col, vals in f1_cols.items():
            t[col] = vals
        f1_tables[species] = t

    print(f"\n## Valid accuracy %, epoch {cfg.epoch} unless named\n")
    print(as_markdown(pd.DataFrame(acc_rows)))
    for species, t in f1_tables.items():
        print(f"\n## {species}: valid F1 per label, fine, 0-1, token given\n")
        print(as_markdown(t))
    tracked = []
    for arm, exps in cfg.heads.items():
        for i, e in enumerate(exps):
            t = other_head(cfg, int(e), data, device)
            if i == 0:
                print(
                    f"\n## exp{e} ({arm}): through the other species' head, token dropped\n"
                )
                print(as_markdown(t))
            for species, labels in cfg.track.items():
                rows = t[(t["species"] == species) & t["label"].isin(list(labels))]
                tracked.append(rows.assign(run=f"exp{e}, {arm}"))
    print("\n## Tracked labels through the other species' head, every run\n")
    t = pd.concat(tracked).sort_values(["label", "run"], kind="stable")
    print(as_markdown(t[["label", "run", "segments"] + list(t.columns[3:-1])]))


if __name__ == "__main__":
    config = {
        "save_path": "/home/fatemeh/Downloads/bird/results",
        "glen": 20,
        "train_per": 0.9,
        "split_seed": 32984,
        "add_magnitudes": True,
        "species": tj.species_config(tj.GULL_TO_SEPARATE, tj.SPOONBILL_TO_SEPARATE),
        "coarse": ths.COARSE,
        "drop": ["drink"],
        # the species-heads model, as in scripts/train_species_heads.py
        "model": {
            "name": "BirdModelSpeciesHeads",
            "parameters": {
                "in_channels": 7,
                "n_species": 2,
                "mid_channels": 20,
                "dropout": 0.15,
            },
        },
        # the single-species model of the baselines and the split check
        "check_model": {
            "name": "BirdModelSmallDilated",
            "parameters": {"in_channels": 7, "mid_channels": 20, "dropout": 0.15},
        },
        # arm: runs; the first run of each arm gets part 3
        "heads": {
            "heads, rotation": [216, 218, 220],
            "heads, no rotation": [217, 219, 221],
        },
        "epoch": 4000,  # read <exp>_4000.pth
        "baselines": {
            "gull": {"exp197 config, rotation": [199, 201, 202]},
            "spoonbill": {"rotation": [206, 208, 209], "no rotation": [207, 210, 211]},
        },
        "best_also": {"gull": 197},
        # labels whose translation part 4 follows over every run
        "track": {"gull": ["SitStand", "TerLoco"]},
    }
    main(OmegaConf.create(config))
