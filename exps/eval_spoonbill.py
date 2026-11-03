"""Score the spoonbill models against the two published spoonbill papers.

The papers score pooled behaviours with the F-measure (F1), 2TP / (2TP + FP +
FN), on 0 to 1. This script pools our 14 labels the same way, from the
valid-split predictions of each run, and prints the paper values beside ours.

- Lok et al. 2023, Animal Biotelemetry 11:5: 9 classes (stand, sit,
  fly-active, fly-passive, walk, search, handle, ingest, drink), pooled as in
  `1_data_preparation/2_combine_annotated_data_with_soaring_flight_data.R` of
  its code (zenodo.org/records/7541131). The values are the best of all its
  segment lengths, from the text; per-length values are only in its Fig. 3.
  It has no begging, so beg is left out of this table; a beg prediction on
  another behaviour still counts as an error.
- Lok et al. 2025, Behav Ecol Sociobiol: 5 groups (forage, walk, fly, rest,
  beg), from its Table S2, at 0.8 and 1.6 s segments; ours are 1.0 s.

Two scorings:

- **Records, as the papers score.** Every valid fix is cut into 1 s windows
  of 20 records from its first record, whatever the labels in them, as
  `create.fixed.segments` of the 2023 code does. Each window is predicted, and
  each of its records is scored against its own label. So seconds with two
  behaviours and bouts shorter than 1 s are tested too. Windows with a missing
  value, fewer than 20 records or a speed of 30 m/s or more are skipped, as
  the papers skip short segments and the loader skips fast ones.
- **Segments, as training scores.** The valid segments of the training file:
  one behaviour each, so only bouts of 1 s or longer.

What still differs from the papers: they use a random forest, a 70/30 split
drawn per segment (so a recording can be in train and test) and the mean of
100 runs. Ours is one split by fix (no fix in train and valid), 505 segments,
training seeds varied with the split held still. Both have the same birds in
train and test.

Runs are read at `<exp>_4000.pth`. Each run's best checkpoint is first checked
to reproduce the valid accuracy in its `app_loss_acc.txt`, which shows the
split is rebuilt right. It also prints the training file per Lok et al. 2025
group (segments, fixes, birds) and how many valid segments of a random split
per segment (exp205) share a fix with train.

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/eval_spoonbill.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts"))
sys.path.insert(0, str(root / "scripts" / "data"))
sys.path.insert(0, str(root / "exps"))
import batch_train_supervised as bts  # noqa: E402
import prepare_spoonbill_data as psd  # noqa: E402
from eval_labeled import as_markdown  # noqa: E402

from behavior import data as bd  # noqa: E402
from behavior import model as bm  # noqa: E402
from behavior import utils as bu  # noqa: E402

STAND = ["stand-rest", "stand-alert", "stand-preen", "stand-other"]
STAND += ["stand-shake-feathers"]
GROUPS_2023 = {
    "stand": STAND,
    "sit": ["sit-alert"],
    "fly-active": ["fly-flap"],
    "fly-passive": ["fly-soar"],
    "walk": ["walk"],
    "search": ["for-search"],
    "handle": ["for-handle"],
    "ingest": ["for-intake"],
    "drink": ["drink"],
}
# Lok et al. 2023, best over segment lengths, from the text. sit, stand and
# both flights: "> 0.90" (sensitivity and precision 0.94-1.00). handle and
# drink: F could not be computed, as they were never predicted.
PAPER_2023 = {
    "stand": "> 0.90",
    "sit": "> 0.90",
    "fly-active": "> 0.90",
    "fly-passive": "> 0.90",
    "walk": "0.65",
    "search": "0.91",
    "handle": "none",
    "ingest": "0.73",
    "drink": "none",
}
GROUPS_2025 = {
    "forage": ["for-search", "for-handle", "for-intake"],
    "walk": ["walk"],
    "fly": ["fly-flap", "fly-soar"],
    "rest": ["sit-alert", "drink"] + STAND,
    "beg": ["beg"],
}
# Lok et al. 2025, Table S2: F-measure at 0.8 and 1.6 s segments
PAPER_2025 = {
    "forage": (0.96, 0.96),
    "walk": (0.60, 0.61),
    "fly": (0.97, 0.98),
    "rest": (0.96, 0.95),
    "beg": (0.86, 0.89),
}


def valid_split(cfg):
    """Valid inputs, labels and fixes of the fix split, as training builds them."""
    igs, ldts = bd.load_csv_pandas(cfg.data_file, cfg.labels_to_use, glen=cfg.glen)
    _, valid = bts.split_by_fix(cfg, ldts[:, 0], torch.device("cpu"))
    valid = valid.numpy()
    keys = bd.burst_keys(cfg.data_file, cfg.labels_to_use, glen=cfg.glen)
    fixes = {f"{keys[i][0]}_{keys[i][1]}" for i in valid}
    data = bd.BirdDataset(igs[valid], ldts[valid], add_magnitudes=True)
    x = torch.from_numpy(data.data).permute(0, 2, 1)  # N x C x L
    return x, ldts[valid, 0], fixes


def data_table(cfg):
    """Segments, fixes and birds per Lok et al. 2025 group, in the training file."""
    _, ldts = bd.load_csv_pandas(cfg.data_file, cfg.labels_to_use, glen=cfg.glen)
    keys = bd.burst_keys(cfg.data_file, cfg.labels_to_use, glen=cfg.glen)
    seg = pd.DataFrame(keys, columns=["bird", "datetime", "start"])
    names = [cfg.label_names[cfg.labels_to_use[int(i)]] for i in ldts[:, 0]]
    to_group = {n: g for g, members in GROUPS_2025.items() for n in members}
    seg["group"] = [to_group[n] for n in names]
    rows = []
    for group, df in seg.groupby("group", sort=False):
        top = df["bird"].value_counts()
        rows.append(
            {
                "group": group,
                "segments": len(df),
                "% of segments": 100 * len(df) / len(seg),
                "fixes": df.groupby(["bird", "datetime"]).ngroups,
                "birds": df["bird"].nunique(),
                "top bird": f"{top.index[0]} ({100 * top.iloc[0] / len(df):.0f}%)",
            }
        )
    fixes, birds = seg.groupby(["bird", "datetime"]).ngroups, seg["bird"].nunique()
    print(f"Training file: {len(seg)} segments, {fixes} fixes, {birds} birds\n")
    print(as_markdown(pd.DataFrame(rows)) + "\n")


def random_split_leak(cfg, device):
    """Valid segments of a random split per segment whose fix is also in train.

    This is the split of exp205 (`stratified_split` on the training device),
    and the reason the later runs split by fix.
    """
    _, ldts = bd.load_csv_pandas(cfg.data_file, cfg.labels_to_use, glen=cfg.glen)
    labels = torch.tensor(ldts[:, 0], device=device)
    split = [cfg.train_per, 1 - cfg.train_per]
    train, valid = bu.stratified_split(labels, split_ratios=split, seed=cfg.split_seed)
    keys = bd.burst_keys(cfg.data_file, cfg.labels_to_use, glen=cfg.glen)
    in_train = {keys[i][:2] for i in train.cpu().numpy()}
    shared = sum(keys[i][:2] in in_train for i in valid.cpu().numpy())
    print(
        f"Random split per segment: {shared} of {len(valid)} valid segments "
        f"share a fix with train segments\n"
    )


def record_windows(cfg, fixes):
    """Every full 1 s window of the valid fixes, cut from each fix's first record.

    Returns the windows as model input, N x C x L, and one row per record with
    its window number and label code.
    """
    frames = [psd.sds.drop_conflicts(df)[0] for df in psd.load_sources(cfg).values()]
    df = pd.concat(frames)[list(range(8))]
    df = df[(df[0].astype(str) + "_" + df[1]).isin(fixes)]
    df = df.sort_values([0, 1, 2]).reset_index(drop=True)
    df["window"] = (df[2] - df.groupby([0, 1])[2].transform("min")) // cfg.glen
    df["bad"] = df[psd.sds.SIGNAL].isna().any(axis=1) | (df[7] >= 30)
    w = df.groupby([0, 1, "window"]).agg(
        n=(2, "size"), lo=(2, "min"), hi=(2, "max"), bad=("bad", "any")
    )
    w["keep"] = (w["n"] == cfg.glen) & (w["hi"] - w["lo"] == cfg.glen - 1) & ~w["bad"]
    print(
        f"Record scoring: {df.groupby([0, 1]).ngroups} valid fixes, {len(df)} "
        f"records; {int(w['keep'].sum())} full windows of {cfg.glen} records "
        f"hold {int(w['keep'].sum()) * cfg.glen} of them"
    )
    df = df.merge(w["keep"].reset_index(), on=[0, 1, "window"])
    df = df[df["keep"]].sort_values([0, 1, 2]).reset_index(drop=True)
    df["window"] = np.repeat(np.arange(len(df) // cfg.glen), cfg.glen)
    igs = df[[4, 5, 6, 7]].values.reshape(-1, cfg.glen, 4)
    igs[:, :, :3] = igs[:, :, :3].clip(-2.0, 2.0)  # as load_csv_pandas
    mixed = (df.groupby("window")[3].nunique() > 1).sum()
    print(f"{mixed} of the windows hold two or more behaviours")
    data = bd.BirdDataset(igs, None, add_magnitudes=True)
    return torch.from_numpy(data.data).permute(0, 2, 1), df[["window", 3]]


@torch.no_grad()
def predict(cfg, checkpoint, x, device):
    model = bts.models[cfg.model.name](**cfg.model.parameters).to(device)
    bm.load_model(Path(cfg.save_path) / checkpoint, model, device)
    model.eval()
    return model(x.to(device)).argmax(1).cpu().numpy()


def file_accuracy(cfg, exp):
    """Valid accuracy written by training, from `app_loss_acc.txt`."""
    stem = Path(cfg.data_file).stem
    path = Path(cfg.save_path) / "failed" / f"{exp}_{stem}" / "app_loss_acc.txt"
    return float(re.search(r"valid: .*Accuracy: ([\d.]+)", path.read_text()).group(1))


def group_f1(true, guess, groups, drop=()):
    """F1 per group, 2TP / (2TP + FP + FN), from label names.

    `drop`: true labels left out. A label in no group is its own group, so
    predicting it is an error. Returns the F1s and the count per true group.
    """
    to_group = {n: g for g, members in groups.items() for n in members}
    true = np.array([to_group.get(n, n) for n in true])
    guess = np.array([to_group.get(n, n) for n in guess])
    keep = ~np.isin(true, list(drop))
    true, guess = true[keep], guess[keep]
    out = {}
    for g in groups:
        tp = np.sum((true == g) & (guess == g))
        fp = np.sum((true != g) & (guess == g))
        fn = np.sum((true == g) & (guess != g))
        out[g] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    return out, pd.Series(true).value_counts()


def summary(per_run, runs):
    """'mean (min-max)' over runs, per row."""
    df = per_run[runs]
    return [
        f"{m:.2f} ({lo:.2f}-{hi:.2f})"
        for m, lo, hi in zip(df.mean(axis=1), df.min(axis=1), df.max(axis=1))
    ]


def comparison(f1s, counts, unit, paper, paper_cols, arms):
    """One table: count per group, the paper's values, ours per arm."""
    t = pd.DataFrame(f1s)
    out = pd.DataFrame({f"valid {unit}": counts.reindex(t.index).fillna(0).astype(int)})
    for i, col in enumerate(paper_cols):
        out[col] = [
            paper[g][i] if isinstance(paper[g], tuple) else paper[g] for g in t.index
        ]
    for arm, runs in arms.items():
        out[f"ours 1 s, {arm}"] = summary(t, runs)
    return out


def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    names = [cfg.label_names[i] for i in cfg.labels_to_use]
    data_table(cfg)
    random_split_leak(cfg, device)
    x, y, fixes = valid_split(cfg)
    xw, records = record_windows(cfg, fixes)
    true_rec = [cfg.label_names[int(c)] for c in records[3]]
    print(f"Segment scoring: {len(y)} valid segments\n")

    rows = []
    f = {k: {} for k in ["rec23", "rec25", "seg23", "seg25", "seg14"]}
    n = {}
    for arm, exps in cfg.arms.items():
        for exp in exps:
            best = (predict(cfg, f"{exp}_best.pth", x, device) == y).mean() * 100
            written = file_accuracy(cfg, exp)
            assert abs(best - written) < 0.01, f"exp{exp}: {best:.2f} vs {written}"
            ckpt = f"{exp}_{cfg.epoch}.pth"
            pred = predict(cfg, ckpt, x, device)
            pred_w = predict(cfg, ckpt, xw, device)
            guess_rec = [names[i] for i in pred_w[records["window"].values]]
            true_seg, guess_seg = [names[i] for i in y], [names[i] for i in pred]
            rows.append(
                {
                    "exp": exp,
                    "arm": arm,
                    "best ckpt acc % (recomputed = file)": f"{best:.2f} = {written}",
                    f"acc % at epoch {cfg.epoch}, segments": (pred == y).mean() * 100,
                    f"acc % at epoch {cfg.epoch}, records": 100
                    * np.mean(np.array(true_rec) == np.array(guess_rec)),
                }
            )
            f["rec23"][exp], n["rec23"] = group_f1(
                true_rec, guess_rec, GROUPS_2023, drop=["beg"]
            )
            f["rec25"][exp], n["rec25"] = group_f1(true_rec, guess_rec, GROUPS_2025)
            f["seg23"][exp], n["seg23"] = group_f1(
                true_seg, guess_seg, GROUPS_2023, drop=["beg"]
            )
            f["seg25"][exp], n["seg25"] = group_f1(true_seg, guess_seg, GROUPS_2025)
            f["seg14"][exp], n["seg14"] = group_f1(
                true_seg, guess_seg, {k: [k] for k in names}
            )
    print(as_markdown(pd.DataFrame(rows)))

    arms = {arm: list(exps) for arm, exps in cfg.arms.items()}
    note = f"F1, 0-1, mean (min-max) over {len(arms['rotation'])} seeds"
    p25, p23 = ["paper 0.8 s", "paper 1.6 s"], ["paper, best length"]
    for unit, how in [("records", "rec"), ("segments", "seg")]:
        print(f"\n## Lok et al. 2025 groups, scored on {unit}, {note}\n")
        t = comparison(f[f"{how}25"], n[f"{how}25"], unit, PAPER_2025, p25, arms)
        print(as_markdown(t.rename_axis("group"), index=True))
        print(f"\n## Lok et al. 2023 classes, beg left out, scored on {unit}, {note}\n")
        t = comparison(f[f"{how}23"], n[f"{how}23"], unit, PAPER_2023, p23, arms)
        print(as_markdown(t.rename_axis("class"), index=True))

    print(f"\n## Our 14 labels, scored on segments, {note}\n")
    t = comparison(f["seg14"], n["seg14"], "segments", {}, [], arms)
    print(as_markdown(t.rename_axis("label"), index=True))
    print("\nnan: no valid example and never predicted, so F1 is undefined.")


if __name__ == "__main__":
    spoonbill_names = {
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
    config = {
        "save_path": "/home/fatemeh/Downloads/bird/results",
        "data_file": "/home/fatemeh/Downloads/bird/data/spoonbill_final/"
        "spoonbill_starts.csv",
        "labels_to_use": sorted(spoonbill_names),
        "label_names": spoonbill_names,
        "glen": 20,  # records per window, 1 s
        # the split of the runs: by fix, as split_by_fix in training
        "train_per": 0.9,
        "split_seed": 32984,
        # the raw records, for scoring on records; as prepare_spoonbill_data.py
        "data_dir": "/home/fatemeh/Downloads/bird/data/spoonbill",
        "video_file": "video_annotated_data.csv",
        "soar_file": "visual_annotated_passive_flight_data.csv",
        "begging_dir": "/home/fatemeh/Downloads/bird/data/spoonbill_postfledging/"
        "tamarlok-postfledging_care-6c40d9b/data/raw",
        "begging_file": "annotated.begging.data.csv",
        "begging_map": {
            "beg": 17,
            "beg_flap": 17,
            "preen_tag": 4,
            "shake_feathers": 15,
        },
        # the model of the runs, as in scripts/batch_train_supervised.py
        "model": {
            "name": "BirdModelSmallDilated",
            "parameters": {
                "in_channels": 7,
                "mid_channels": 20,
                "out_channels": len(spoonbill_names),
                "dropout": 0.15,
            },
        },
        # seeds 32984, 1234, 5678 per arm
        "arms": {"rotation": [206, 208, 209], "no rotation": [207, 210, 211]},
        "epoch": 4000,  # read <exp>_4000.pth, a fixed epoch for every run
    }
    main(OmegaConf.create(config))
