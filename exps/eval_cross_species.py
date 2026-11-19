"""Each species' model on the other species' data: which classes carry over.

- exp197, the gull model, on all 5080 segments of `spoonbill_starts.csv`, read
  at `197_best.pth`, its only checkpoint.
- The spoonbill models on all 4338 bursts of `starts.csv`: exp207, 210, 211
  (no rotation) and exp206, 208, 209 (rotation), seeds 32984, 1234, 5678, read
  at `<exp>_4000.pth`.

Every segment is used, not only a valid split: neither model saw the other
species.

What counts as right comes from a table by Julia, an ecologist, received
2026-11-16: `/home/fatemeh/Downloads/behaviours_spoonbills_to_gulls.xlsx`,
copied into `SPOONBILL_TO_GULL`. Per spoonbill label it gives the gull class she
expects the gull model to give, an alternative it may give, and the general
activity of the gull classes, from Table S1 of the Judy supplement. The
spoonbill models are scored with the same table read backwards: a gull class
should get the spoonbill labels that map to it.

Three scores per class, in % of its segments (gull: bursts):

- expected: the prediction is a class the table expects.
- expected or alternative: or a class it gives as the alternative.
- same general activity: the prediction has the same general activity.

A score is `-` where the table names no class the model has:

- stand-preen, stand-other and stand-shake-feathers have no expected gull
  class, only the alternative SitStand.
- Float and Boat have no spoonbill counterpart (X in the table). Boat's general
  activity, Sit-Stand, has one.
- beg is not in the table: it is shown but not scored, and a gull burst
  predicted as beg counts as wrong.
- sit-rest, sit-preen and washing are in the table but have no data.

The `all scored` row pools the classes that have a score: hits over their
segments.

Two checks first. The mean x, y and z of the acceleration per class of both
species: gravity is along +z in both, so the tags share the axis convention,
and a model trained without rotation is not wrong on the other species for that
reason alone. And each model's accuracy on all segments of its own species,
mostly its training data: high only if the label order is read right.

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/eval_cross_species.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "exps"))
from eval_labeled import as_markdown  # noqa: E402

from behavior import data as bd  # noqa: E402
from behavior import model as bm  # noqa: E402

LEVELS = ["expected", "expected or alternative", "same general activity"]

# General activity of each gull class, Table S1 of the Judy supplement, as in
# column General_activity_gulls of Julia's table.
GULL_GENERAL = {
    "Flap": "Flight",
    "ExFlap": "Flight",
    "Soar": "Flight",
    "Boat": "Sit-Stand",
    "Float": "Float",
    "SitStand": "Sit-Stand",
    "TerLoco": "Terrestrial locomotion",
    "Manouvre": "Flight",
    "Pecking": "Terrestrial locomotion",
}

# Julia's table, spoonbill label: (expected gull classes, alternative gull
# classes, general activity, None where there is none). Her names in our
# names: Stationary is SitStand, Walk TerLoco, Peck Pecking, Mixed Manouvre.
# fly-flap takes three rows of her table.
SPOONBILL_TO_GULL = {
    "stand-rest": (["SitStand"], [], "Sit-Stand"),
    "stand-preen": ([], ["SitStand"], None),
    "sit-alert": (["SitStand"], [], "Sit-Stand"),
    "stand-alert": (["SitStand"], [], "Sit-Stand"),
    "fly-flap": (["Flap", "ExFlap", "Manouvre"], [], "Flight"),
    "fly-soar": (["Soar"], [], "Flight"),
    "walk": (["TerLoco"], [], "Terrestrial locomotion"),
    "for-search": (["Pecking"], ["TerLoco"], "Terrestrial locomotion"),
    "for-handle": (["Pecking"], [], "Terrestrial locomotion"),
    "for-intake": (["Pecking"], [], "Terrestrial locomotion"),
    "stand-other": ([], ["SitStand"], None),
    "stand-shake-feathers": ([], ["SitStand"], None),
    "drink": (["SitStand"], ["Pecking"], "Sit-Stand"),
}


def gull_targets():
    """Per spoonbill label, the gull classes counted right at each level."""
    return {
        label: {
            "expected": exp,
            "expected or alternative": exp + alt,
            "same general activity": [
                g for g, act in GULL_GENERAL.items() if act == general
            ],
        }
        for label, (exp, alt, general) in SPOONBILL_TO_GULL.items()
    }


def spoonbill_targets():
    """Per gull class, the spoonbill labels counted right: the table backwards."""
    out = {}
    for gull, general in GULL_GENERAL.items():
        exp = [s for s, (e, _, _) in SPOONBILL_TO_GULL.items() if gull in e]
        alt = [s for s, (_, a, _) in SPOONBILL_TO_GULL.items() if gull in a]
        gen = [s for s, (_, _, g) in SPOONBILL_TO_GULL.items() if g == general]
        out[gull] = {
            "expected": exp,
            "expected or alternative": exp + alt,
            "same general activity": gen,
        }
    return out


def load(data_file, labels, names):
    """Model input N x 7 x L, as training builds it, and the class name per segment."""
    igs, ldts = bd.load_csv_pandas(data_file, labels)
    # load_csv_pandas numbers the labels present, so all must be there
    assert len(np.unique(ldts[:, 0])) == len(labels), data_file
    data = bd.BirdDataset(igs, ldts, add_magnitudes=True)
    x = torch.from_numpy(data.data).permute(0, 2, 1)
    return x, np.array([names[int(i)] for i in ldts[:, 0]]), igs


@torch.no_grad()
def predict(save_path, checkpoint, names, x, device):
    """Class name per segment."""
    model = bm.BirdModelSmallDilated(7, 20, len(names)).to(device)
    bm.load_model(Path(save_path) / checkpoint, model, device)
    model.eval()
    return np.array(names)[model(x.to(device)).argmax(1).cpu().numpy()]


def axis_table(species):
    """Mean x, y, z and |a| in g, and median GPS speed, per class of each species."""
    rows = []
    for name, (igs, true, order) in species.items():
        for c in order:
            a = igs[true == c]
            acc = a[:, :, :3]
            rows.append(
                {
                    "species": name,
                    "class": c,
                    "segments": len(a),
                    "mean x": acc[..., 0].mean(),
                    "mean y": acc[..., 1].mean(),
                    "mean z": acc[..., 2].mean(),
                    "mean abs a": np.linalg.norm(acc, axis=-1).mean(),
                    "median speed m/s": np.median(a[:, 0, 3]),
                }
            )
    return pd.DataFrame(rows)


def cross_table(true, guess, true_order, guess_order):
    """Counts: rows the true class, columns the predicted class."""
    t = pd.crosstab(
        pd.Categorical(true, true_order), pd.Categorical(guess, guess_order)
    )
    t.insert(0, "total", t.sum(axis=1))
    return t.rename_axis(index="true", columns=None)


def hits(true, guess, targets, order):
    """Per true class and level: (hits, count), or None where nothing can hit."""
    out = {}
    for c in order:
        m = true == c
        for level in LEVELS:
            ok = targets.get(c, {}).get(level, [])
            out[c, level] = (
                (int(np.isin(guess[m], ok).sum()), int(m.sum())) if ok else None
            )
    return out


def pooled(h, order, level):
    """Hits over segments, over the classes that have a score at this level."""
    pairs = [h[c, level] for c in order if h[c, level] is not None]
    return sum(p[0] for p in pairs), sum(p[1] for p in pairs)


def fmt(pairs):
    """% right from (hits, count) per run: one value, or 'mean (min-max)'."""
    if pairs[0] is None or pairs[0][1] == 0:
        return "-"
    vals = [100 * k / n for k, n in pairs]
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{np.mean(vals):.1f} ({min(vals):.1f}-{max(vals):.1f})"


def score_table(runs, true, order, targets, unit):
    """% right per true class and level. runs: one `hits` dict per run.

    The last row pools the classes that have a score, with their count.
    """
    rows = []
    for c in order:
        t = targets.get(c, {})
        exp = t.get("expected", [])
        alt = [a for a in t.get("expected or alternative", []) if a not in exp]
        row = {"true": c, unit: int((true == c).sum())}
        row["expected"] = ", ".join(exp) or "-"
        row["alternative"] = ", ".join(alt) or "-"
        for level in LEVELS:
            row[f"% {level}"] = fmt([h[c, level] for h in runs])
        rows.append(row)
    row = {"true": "all scored", unit: "", "expected": "", "alternative": ""}
    for level in LEVELS:
        pairs = [pooled(h, order, level) for h in runs]
        row[f"% {level}"] = f"{fmt(pairs)} of {pairs[0][1]}"
    rows.append(row)
    return pd.DataFrame(rows)


def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gull_names = [cfg.gull_names[i] for i in cfg.gull_labels]
    sb_names = [cfg.spoonbill_names[i] for i in cfg.spoonbill_labels]
    xg, yg, igs_g = load(cfg.gull_file, cfg.gull_labels, gull_names)
    xs, ys, igs_s = load(cfg.spoonbill_file, cfg.spoonbill_labels, sb_names)
    print(f"gull: {len(yg)} bursts, spoonbill: {len(ys)} segments, of 20 records\n")

    print("## Mean acceleration per class, in g, all segments\n")
    species = {"gull": (igs_g, yg, gull_names), "spoonbill": (igs_s, ys, sb_names)}
    print(as_markdown(axis_table(species)) + "\n")

    print("## Check: accuracy % on all segments of its own species, mostly train\n")
    own = {cfg.gull_exp: (cfg.gull_checkpoint, gull_names, xg, yg)}
    for exps in cfg.spoonbill_arms.values():
        for e in exps:
            own[e] = (f"{e}_{cfg.epoch}.pth", sb_names, xs, ys)
    for e, (ckpt, names, x, y) in own.items():
        acc = 100 * (predict(cfg.save_path, ckpt, names, x, device) == y).mean()
        print(f"- exp{e} ({ckpt}): {acc:.2f}")

    # the gull model on spoonbill data
    targets = gull_targets()
    guess = predict(cfg.save_path, cfg.gull_checkpoint, gull_names, xs, device)
    print(f"\n## exp{cfg.gull_exp} (gull) on spoonbill segments, segments\n")
    print(as_markdown(cross_table(ys, guess, sb_names, gull_names), index=True))
    h = hits(ys, guess, targets, sb_names)
    print(f"\n## exp{cfg.gull_exp} (gull) on spoonbill segments, % of segments\n")
    print(as_markdown(score_table([h], ys, sb_names, targets, "segments")))

    # the spoonbill models on gull data
    targets = spoonbill_targets()
    for arm, exps in cfg.spoonbill_arms.items():
        runs = []
        for e in exps:
            guess = predict(cfg.save_path, f"{e}_{cfg.epoch}.pth", sb_names, xg, device)
            runs.append(hits(yg, guess, targets, gull_names))
            if e == exps[0]:
                print(f"\n## exp{e} (spoonbill, {arm}) on gull bursts, bursts\n")
                t = cross_table(yg, guess, gull_names, sb_names)
                print(as_markdown(t, index=True))
        seeds = ", ".join(f"exp{e}" for e in exps)
        print(
            f"\n## Spoonbill, {arm}, on gull bursts, % of bursts, "
            f"mean (min-max) of {seeds}\n"
        )
        print(as_markdown(score_table(runs, yg, gull_names, targets, "bursts")))


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
        "gull_file": "/home/fatemeh/Downloads/bird/data/final/starts.csv",
        "gull_labels": [0, 1, 2, 3, 4, 5, 6, 8, 9],
        "gull_names": {
            0: "Flap",
            1: "ExFlap",
            2: "Soar",
            3: "Boat",
            4: "Float",
            5: "SitStand",
            6: "TerLoco",
            8: "Manouvre",
            9: "Pecking",
        },
        "spoonbill_file": "/home/fatemeh/Downloads/bird/data/spoonbill_final/"
        "spoonbill_starts.csv",
        "spoonbill_labels": sorted(spoonbill_names),
        "spoonbill_names": spoonbill_names,
        "gull_exp": 197,
        "gull_checkpoint": "197_best.pth",
        # seeds 32984, 1234, 5678 per arm; the first run gets a count table
        "spoonbill_arms": {
            "no rotation": [207, 210, 211],
            "rotation": [206, 208, 209],
        },
        "epoch": 4000,  # spoonbill runs: read <exp>_4000.pth
    }
    main(OmegaConf.create(config))
