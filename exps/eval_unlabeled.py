"""Compare models on **unlabeled** data, without ground truth.

Softmax confidence cannot answer "is this model healthy on a new bird". Measured
on the orientation shift: exp194 keeps 0.735 mean confidence while its accuracy
falls to 13%, so a confidence threshold raises no alarm (see
docs/lesson_learned.md). Confidence is only good for *ranking* which bursts to
inspect (AUROC 0.89 within a dataset).

These four checks need no labels. Each counts predictions that contradict
something we know independently of the model, so the rate is a real error rate,
not a proxy:

1. **speed_conflict** -- the predicted class contradicts the GPS speed, using the
   Table S1 rules of `exps/find_label_noise.py` (a bird predicted to be sitting
   still cannot be moving at 13 m/s).
2. **place_conflict** -- the predicted class contradicts the location: walking or
   pecking out at sea, floating or on a boat well inland. Uses `global_land_mask`
   (1/100 deg, ~1.1 km), so bursts within `coast_margin_deg` of a coastline are
   excluded rather than flagged; a gull on a beach or pier is genuinely ambiguous
   at that resolution.
3. **time_flip** -- consecutive bursts inside one (device, datetime) fix are one
   second apart, and a gull does not alternate behaviours that fast. The share of
   adjacent pairs whose prediction changes is an instability rate.
4. **rotation_flip** -- the same burst re-predicted under random SO(3) rotations.
   A model that changes its mind when the logger is mounted differently is
   unreliable on any logger it was not trained on.

Input is the app CSV format written by `scripts/data/bird_behavior_app_data.py`:
`device_id, datetime, index, gt_label, imu_x, imu_y, imu_z, gps, label, conf,
lat, lon[, altitude]`. Columns 8-9 hold whichever model produced the file; they
are ignored, since every model listed in `model_exps` is run here directly.

Writes a per-burst triage CSV, worst first, ready to open in the app.

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/eval_unlabeled.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from global_land_mask import globe
from omegaconf import OmegaConf

from behavior import model as bm
from behavior import utils as bu
from behavior.data_augmentation import random_rotation_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_label_noise import RULES  # noqa: E402  (same directory)

# Classes whose Table S1 definition places the bird on land or at sea.
ON_LAND = {"TerLoco", "Pecking"}
AT_SEA = {"Float", "Boat"}


def load_app_csv(path, glen):
    """Bursts of glen rows from the app CSV format."""
    df = pd.read_csv(path, header=None)
    n = len(df) // glen
    first = lambda col: df[col].values.reshape(n, glen)[:, 0]  # noqa: E731
    return {
        "device_id": first(0),
        "datetime": np.asarray(df[1], dtype=object).reshape(n, glen)[:, 0],
        "start_index": first(2),
        "imu": df[[4, 5, 6]].values.reshape(n, glen, 3),
        "gps": first(7),
        "lat": first(10),
        "lon": first(11),
    }


def predict(bursts, exp, cfg, device):
    """Run one checkpoint over every burst; return predicted names and confidence."""
    imu, gps = bursts["imu"], bursts["gps"]
    data = np.concatenate([imu, gps[:, None, None].repeat(imu.shape[1], 1)], axis=2)
    data[:, :, :3] = np.clip(data[:, :, :3], -2.0, 2.0)
    data[:, :, 3] /= cfg.gps_norm
    x = torch.tensor(data, dtype=torch.float32, device=device).transpose(1, 2)

    model = bm.BirdModelSmallDilated(4, 20, len(cfg.labels_to_use)).to(device)
    bm.load_model(f"{cfg.save_path}/{exp}_best.pth", model, device)
    model.eval()

    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(x), cfg.batch_size):
            p = torch.softmax(model(x[i : i + cfg.batch_size]), dim=1)
            c, k = p.max(1)
            preds.append(k)
            confs.append(c)
    pred = torch.cat(preds)
    conf = torch.cat(confs).cpu().numpy()
    names = np.array([bu.ind2name[cfg.labels_to_use[int(k)]] for k in pred.cpu()])
    return names, conf, x, model


def rotation_flip_rate(x, model, names, cfg, device):
    """Share of rotations under which the prediction changes."""
    name2col = {bu.ind2name[lab]: col for col, lab in enumerate(cfg.labels_to_use)}
    base = torch.tensor([name2col[n] for n in names], device=device)
    flips = torch.zeros(len(x), device=device)
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(cfg.n_rotations):
            rot = random_rotation_matrix(device=device, dtype=x.dtype)
            preds = []
            for i in range(0, len(x), cfg.batch_size):
                chunk = x[i : i + cfg.batch_size].clone()
                chunk[:, :3] = torch.einsum("ij,njt->nit", rot, chunk[:, :3])
                preds.append(model(chunk).argmax(1))
            flips += (torch.cat(preds) != base).float()
    return (flips / cfg.n_rotations).cpu().numpy()


def speed_conflict(names, gps):
    """Predicted class contradicts GPS speed (Table S1 rules)."""
    feat = pd.DataFrame({"gps": gps, "label": names})
    reason = np.array([""] * len(names), dtype=object)
    for rule in RULES:
        hit = ((feat.label == rule.label) & rule.fires(feat)).values
        for i in np.flatnonzero(hit):
            reason[i] = rule.reason(gps[i])
    return reason


def place_conflict(names, lat, lon, margin):
    """Predicted class contradicts land/sea, ignoring the coastal strip."""
    is_land = globe.is_land(lat, lon)
    # A burst is "clearly" inland/offshore only if the mask agrees at +-margin.
    clear = np.ones(len(lat), dtype=bool)
    for dlat in (-margin, 0, margin):
        for dlon in (-margin, 0, margin):
            clear &= globe.is_land(lat + dlat, lon + dlon) == is_land

    reason = np.array([""] * len(names), dtype=object)
    on_land = np.isin(names, list(ON_LAND))
    at_sea = np.isin(names, list(AT_SEA))
    for i in np.flatnonzero(clear & on_land & ~is_land):
        reason[i] = f"place_conflict: {names[i]} predicted at sea"
    for i in np.flatnonzero(clear & at_sea & is_land):
        reason[i] = f"place_conflict: {names[i]} predicted inland"
    return reason, clear


def time_flip(bursts, names):
    """Mark bursts whose prediction differs from the previous burst of the fix."""
    key = pd.DataFrame(
        {"d": bursts["device_id"], "t": bursts["datetime"], "i": bursts["start_index"]}
    )
    order = np.lexsort((key.i.values, key.t.values, key.d.values))
    flip = np.zeros(len(names), dtype=bool)
    same_fix = (key.d.values[order][1:] == key.d.values[order][:-1]) & (
        key.t.values[order][1:] == key.t.values[order][:-1]
    )
    changed = names[order][1:] != names[order][:-1]
    flip[order[1:]] = same_fix & changed
    return flip, same_fix.sum()


def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bursts = load_app_csv(cfg.data_file, cfg.glen)
    n = len(bursts["gps"])
    print(f"{n:,} bursts from {cfg.data_file}\n")

    rows, per_model = [], {}
    for exp in cfg.model_exps:
        names, conf, x, model = predict(bursts, exp, cfg, device)
        speed = speed_conflict(names, bursts["gps"])
        place, clear = place_conflict(
            names, bursts["lat"], bursts["lon"], cfg.coast_margin_deg
        )
        flip, n_pairs = time_flip(bursts, names)
        rot = rotation_flip_rate(x, model, names, cfg, device)
        del x, model
        torch.cuda.empty_cache()

        per_model[exp] = dict(
            names=names, conf=conf, speed=speed, place=place, flip=flip, rot=rot
        )
        rows.append(
            {
                "model": f"exp{exp}",
                "mean_conf": conf.mean(),
                "speed_conflict_%": 100 * (speed != "").mean(),
                "place_conflict_%": 100 * (place != "").mean(),
                "time_flip_%": 100 * flip.sum() / max(n_pairs, 1),
                "rotation_flip_%": 100 * rot.mean(),
            }
        )
    print("Label-free checks (lower is better, except mean_conf):")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print(
        f"\nplace_conflict is measured on the "
        f"{100 * clear.mean():.0f}% of bursts that are clearly inland or "
        f"offshore; the coastal strip is excluded."
    )

    a, b = cfg.model_exps[0], cfg.model_exps[1]
    agree = (per_model[a]["names"] == per_model[b]["names"]).mean()
    print(f"exp{a} and exp{b} agree on {100 * agree:.1f}% of bursts.")

    print("\nPredicted class distribution (%):")
    dist = pd.DataFrame(
        {
            f"exp{e}": pd.Series(per_model[e]["names"]).value_counts(normalize=True)
            * 100
            for e in cfg.model_exps
        }
    )
    print(dist.round(1).sort_values(dist.columns[0], ascending=False).to_string())

    # Triage list from the last model: worst offenders first.
    exp = cfg.model_exps[-1]
    m = per_model[exp]
    problem = (m["speed"] != "") | (m["place"] != "") | m["flip"]
    out = pd.DataFrame(
        {
            "device_id": bursts["device_id"],
            "datetime": bursts["datetime"],
            "start_index": bursts["start_index"],
            f"pred_exp{exp}": m["names"],
            f"pred_exp{a}": per_model[a]["names"],
            "conf": m["conf"].round(3),
            "rotation_flip": m["rot"].round(3),
            "gps_speed": bursts["gps"].round(3),
            "lat": bursts["lat"],
            "lon": bursts["lon"],
            "speed_conflict": m["speed"],
            "place_conflict": m["place"],
            "time_flip": m["flip"],
        }
    )[problem]
    out = out.sort_values(["conf"]).head(cfg.max_triage_rows)
    dest = Path(cfg.out_dir) / cfg.out_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(
        f"\n{problem.sum():,} bursts fail at least one check for exp{exp}; "
        f"lowest-confidence {len(out):,} written to {dest}"
    )


if __name__ == "__main__":
    config = {
        "data_file": (
            "/home/fatemeh/Downloads/bird/data/ssl/gimu_behavior/gull/6004_194.csv"
        ),
        "save_path": "/home/fatemeh/Downloads/bird/results",
        "out_dir": "/home/fatemeh/Downloads/bird/data/final",
        "out_name": "unlabeled_triage_6004.csv",
        "model_exps": [194, 196],  # baseline first, then the model of interest
        "labels_to_use": [0, 1, 2, 3, 4, 5, 6, 8, 9],
        "gps_norm": 22.3012351755624,
        "glen": 20,
        "batch_size": 65536,
        "n_rotations": 10,
        "coast_margin_deg": 0.02,  # ~2 km; skip bursts near a coastline
        "max_triage_rows": 500,
    }
    main(OmegaConf.create(config))
