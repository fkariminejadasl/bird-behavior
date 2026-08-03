"""Find likely mislabeled bursts in the labeled data.

Only **GPS speed** is used. Table S1 of Shamoun-Baranes et al. 2016 ("Flap or
soar?", supplementary `Judy_features_supp1.pdf`) defines each class by an
accelerometer characteristic and, for several classes, a speed. The speed part
is a hard physical constraint -- a bird labelled as sitting still cannot be
moving at 13 m/s -- so contradicting it is real evidence of a bad label.

Accelerometer-shape rules were tried and removed, because checking them against
the ground-truth plots in `~/Downloads/bird/results/gt2_starts/all` (made by
`exps/save_plots_gt.py`) showed they were wrong:

- An ODBA rule ("Stationary means constant on all 3 axes") flagged sitting birds
  that shifted once mid-burst. Sitting in the tail of a class's own distribution
  is not the same as being mislabeled.
- A spectral rule ("Soar means no wing beat", via the height of the strongest
  FFT peak of heave) flagged smooth glides. Only the mean is removed before the
  transform, not the trend, so the slow drift of a glide lands in the low
  frequency bins and imitates a beat over a 20-sample window.

A second opinion comes from a trained model, but it is reported, never used to
flag. Prefer a model trained *with* augmentation (exp196): it underfits, so it
never memorised the labels and its opinion is independent of the label under
test. A no-augmentation model such as exp194 reaches ~99.4% train accuracy by
learning the bad labels too, and will confirm them. See docs/lesson_learned.md.

Output is a CSV keyed by `device_id, datetime, start_index`. Look a row up in
`starts.csv`, or find its plot in `~/Downloads/bird/results/gt2_starts/all`,
which is named `<n>,<device_id>,<datetime>.png`.

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/find_label_noise.py
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from behavior import model as bm
from behavior import utils as bu


@dataclass
class Rule:
    """A speed that contradicts the burst's own label.

    Fires when the GPS speed of a burst labelled `label` is on the wrong side of
    `threshold` m/s. `paper` quotes the Table S1 definition being contradicted;
    `plain` says it in ordinary words. `describe_thresholds` prints each
    threshold beside the speed distribution of the class it tests, so it is
    visible that a rule fires only on bursts extreme for their own label.
    """

    label: str
    name: str
    op: str  # ">=" means "flag when speed >= threshold"
    threshold: float
    paper: str
    plain: str

    def fires(self, feat):
        speed = feat["gps"]
        return speed >= self.threshold if self.op == ">=" else speed < self.threshold

    def reason(self, value):
        return (
            f"{self.name}: gps={value:.3f} {self.op} {self.threshold} m/s - "
            f"{self.plain} [Table S1 {self.label}: {self.paper}]"
        )


# Flight averages 8.8 m/s ground speed (Table S3) and the ground-speed
# distribution for flight peaks near 9 m/s (Figure S2), so a ground or resting
# class at >= 5 m/s is moving at a speed only flight explains.
RULES = [
    Rule(
        "SitStand",
        "speed_not_zero",
        ">=",
        3.0,
        '"speed is close to zero"',
        "sitting still, yet moving at walking-to-flight speed",
    ),
    Rule(
        "Float",
        "speed_too_high",
        ">=",
        3.0,
        '"floating with the currents at sea"',
        "drifting on water, yet moving faster than any current",
    ),
    Rule(
        "TerLoco",
        "speed_too_high",
        ">=",
        5.0,
        '"typical periodic pattern of the surge (x-axis)" (walking)',
        "walking, yet moving at flight speed",
    ),
    Rule(
        "Pecking",
        "speed_too_high",
        ">=",
        5.0,
        '"irregular active pattern, low speed, on land"',
        "pecking on land, yet moving at flight speed",
    ),
]
# Boat is deliberately not checked. Table S1 says "constant speed (9-11 km/h)",
# but a bird on an anchored boat is legitimately near zero, so neither a slow
# nor a fast Boat burst is evidence of a bad label.


def load_bursts(csv, glen):
    """Raw bursts, in the same order and count as bd.load_csv_pandas."""
    df = pd.read_csv(csv, header=None)
    n = len(df) // glen
    return {
        "device_id": df[0].values.reshape(n, glen)[:, 0],
        "datetime": np.asarray(df[1], dtype=object).reshape(n, glen)[:, 0],
        "start_index": df[2].values.reshape(n, glen)[:, 0],
        "label": df[3].values.reshape(n, glen)[:, 0],
        "imu": df[[4, 5, 6]].values.reshape(n, glen, 3),
        "gps": df[7].values.reshape(n, glen)[:, 0],
    }


def compute_features(imu, gps):
    """Speed, which the rules test, plus two columns for context in the CSV.

    ODBA (overall dynamic body acceleration) is how much the accelerometer moves
    once the constant gravity offset is removed. It is reported so a reviewer can
    see at a glance whether a flagged burst is active or still. It is not a rule,
    for the reason in the module docstring.
    """
    return pd.DataFrame(
        {
            "gps": gps,
            "odba": np.abs(imu - imu.mean(1, keepdims=True)).sum(2).mean(1),
            "std_z": imu[:, :, 2].std(1),
        }
    )


def describe_thresholds(feat, rules):
    """Show each threshold against the speed distribution of the class it tests."""
    rows = []
    for rule in rules:
        speed = feat.loc[feat.label == rule.label, "gps"]
        rows.append(
            {
                "label": rule.label,
                "rule": rule.name,
                "test": f"gps {rule.op} {rule.threshold}",
                "n_class": len(speed),
                "class_median": speed.median(),
                "class_p99": speed.quantile(0.99),
                "class_max": speed.max(),
                "flagged": int(rule.fires(feat)[feat.label == rule.label].sum()),
            }
        )
    return pd.DataFrame(rows).round(3)


def model_predictions(cfg, imu, gps, device):
    """Predictions of a trained checkpoint, preprocessed as in training."""
    data = np.concatenate([imu, gps[:, None, None].repeat(imu.shape[1], 1)], axis=2)
    data[:, :, :3] = np.clip(data[:, :, :3], -2.0, 2.0)
    data[:, :, 3] /= cfg.gps_norm
    tensor = torch.tensor(data, dtype=torch.float32, device=device).transpose(1, 2)
    model = bm.BirdModelSmallDilated(4, 20, len(cfg.labels_to_use)).to(device)
    bm.load_model(f"{cfg.save_path}/{cfg.model_exp}_best.pth", model, device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    conf, pred = probs.max(1)
    return pred.cpu().numpy(), conf.cpu().numpy()


def main(cfg):
    bursts = load_bursts(cfg.data_file, cfg.glen)
    feat = compute_features(bursts["imu"], bursts["gps"])
    names = np.array([bu.ind2name[v] for v in bursts["label"]])
    feat["label"] = names
    print(f"{len(feat)} bursts from {cfg.data_file}")
    print("\nEach threshold against the speed distribution of the class it tests:")
    print(describe_thresholds(feat, RULES).to_string(index=False))

    reasons = [[] for _ in range(len(feat))]
    for rule in RULES:
        hit = (feat.label == rule.label) & rule.fires(feat)
        for i in np.flatnonzero(hit.values):
            reasons[i].append(rule.reason(feat["gps"].values[i]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred, conf = model_predictions(cfg, bursts["imu"], bursts["gps"], device)
    pred_name = np.array([bu.ind2name[cfg.labels_to_use[int(p)]] for p in pred])
    base_rate = (pred_name != names).mean()

    flagged = np.flatnonzero([len(r) > 0 for r in reasons])
    out = pd.DataFrame(
        {
            "device_id": bursts["device_id"][flagged],
            "datetime": bursts["datetime"][flagged],
            "start_index": bursts["start_index"][flagged],
            "label": names[flagged],
            "model_pred": pred_name[flagged],
            "model_conf": conf[flagged].round(3),
            "gps_speed": feat["gps"].values[flagged].round(3),
            "odba": feat["odba"].values[flagged].round(3),
            "std_z": feat["std_z"].values[flagged].round(3),
            "reason": ["; ".join(reasons[i]) for i in flagged],
        }
    )
    out.insert(6, "model_disagrees", out.label != out.model_pred)
    # Flags bunched on one device and day are more likely a GPS or device
    # problem over that period than that many independent labelling mistakes.
    day = out.datetime.str[:10]
    out["device_day_flags"] = out.groupby([out.device_id, day]).device_id.transform(
        "size"
    )
    out = out.sort_values("gps_speed", ascending=False)

    dest = Path(cfg.out_dir) / cfg.out_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"\n{len(out)} bursts flagged ({len(out) / len(feat):.2%})")
    print(
        f"exp{cfg.model_exp} disagrees with {out.model_disagrees.mean():.0%} of them "
        f"and with {base_rate:.1%} of all bursts -- compare the two before reading "
        "its disagreement as corroboration"
    )
    print(
        "\nflags per device and day (>=3 suggests a device/GPS problem over "
        "that period rather than that many labelling mistakes):"
    )
    print(out.groupby([out.device_id, out.datetime.str[:10]]).size().to_string())
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    config = {
        "data_file": "/home/fatemeh/Downloads/bird/data/final/starts.csv",
        "save_path": "/home/fatemeh/Downloads/bird/results",
        "out_dir": "/home/fatemeh/Downloads/bird/screenshots",
        "out_name": "label_noise_candidates.csv",
        # Rotation-augmented model: it underfits, so it never memorised the
        # labels and its opinion is independent of the label being tested.
        "model_exp": 196,
        "labels_to_use": [0, 1, 2, 3, 4, 5, 6, 8, 9],
        "gps_norm": 22.3012351755624,
        "glen": 20,
    }
    main(OmegaConf.create(config))
