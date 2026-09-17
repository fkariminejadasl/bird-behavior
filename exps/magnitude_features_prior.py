"""Every number quoted for exp197 in docs/lesson_learned.md, without training.

A random forest on burst-level summaries stands in for the CNN: it shares the
relevant weakness (it cannot form sqrt(x^2+y^2+z^2) from per-axis statistics),
runs in seconds, and answers whether the three magnitude channels carry anything
once the accelerometer frame is scrambled.

  /home/fatemeh/miniconda3/envs/bird/bin/python \
    /home/fatemeh/Downloads/bird/claude/magnitude_feature_prior.py
"""

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from behavior import data as bd
from behavior import model as bm
from behavior import utils as bu

DATA = "/home/fatemeh/Downloads/bird/data/final/starts.csv"
LABELS = [0, 1, 2, 3, 4, 5, 6, 8, 9]
SEED, N_REP = 32984, 5


def rotate(igs, rng):
    """Random SO(3) per burst, acc channels only (as BatchRandomRotation3D)."""
    q = rng.normal(size=(len(igs), 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, i, j, k = q.T
    rot = np.stack(
        [
            1 - 2 * (j * j + k * k),
            2 * (i * j - w * k),
            2 * (i * k + w * j),
            2 * (i * j + w * k),
            1 - 2 * (i * i + k * k),
            2 * (j * k - w * i),
            2 * (i * k - w * j),
            2 * (j * k + w * i),
            1 - 2 * (i * i + j * j),
        ],
        axis=1,
    ).reshape(len(igs), 3, 3)
    out = igs.copy()
    out[:, :, :3] = np.einsum("nij,ntj->nti", rot, igs[:, :, :3])
    return out


def derived(igs):
    return bd.add_magnitude_features(igs)[:, :, 4:]


def summarize(x):
    """N x T x C -> N x 5C: mean, std, min, max, mean |diff| over time."""
    return np.concatenate(
        [x.mean(1), x.std(1), x.min(1), x.max(1), np.abs(np.diff(x, axis=1)).mean(1)],
        axis=1,
    )


igs, ldts = bd.load_csv_pandas(DATA, LABELS, glen=20)
y = ldts[:, 0]
present = sorted(set(y))
names = [bu.ind2name[LABELS[i]] for i in present]
extra = derived(igs)
print(f"{len(igs)} bursts from {DATA}\n")

# --- 1. rotation invariance --------------------------------------------------
rng = np.random.default_rng(SEED)
print("Max |derived(rotated) - derived(clean)|, float64 then float32:")
for cast in (np.float64, np.float32):
    g = igs.astype(cast)
    d, dr = derived(g), derived(rotate(g, rng).astype(cast))
    per = [np.abs(dr[:, :, c] - d[:, :, c]).max() for c in range(3)]
    print(f"  {cast.__name__:>8}: " + "  ".join(f"{v:.2e}" for v in per))

# --- 2. per-class distributions ---------------------------------------------
print("\nPer-class burst means (g; jerk per 1/20 s step)")
print(f"{'class':>9} {'n':>5} {'mag':>7} {'dyn_mag':>8} {'jerk_mag':>9} {'gps m/s':>8}")
for i, name in zip(present, names):
    m = y == i
    print(
        f"{name:>9} {m.sum():>5} {extra[m, :, 0].mean():>7.3f} "
        f"{extra[m, :, 1].mean():>8.3f} {extra[m, :, 2].mean():>9.3f} "
        f"{igs[m, :, 3].mean():>8.2f}"
    )

print("\nCorrelation of the three channels' burst means:")
print(np.corrcoef(extra.mean(1).T).round(3))

# --- 3. discriminability proxy, clean vs rotated frame -----------------------
splits = bu.stratified_split(torch.tensor(y), split_ratios=[0.9, 0.1], seed=SEED)
tr, va = splits[0].numpy(), splits[1].numpy()

variants = [
    ([], "4ch baseline"),
    ([0], "+ mag"),
    ([1], "+ dyn_mag"),
    ([2], "+ jerk_mag"),
    ([0, 1, 2], "+ all three"),
]
acc = {tag: {"clean": [], "rot": []} for _, tag in variants}
f1 = {"4ch": [], "7ch": []}

for rep in range(N_REP):
    rng = np.random.default_rng(SEED + rep)
    tr_rot, va_rot = rotate(igs[tr], rng), rotate(igs[va], rng)
    for keep, tag in variants:

        def feats(g):
            if not keep:
                return summarize(g)
            return summarize(np.concatenate([g, derived(g)[:, :, keep]], axis=2))

        for regime, (a, b) in {
            "clean": (igs[tr], igs[va]),
            "rot": (tr_rot, va_rot),
        }.items():
            clf = RandomForestClassifier(
                n_estimators=300, random_state=SEED + rep, n_jobs=-1
            )
            clf.fit(feats(a), y[tr])
            acc[tag][regime].append(100 * clf.score(feats(b), y[va]))
            if regime == "rot" and tag in ("4ch baseline", "+ all three"):
                key = "4ch" if not keep else "7ch"
                f1[key].append(
                    f1_score(y[va], clf.predict(feats(b)), average=None, labels=present)
                )

print(f"\nRandom forest on burst summaries, {N_REP} draws, valid acc % (mean +/- std)")
print(f"{'features':>22} {'clean':>14} {'rotated':>14}")
for _, tag in variants:
    c, r = np.array(acc[tag]["clean"]), np.array(acc[tag]["rot"])
    print(
        f"{tag:>22} {c.mean():>7.2f}+/-{c.std():<5.2f} "
        f"{r.mean():>7.2f}+/-{r.std():<5.2f}"
    )

a, b = np.array(f1["4ch"]).mean(0), np.array(f1["7ch"]).mean(0)
print("\nRotated regime, mean per-class valid F1")
print(f"{'class':>9} {'n valid':>8} {'4ch':>6} {'7ch':>6} {'delta':>7}")
for i, name, u, v in zip(present, names, a, b):
    print(f"{name:>9} {(y[va] == i).sum():>8} {u:>6.2f} {v:>6.2f} {v - u:>+7.2f}")

# --- 4. cost -----------------------------------------------------------------
print()
for c in (4, 7):
    n = sum(p.numel() for p in bm.BirdModelSmallDilated(c, 20, 9, 0.15).parameters())
    print(f"BirdModelSmallDilated({c}, 20, 9): {n:,} params")
d7 = bd.BirdDataset(igs, ldts, None, channel_first=True, add_magnitudes=True)
print("Channel means as the model sees them:", d7.data.mean(axis=(0, 1)).round(3))
