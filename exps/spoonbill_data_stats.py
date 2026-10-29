"""How much spoonbill data is there, per behaviour, in 20-sample windows?

The spoonbill data of Lok et al. (Eurasian spoonbill, UvA-BiTS tags) sits in
`~/Downloads/bird/data/spoonbill`, one sample per row:

- `video_annotated_data.csv`: behaviours labeled from video, one label per
  sample, several labels per fix.
- `visual_annotated_passive_flight_data.csv`: soaring (fly-soar), labeled by eye.
  Its rows are copied from the migration files; the script checks that.
- `acc_data_migration_<bird>.csv`: three birds on migration, no labels.
- `behaviours.csv`: label index to name.

A window is 20 consecutive samples with one label, the gull model input. They
are cut as `data_processing.slice_from_first_label` cuts `starts.csv`: per fix,
each run of consecutive samples with one label gives `length // 20`
non-overlapping windows, from the run's first sample. `sliding` counts every
stride-1 window inside a run instead, an upper bound like the gull `shift`.
`complete` windows have a value in all of x, y, z and speed.

A sample listed twice in one fix with two labels is dropped, both copies.

Also checks the sampling rate against the video clock (`time.on.video`, one
value per second of video) and prints data quality per file. Prints only,
writes nothing.

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/spoonbill_data_stats.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_labeled import as_markdown  # noqa: E402
from label_cooccurrence import label_ranges_per_fix  # noqa: E402

# spoonbill column -> gull column: 0 device, 1 datetime, 2 index, 3 label,
# 4-6 IMU x, y, z, 7 GPS speed
COLUMNS = {
    "BirdID": 0,
    "date.time": 1,
    "Index": 2,
    "behaviour.index": 3,
    "x": 4,
    "y": 5,
    "z": 6,
    "speed_2d": 7,
}
SIGNAL = [4, 5, 6, 7]


def load(path, label=None):
    """One file in the gull column order. `label` fills a file without labels."""
    df = pd.read_csv(path)
    if "date.time" not in df:
        df["date.time"] = df["Date"] + " " + df["Time"]
    if label is not None:
        df["behaviour.index"] = label
    return df.rename(columns=COLUMNS)


def drop_conflicts(df):
    """Drop every sample listed twice in one fix. Returns (kept, dropped)."""
    dup = df.duplicated([0, 1, 2], keep=False)
    return df[~dup], df[dup]


def cut_windows(df, glen):
    """One row per non-overlapping window, and the sliding count per label."""
    ranges = label_ranges_per_fix(df)
    missing = df[df[SIGNAL].isna().any(axis=1)].groupby([0, 1])[2].apply(set)
    rows, sliding = [], {}
    for key, spans in ranges.items():
        bad = missing.get(key, set())
        for s, e, lab in spans:
            n = e - s + 1
            sliding[lab] = sliding.get(lab, 0) + max(0, n - glen + 1)
            for st in range(s, s + n // glen * glen, glen):
                complete = bad.isdisjoint(range(st, st + glen))
                rows.append((key[0], key[1], st, lab, complete))
    cols = ["device", "datetime", "start", "label", "complete"]
    return pd.DataFrame(rows, columns=cols), sliding, ranges


def quality(name, df):
    """One row of data-quality counts for a file."""
    fix = df.groupby([0, 1])
    gaps = fix[2].apply(lambda s: int((np.diff(np.sort(s.values)) > 1).any()))
    return {
        "file": name,
        "samples": len(df),
        "fixes": fix.ngroups,
        "birds": df[0].nunique(),
        "missing x/y/z": "/".join(str(df[c].isna().sum()) for c in [4, 5, 6]),
        "missing speed": int(df[7].isna().sum()),
        "fixes no speed": int(fix[7].apply(lambda s: s.isna().all()).sum()),
        "fixes with gap": int(gaps.sum()),
        "IMU beyond 2 g": int((df[[4, 5, 6]].abs() > 2).any(axis=1).sum()),
        "speed max": float(df[7].max()),
    }


def samples_per_second(raw):
    """Samples per second of video, inner seconds of gap-free fixes only.

    The first and last second of a fix are partial, so they are left out.
    """
    counts = []
    for _, fix in raw.groupby(["BirdID", "date.time"]):
        if (np.diff(np.sort(fix["Index"].values)) != 1).any():
            continue
        per_sec = fix.groupby("time.on.video").size().sort_index().values
        counts.extend(per_sec[1:-1])
    return pd.Series(counts).value_counts().sort_index()


def main(cfg):
    data = Path(cfg.data_dir)
    names = pd.read_csv(data / cfg.names_file)
    names = dict(zip(names["behaviour.index"], names["behaviour"]))

    raw_video = pd.read_csv(data / cfg.video_file)
    video, dropped = drop_conflicts(raw_video.rename(columns=COLUMNS))
    all_video = raw_video.rename(columns=COLUMNS)
    soar = load(data / cfg.soar_file)
    migr = pd.concat(
        [load(data / f"{cfg.migration_prefix}{b}.csv", label=0) for b in cfg.birds]
    )

    print("## Files\n")
    rows = [quality(cfg.video_file, all_video), quality(cfg.soar_file, soar)]
    rows.append(quality(f"{cfg.migration_prefix}*.csv", migr))
    print(as_markdown(pd.DataFrame(rows)))
    print(
        f"\n{len(dropped)} samples dropped from {cfg.video_file}: one index, two "
        f"labels in one fix"
    )
    print(dropped.groupby([0, 1, 2])[3].apply(list).to_string())

    per_sec = samples_per_second(raw_video)
    print(f"\nSamples per inner second of video: {per_sec.to_dict()} (n seconds)")

    # The soaring file should be a labeled copy of the migration rows.
    copy = migr[[0, 1, 2] + SIGNAL].rename(columns={c: f"m{c}" for c in SIGNAL})
    both = soar.merge(copy, on=[0, 1, 2])
    same = np.allclose(
        both[SIGNAL].values, both[[f"m{c}" for c in SIGNAL]].values, equal_nan=True
    )
    print(
        f"Soaring file rows found in the migration files: {len(both)} of "
        f"{len(soar)}, values identical: {same}"
    )

    print("\n## Video fixes: labels per fix\n")
    n_lab = video.groupby([0, 1])[3].nunique().value_counts().sort_index()
    print(", ".join(f"{k} label(s): {v} fixes" for k, v in n_lab.items()))
    lens = video.groupby([0, 1]).size()
    print(f"samples per fix: median {lens.median():.0f}, max {lens.max()}")

    print(f"\n## Windows of {cfg.glen} samples per behaviour\n")
    win_v, slide_v, ranges_v = cut_windows(video, cfg.glen)
    win_s, slide_s, ranges_s = cut_windows(soar, cfg.glen)
    win = pd.concat([win_v, win_s])
    runs = pd.Series(
        [lab for r in (ranges_v, ranges_s) for v in r.values() for _, _, lab in v]
    ).value_counts()
    samples = pd.concat([video[3], soar[3]]).value_counts()
    n_win = win["label"].value_counts()
    table = pd.DataFrame(
        {
            "label": list(names),
            "name": list(names.values()),
            "samples": [samples.get(k, 0) for k in names],
            "runs": [runs.get(k, 0) for k in names],
            "windows": [n_win.get(k, 0) for k in names],
            "%": [100 * n_win.get(k, 0) / len(win) for k in names],
            "complete": [win[win.label == k]["complete"].sum() for k in names],
            "sliding": [slide_v.get(k, 0) + slide_s.get(k, 0) for k in names],
        }
    )
    total = table.drop(columns=["label", "name"]).sum()
    total = {"label": "", "name": "**total**", **total.to_dict()}
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
    counts = ["samples", "runs", "windows", "complete", "sliding"]
    print(as_markdown(table.astype({c: int for c in counts})))
    print(
        f"\n{names[cfg.soar_label]}: {(win_v.label == cfg.soar_label).sum()} windows "
        f"from video, {len(win_s)} from {cfg.soar_file}"
    )

    print("\n## Windows per behaviour and bird\n")
    per_bird = win.pivot_table(
        index="label", columns="device", values="start", aggfunc="count", fill_value=0
    )
    per_bird.insert(0, "name", [names[k] for k in per_bird.index])
    per_bird.columns = [str(c) for c in per_bird.columns]
    print(as_markdown(per_bird, index=True))

    print("\n## Migration files (no labels)\n")
    win_m, _, _ = cut_windows(migr, cfg.glen)
    soar_fixes = soar.groupby(0)[1].nunique()
    rows = []
    for bird, df in migr.groupby(0):
        lens = df.groupby(1).size()
        rows.append(
            {
                "bird": bird,
                "first": df[1].min(),
                "last": df[1].max(),
                "fixes": len(lens),
                f"fixes of {lens.mode()[0]}": int((lens == lens.mode()[0]).sum()),
                "windows": int((win_m.device == bird).sum()),
                "complete": int(win_m[win_m.device == bird]["complete"].sum()),
                "fixes in soar file": int(soar_fixes.get(bird, 0)),
            }
        )
    print(as_markdown(pd.DataFrame(rows)))


if __name__ == "__main__":
    config = {
        "data_dir": "/home/fatemeh/Downloads/bird/data/spoonbill",
        "video_file": "video_annotated_data.csv",
        "soar_file": "visual_annotated_passive_flight_data.csv",
        "names_file": "behaviours.csv",
        "migration_prefix": "acc_data_migration_",
        "birds": [6283, 6287, 6291],  # one migration file per bird
        "soar_label": 9,  # fly-soar, the only label of soar_file
        "glen": 20,  # samples per window, as for the gull model
    }
    main(OmegaConf.create(config))
