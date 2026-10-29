"""How much spoonbill data is there, per behaviour, in 20-record segments?

The spoonbill data of Lok et al. 2023 (Eurasian spoonbill, UvA-BiTS tags,
20 Hz) sits in `~/Downloads/bird/data/spoonbill`:

- `video_annotated_data.csv`: behaviours labeled from video, one label per
  record, several labels per fix.
- `visual_annotated_passive_flight_data.csv`: soaring (fly-soar), labeled by eye
  from plots of migration data. Its records are copied from the migration
  files; the script checks that.
- `acc_data_migration_<bird>.csv`: three birds on migration, no labels.
- `behaviours.csv`: label index to name.

The begging data of Lok et al. 2025 (post-fledging care) sits in
`~/Downloads/bird/data/spoonbill_postfledging`, as released on Zenodo. Only
`data/raw/annotated.begging.data.csv` has labeled IMU records: one juvenile,
text labels (beg, beg_flap, preen_tag, shake_feathers), each labeled from video
or graphically, from the signal alone. Its other files hold GPS and the paper's
model output, no IMU. It gets the same counts, and segments per annotation
method.

Terms, as in `docs/descriptions.md`, Terms:

- record: one row, one IMU reading x, y, z, with the GPS speed of its fix.
- fix: one GPS timestamp of one bird, with one speed. The records measured
  with it are its burst.
- bout: consecutive records of one fix with one label.
- segment: 20 consecutive records with one label, the gull model input.

Segments are cut as `data_processing.slice_from_first_label` cuts `starts.csv`:
a bout of n records gives `n // 20` non-overlapping segments, from its first
record. A bout shorter than 20 gives none, and a longer bout leaves its last
`n % 20` records unused; `% records used` is the share of records inside a
segment, `% of segments` the share of all segments.
`sliding` counts every stride-1 segment inside a bout instead, an upper bound
like the gull `shift`. `complete` segments have a value in all of x, y, z and
speed.

A record listed twice in one fix with two labels is dropped, both copies.

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
    """Drop every record listed twice in one fix. Returns (kept, dropped)."""
    dup = df.duplicated([0, 1, 2], keep=False)
    return df[~dup], df[dup]


def cut_segments(df, glen):
    """The non-overlapping segments and the bouts, one row each."""
    missing = df[df[SIGNAL].isna().any(axis=1)].groupby([0, 1])[2].apply(set)
    segments, bouts = [], []
    for key, spans in label_ranges_per_fix(df).items():
        bad = missing.get(key, set())
        for s, e, lab in spans:
            n = e - s + 1
            bouts.append((key[0], key[1], s, lab, n))
            for st in range(s, s + n // glen * glen, glen):
                complete = bad.isdisjoint(range(st, st + glen))
                segments.append((key[0], key[1], st, lab, complete))
    cols = ["device", "datetime", "start", "label"]
    segments = pd.DataFrame(segments, columns=cols + ["complete"])
    bouts = pd.DataFrame(bouts, columns=cols + ["length"])
    return segments, bouts


def per_label(segments, bouts, glen):
    """Counts per label, from the bouts and their segments."""
    out = bouts.assign(
        short=bouts["length"] < glen,
        sliding=(bouts["length"] - glen + 1).clip(lower=0),
    ).groupby("label")
    out = out.agg(
        records=("length", "sum"),
        bouts=("length", "size"),
        short=("short", "sum"),
        sliding=("sliding", "sum"),
    )
    by_label = segments.groupby("label")
    out["segments"] = by_label.size()
    out["complete"] = by_label["complete"].sum()
    out = out.fillna(0).astype(int)
    out.loc["total"] = out.sum()
    out["% of segments"] = 100 * out["segments"] / out.loc["total", "segments"]
    out["% records used"] = 100 * glen * out["segments"] / out["records"]
    cols = ["records", "bouts", "short", "segments", "% of segments"]
    cols += ["% records used", "complete"]
    return out[cols + ["sliding"]].rename(columns={"short": f"bouts < {glen}"})


def quality(name, df):
    """One row of data-quality counts for a file.

    Speed is one value per fix, so it is counted in fixes; x, y, z in records.
    """
    fix = df.groupby([0, 1])
    gaps = fix[2].apply(lambda s: int((np.diff(np.sort(s.values)) > 1).any()))
    return {
        "file": name,
        "records": len(df),
        "fixes": fix.ngroups,
        "birds": df[0].nunique(),
        "records missing x/y/z": "/".join(str(df[c].isna().sum()) for c in [4, 5, 6]),
        "fixes missing speed": int(fix[7].apply(lambda s: s.isna().any()).sum()),
        "fixes with gap": int(gaps.sum()),
        "records beyond 2 g": int((df[[4, 5, 6]].abs() > 2).any(axis=1).sum()),
        "speed max": float(df[7].max()),
    }


def records_per_second(raw):
    """Records per second of video, inner seconds of gap-free fixes only.

    The first and last second of a fix are partial, so they are left out.
    """
    counts = []
    for _, fix in raw.groupby(["BirdID", "date.time"]):
        if (np.diff(np.sort(fix["Index"].values)) != 1).any():
            continue
        per_sec = fix.groupby("time.on.video").size().sort_index().values
        counts.extend(per_sec[1:-1])
    return pd.Series(counts).value_counts().sort_index()


def load_begging(path):
    """The begging file in the gull column order, text labels as codes.

    Returns the frame and {code: name}.
    """
    df = pd.read_csv(path)
    names = dict(enumerate(sorted(df["behaviour"].unique())))
    df["behaviour.index"] = df["behaviour"].map({v: k for k, v in names.items()})
    return df.rename(columns=COLUMNS), names


def begging_report(cfg):
    """The same counts for the begging file, and segments per annotation method."""
    beg, names = load_begging(Path(cfg.begging_dir) / cfg.begging_file)
    print(f"\n# Begging data, {cfg.begging_file}\n")
    print(as_markdown(pd.DataFrame([quality(cfg.begging_file, beg)])))
    beg, dropped = drop_conflicts(beg)
    print(f"\n{len(dropped)} records dropped: the same index twice in one fix")
    lens = beg.groupby([0, 1]).size()
    print(f"records per fix: median {lens.median():.0f}, max {lens.max()}")

    print(f"\n## Segments of {cfg.glen} records per behaviour\n")
    seg, bouts = cut_segments(beg, cfg.glen)
    table = per_label(seg, bouts, cfg.glen)
    table.insert(0, "name", [names.get(k, "") for k in table.index])
    print(as_markdown(table.rename_axis("label"), index=True))

    print("\n## Segments per behaviour and annotation method\n")
    # a segment takes the method of its first record
    method = beg[[0, 1, 2, "annotation.method"]]
    method = method.rename(columns={0: "device", 1: "datetime", 2: "start"})
    seg = seg.merge(method, on=["device", "datetime", "start"])
    per_method = pd.crosstab(
        seg["label"].map(names).rename("name"), seg["annotation.method"], margins=True
    )
    print(as_markdown(per_method, index=True))


def main(cfg):
    data = Path(cfg.data_dir)
    names = pd.read_csv(data / cfg.names_file)
    names = dict(zip(names["behaviour.index"], names["behaviour"]))

    raw_video = pd.read_csv(data / cfg.video_file)
    all_video = raw_video.rename(columns=COLUMNS)
    video, dropped = drop_conflicts(all_video)
    soar = load(data / cfg.soar_file)
    migr = pd.concat(
        [load(data / f"{cfg.migration_prefix}{b}.csv", label=0) for b in cfg.birds]
    )

    print("## Files\n")
    rows = [quality(cfg.video_file, all_video), quality(cfg.soar_file, soar)]
    rows.append(quality(f"{cfg.migration_prefix}*.csv", migr))
    print(as_markdown(pd.DataFrame(rows)))
    print(
        f"\n{len(dropped)} records dropped from {cfg.video_file}: the same index "
        f"twice in one fix, with two labels"
    )
    print(dropped.groupby([0, 1, 2])[3].apply(list).to_string())

    per_sec = records_per_second(raw_video)
    counts = ", ".join(f"{v} seconds hold {k} records" for k, v in per_sec.items())
    print(
        f"\nVideo seconds, first and last of each fix skipped: {counts} "
        f"(20 records a second is 20 Hz)"
    )

    # The soaring file should be a labeled copy of the migration records.
    copy = migr[[0, 1, 2] + SIGNAL].rename(columns={c: f"m{c}" for c in SIGNAL})
    both = soar.merge(copy, on=[0, 1, 2])
    same = np.allclose(
        both[SIGNAL].values, both[[f"m{c}" for c in SIGNAL]].values, equal_nan=True
    )
    print(
        f"Soaring file records found in the migration files: {len(both)} of "
        f"{len(soar)}, values identical: {same}"
    )

    print("\n## Video fixes: labels per fix\n")
    n_lab = video.groupby([0, 1])[3].nunique().value_counts().sort_index()
    print(", ".join(f"{k} label(s): {v} fixes" for k, v in n_lab.items()))
    lens = video.groupby([0, 1]).size()
    print(f"records per fix: median {lens.median():.0f}, max {lens.max()}")

    print(f"\n## Segments of {cfg.glen} records per behaviour\n")
    seg_v, bouts_v = cut_segments(video, cfg.glen)
    seg_s, bouts_s = cut_segments(soar, cfg.glen)
    seg = pd.concat([seg_v, seg_s])
    table = per_label(seg, pd.concat([bouts_v, bouts_s]), cfg.glen)
    table.insert(0, "name", [names.get(k, "") for k in table.index])
    print(as_markdown(table.rename_axis("label"), index=True))
    absent = [names[k] for k in names if k not in table.index]
    print(f"\nNo records: {', '.join(absent)}")
    print(
        f"{names[cfg.soar_label]}: {(seg_v.label == cfg.soar_label).sum()} "
        f"segments from video, {len(seg_s)} from {cfg.soar_file}"
    )

    print("\n## Segments per behaviour and bird\n")
    per_bird = seg.pivot_table(
        index="label", columns="device", values="start", aggfunc="count", fill_value=0
    )
    per_bird.insert(0, "name", [names[k] for k in per_bird.index])
    per_bird.columns = [str(c) for c in per_bird.columns]
    print(as_markdown(per_bird, index=True))

    print("\n## Migration files (no labels)\n")
    seg_m, _ = cut_segments(migr, cfg.glen)
    soar_fixes = soar.groupby(0)[1].nunique()
    rows = []
    for bird, df in migr.groupby(0):
        lens = df.groupby(1).size()
        mode = lens.mode()[0]
        rows.append(
            {
                "bird": bird,
                "first": df[1].min(),
                "last": df[1].max(),
                "fixes": len(lens),
                f"fixes of {mode} records": int((lens == mode).sum()),
                "segments": int((seg_m.device == bird).sum()),
                "complete": int(seg_m[seg_m.device == bird]["complete"].sum()),
                "fixes in soar file": int(soar_fixes.get(bird, 0)),
            }
        )
    print(as_markdown(pd.DataFrame(rows)))

    begging_report(cfg)


if __name__ == "__main__":
    config = {
        "data_dir": "/home/fatemeh/Downloads/bird/data/spoonbill",
        "video_file": "video_annotated_data.csv",
        "soar_file": "visual_annotated_passive_flight_data.csv",
        "names_file": "behaviours.csv",
        "migration_prefix": "acc_data_migration_",
        "birds": [6283, 6287, 6291],  # one migration file per bird
        "soar_label": 9,  # fly-soar, the only label of soar_file
        "glen": 20,  # records per segment, as for the gull model
        # Lok et al. 2025, the Zenodo release extracted as is
        "begging_dir": "/home/fatemeh/Downloads/bird/data/spoonbill_postfledging/"
        "tamarlok-postfledging_care-6c40d9b/data/raw",
        "begging_file": "annotated.begging.data.csv",
    }
    main(OmegaConf.create(config))
