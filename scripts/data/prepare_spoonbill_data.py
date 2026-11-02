"""Write the spoonbill training file, in the `starts.csv` format.

Reads the labeled spoonbill data of both papers (`docs/descriptions.md`,
Spoonbill data) and writes one CSV that `data.py::load_csv_pandas` reads like
`starts.csv`: no header, 8 columns (device, datetime, index, label, x, y, z,
speed), whole 20-record segments, one after the other.

- Lok et al. 2023: `video_annotated_data.csv` and
  `visual_annotated_passive_flight_data.csv` (fly-soar), with their own label
  codes, 1-16 (`behaviours.csv`).
- Lok et al. 2025: `annotated.begging.data.csv`, text labels mapped as that
  paper maps them: beg_flap to beg, preen_tag to stand-preen (4),
  shake_feathers to stand-shake-feathers (15). beg is the new code 17. Its
  `Index` starts at 1; it is written from 0, as in the other files.

The labels are mapped first, then the segments are cut, so a beg_flap bout next
to a beg bout joins it. Segments are cut as `exps/spoonbill_data_stats.py`
cuts them (`cut_segments`): a bout of n records gives n // 20 segments. A record
listed twice in one fix is dropped, both copies. Only complete segments are
written: `load_csv_pandas` drops a missing speed, but a missing x, y or z would
go into training. Values are written with 6 decimals, as in `starts.csv`; the
2 g clip and the 30 m/s cut are left to the loader, as for the gulls.

Prints the segments per label and per bird, and checks the file reads back as
whole segments.

  /home/fatemeh/miniconda3/envs/bird/bin/python scripts/data/prepare_spoonbill_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exps"))
import spoonbill_data_stats as sds  # noqa: E402

from behavior import data as bd  # noqa: E402


def load_begging(path, label_map):
    """The begging file in the gull column order, labels as codes."""
    df = pd.read_csv(path)
    label_map = dict(label_map)
    unknown = set(df["behaviour"]) - set(label_map)
    assert not unknown, f"no code for {unknown}"
    df["behaviour.index"] = df["behaviour"].map(label_map)
    df["Index"] = df["Index"] - 1
    return df.rename(columns=sds.COLUMNS)


def complete_segments(df, glen):
    """The records of every complete segment, in segment order."""
    df, dropped = sds.drop_conflicts(df)
    segments, _ = sds.cut_segments(df, glen)
    segments = segments[segments["complete"]].reset_index(drop=True)
    # one row per record: segment s, record offset k
    rows = segments.loc[segments.index.repeat(glen), ["device", "datetime", "start"]]
    rows["order"] = np.arange(len(rows))
    rows[2] = rows["start"] + np.tile(np.arange(glen), len(segments))
    rows = rows.rename(columns={"device": 0, "datetime": 1})
    out = rows.merge(df, on=[0, 1, 2], how="left").sort_values("order")
    assert len(out) == len(rows) and not out[sds.SIGNAL].isna().any().any()
    return out[list(range(8))], len(dropped)


def main(cfg):
    data = Path(cfg.data_dir)
    sources = {
        cfg.video_file: pd.read_csv(data / cfg.video_file).rename(columns=sds.COLUMNS),
        cfg.soar_file: sds.load(data / cfg.soar_file),
        cfg.begging_file: load_begging(
            Path(cfg.begging_dir) / cfg.begging_file, cfg.begging_map
        ),
    }
    parts = []
    for name, df in sources.items():
        part, n_dropped = complete_segments(df, cfg.glen)
        print(
            f"{name}: {len(part) // cfg.glen} complete segments, "
            f"{n_dropped} records dropped as listed twice"
        )
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)

    # each block of glen rows is one segment: one fix, one label, consecutive
    blocks = out.values.reshape(-1, cfg.glen, 8)
    assert (blocks[:, :, [0, 1, 3]] == blocks[:, :1, [0, 1, 3]]).all()
    assert (np.diff(blocks[:, :, 2].astype(int), axis=1) == 1).all()

    out_file = Path(cfg.out_file)
    assert not out_file.exists(), f"{out_file} exists; write under a new name"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, header=False, index=False, float_format="%.6f")
    print(f"\nWrote {len(out)} records, {len(blocks)} segments to {out_file}")

    names = pd.read_csv(data / cfg.names_file)
    names = dict(zip(names["behaviour.index"], names["behaviour"])) | {17: "beg"}
    seg = pd.DataFrame(blocks[:, 0, [0, 3]], columns=["bird", "label"])
    per_bird = pd.crosstab(
        seg["label"], seg["bird"], margins=True, margins_name="total"
    )
    per_bird.insert(0, "name", [names.get(k, "") for k in per_bird.index])
    per_bird.columns = [str(c) for c in per_bird.columns]
    print("\n## Segments per label and bird\n")
    print(sds.as_markdown(per_bird, index=True))

    # read back as training reads it
    labels = sorted(seg["label"].unique())
    igs, ldts = bd.load_csv_pandas(out_file, labels, glen=cfg.glen)
    print(f"\nload_csv_pandas: {igs.shape[0]} segments, labels {labels}")


if __name__ == "__main__":
    config = {
        "data_dir": "/home/fatemeh/Downloads/bird/data/spoonbill",
        "video_file": "video_annotated_data.csv",
        "soar_file": "visual_annotated_passive_flight_data.csv",
        "names_file": "behaviours.csv",
        # Lok et al. 2025, the Zenodo release extracted as is
        "begging_dir": "/home/fatemeh/Downloads/bird/data/spoonbill_postfledging/"
        "tamarlok-postfledging_care-6c40d9b/data/raw",
        "begging_file": "annotated.begging.data.csv",
        # the second paper's map; beg is new, after the 16 codes of behaviours.csv
        "begging_map": {
            "beg": 17,
            "beg_flap": 17,
            "preen_tag": 4,
            "shake_feathers": 15,
        },
        "glen": 20,  # records per segment, as for the gull model
        "out_file": "/home/fatemeh/Downloads/bird/data/spoonbill_final/"
        "spoonbill_starts.csv",
    }
    main(OmegaConf.create(config))
