"""Which behaviours follow one another inside a single recording?

A GPS fix holds up to ten consecutive one-second bursts. `exps/eval_unlabeled.py`
counts a prediction change between two of them as instability, but some changes
are perfectly normal: a gull walks, pecks, walks again. Treating those as errors
punishes a model for being right.

This measures which transitions actually occur, from the labeled data itself.
It is the reproducible version of `test_labels_comes_together` in
`tests/test_data_processing.py`, which recorded the same counts in a comment.

Two views of the same thing:

- **co-occurrence** -- which labels share a fix at all, ignoring order. This is
  the number the old test printed, kept so the two can be compared.
- **transitions** -- which label directly follows which, between neighbouring
  label ranges of a fix. This is what `eval_unlabeled.py` needs, since it looks
  at consecutive bursts.

Transitions are symmetrised before being written: Pecking -> TerLoco and
TerLoco -> Pecking are the same claim about what a gull does, and with counts
this small the direction is not reliably estimated.

Writes `label_transitions.csv` (from, to, count) next to the data.

  /home/fatemeh/miniconda3/envs/bird/bin/python exps/label_cooccurrence.py
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from behavior import data_processing as bdp
from behavior import utils as bu


def label_ranges_per_fix(df):
    """{(device, datetime): [(start, end, label), ...]} sorted by start index.

    Same result as `data_processing.get_label_ranges_per_dt` per fix, but by one
    groupby instead of a full scan per fix, which matters at 200k rows.
    """
    df = df[df[3] != -1].sort_values([0, 1, 2])
    out = {}
    for key, grp in df.groupby([0, 1], sort=False):
        idx = grp[2].values
        lab = grp[3].values
        # a new range starts where the label changes or the index jumps
        starts = np.flatnonzero(np.r_[True, (np.diff(idx) != 1) | (np.diff(lab) != 0)])
        ends = np.r_[starts[1:] - 1, len(idx) - 1]
        out[key] = [(idx[s], idx[e], int(lab[s])) for s, e in zip(starts, ends)]
    return out


def fixes_with_labels(ranges, labels):
    """Every (device, datetime) whose fix contains all of `labels`.

    Note `ranges` maps a fix to a list of `(start, end, label)` tuples, so
    `(1, 5, 1) in v` asks for a range spanning index 1 to 5 with label 1 -- not
    for labels 1 and 5 being present. Use this instead.
    """
    want = set(labels)
    return {k: v for k, v in ranges.items() if want <= {lab for _, _, lab in v}}


def check_against_reference(df, ranges, n_checks):
    """Verify the fast path reproduces get_label_ranges_per_dt on a few fixes."""
    keys = list(ranges)[:n_checks]
    for key in keys:
        expected = bdp.get_label_ranges_per_dt(df, key)
        got = {(s, e): lab for s, e, lab in ranges[key]}
        if expected != got:
            raise AssertionError(f"mismatch at {key}: {expected} != {got}")
    return len(keys)


def main(cfg):
    df = pd.read_csv(cfg.data_file, header=None)
    ranges = label_ranges_per_fix(df)
    n = check_against_reference(df, ranges, cfg.n_reference_checks)
    print(
        f"{len(df):,} rows, {len(ranges):,} fixes "
        f"({n} verified against get_label_ranges_per_dt)\n"
    )

    # co-occurrence: which labels share a fix, order ignored
    cooc = Counter(
        tuple(sorted({lab for _, _, lab in v}))
        for v in ranges.values()
        if len({lab for _, _, lab in v}) > 1
    )
    print("Labels sharing a fix (the old test_labels_comes_together number):")
    for labels, count in cooc.most_common():
        print(f"  {count:>3}  " + " + ".join(bu.ind2name[i] for i in labels))

    # transitions: which label directly follows which, symmetrised
    trans = Counter()
    for v in ranges.values():
        for (_, _, a), (_, _, b) in zip(v, v[1:]):
            if a != b:
                trans[tuple(sorted((a, b)))] += 1

    rows = [
        {
            "from_label": a,
            "to_label": b,
            "from_name": bu.ind2name[a],
            "to_name": bu.ind2name[b],
            "count": c,
        }
        for (a, b), c in sorted(trans.items(), key=lambda kv: -kv[1])
    ]
    out = pd.DataFrame(rows)
    print(
        f"\n{len(out)} label pairs are ever adjacent inside a fix "
        f"(of {9 * 8 // 2} possible):"
    )
    print(out[["from_name", "to_name", "count"]].to_string(index=False))

    # A rare pair is worth eyeballing, so say exactly where it came from.
    rare = out[out["count"] <= cfg.show_fixes_below]
    if len(rare):
        print(
            f"\nPairs seen {cfg.show_fixes_below} time(s) or fewer, and the "
            f"fixes they come from:"
        )
        for row in rare.itertuples():
            print(f"  {row.from_name} + {row.to_name}  (count {row.count})")
            for key, spans in fixes_with_labels(
                ranges, (row.from_label, row.to_label)
            ).items():
                inside = ", ".join(f"{s}-{e} {bu.ind2name[lab]}" for s, e, lab in spans)
                print(f"    device {key[0]}, {key[1]}: {inside}")

    dest = Path(cfg.out_dir) / cfg.out_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"\nwritten to {dest}")
    print(
        f"Pairs at or above min_count={cfg.min_count} are the plausible "
        f"transitions: {int((out['count'] >= cfg.min_count).sum())} of {len(out)}"
    )


if __name__ == "__main__":
    config = {
        # per-item labeled data, before slicing into bursts
        "data_file": "/home/fatemeh/Downloads/bird/data/final/combined.csv",
        "out_dir": "/home/fatemeh/Downloads/bird/data/final",
        "out_name": "label_transitions.csv",
        "n_reference_checks": 200,
        "min_count": 3,  # below this, a pair is too rare to call normal
        "show_fixes_below": 3,  # list the fixes behind pairs this rare
    }
    main(OmegaConf.create(config))
