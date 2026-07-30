---
name: laptop-run
description: Run a bird-behavior supervised experiment on the laptop and compare it against a previous one. Use when the user wants to train locally, e.g. "run exp196", "run a new experiment and compare with exp195", "train this config".
---

# Run a supervised experiment on the laptop

Everything runs through `scripts/batch_train_supervised.py`, which trains one
config per entry in the `experiments` list. Results go to
`/home/fatemeh/Downloads/bird/results` (`save_path`).

**Ask permission before launching, unless the user just asked for the run.**
It costs GPU time. Never `git commit`.

## 1. Pick a free experiment number

Numbers are global and reused numbers silently overwrite results:

```bash
ls -d /home/fatemeh/Downloads/bird/results/<n>_best.pth \
      /home/fatemeh/Downloads/bird/results/failed/<n>_* 2>/dev/null
```

Nothing listed means `<n>` is free.

## 2. Edit the config

Both `base_config` and `experiments` live under `if __name__ == "__main__":`.
No argparse — edit in place, leave the change uncommitted until the user commits.

- Bump `"exp"` in the `experiments` entry.
- Update the comment above `experiments` to say what this run is testing.
- Augmentation is the `transforms` object in `main()`. Use the **batched**
  `bau.Batch*` transforms — `GpuBatches` hands them a channel-first `(N, C, T)`
  batch, so the per-sample classes will not work there.
- `transforms = None` disables augmentation.

## 3. Launch in the background

~5.5 min for 4000 epochs with `BirdModelSmallDilated` on `starts.csv`. Use the
full python path (`conda run`/`activate` can silently pick the wrong env), and
log to `tensorboard/<exp>.txt` next to the tensorboard dir:

```bash
/home/fatemeh/miniconda3/envs/bird/bin/python scripts/batch_train_supervised.py \
  > /home/fatemeh/Downloads/bird/results/tensorboard/<exp>.txt 2>&1
```

Confirm it started before waiting on it — the header line should show the split:

```bash
head -6 /home/fatemeh/Downloads/bird/results/tensorboard/<exp>.txt
# Device: cuda:0, Train samples: 3,900, Validation samples: 438, ...
```

## 4. Read the results

Per run, in `/home/fatemeh/Downloads/bird/results/`:

- `<exp>_best.pth` — best-validation checkpoint
- `failed/<exp>_<data stem>/app_loss_acc.txt` — the headline train/valid numbers
- `failed/<exp>_<data stem>/per_class_metrics_{,balanced_}{train,valid}.csv`
- `failed/<exp>_<data stem>/confusion_matrix_{train,valid}.png`
- `tensorboard/<exp>/` — scalars; `tensorboard/<exp>.txt` — the stdout log

`failed/` is just where per-run metrics land. It does not mean the run failed.

## 5. Compare against the previous run

```bash
/home/fatemeh/miniconda3/envs/bird/bin/python exps/compare_per_class_metrics.py
/home/fatemeh/miniconda3/envs/bird/bin/python exps/eval_rotation_robustness.py
```

Both take the experiment pair from the config dict in their `__main__` — edit it.
The second is only meaningful for orientation-augmentation runs, and its `clean`
row must reproduce each run's `app_loss_acc.txt` or the comparison is not sound.

## 6. Update the docs (same change, always)

- `docs/experiments_log.md`: new entry **at the top**, terse.
- `docs/lesson_learned.md`: only if the run taught something that generalizes,
  with the reason. Keep both short.
- `docs/description.md`: only if script behavior changed.

## Gotchas learned the hard way

- **The reported val is optimistically biased.** `app_loss_acc.txt` comes from
  reloading `<exp>_best.pth`, the max over 4000 noisy epochs on a 438-sample
  split, so it sits above the last epoch's number. Check the plateau in the log
  (`grep "valid: epoch" <exp>.txt`) before treating a peak as real.
- **Seeds and data slices move validation by ~2.5 points.** Do not read a small
  difference between two runs as a real effect.
- **Two runs are only comparable if one thing changed.** Diff the configs, and
  check the git hash each was run at; record it in the log entry.
- **RNG differs between the per-sample and batched augmentation paths**, so a
  repeat through `GpuBatches` will not be bit-identical to an older run — expect
  agreement within seed noise, not equality.
- A `DataLoader` with `num_workers > 0` needs its caller under a `__main__`
  guard; without it the forkserver re-imports the module and dies with
  `ConnectionResetError`.
- The run is a local process and dies if the laptop sleeps. Keep it awake.
