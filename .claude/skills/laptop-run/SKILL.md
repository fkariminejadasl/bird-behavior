---
name: laptop-run
description: Run a bird-behavior supervised experiment on the laptop and compare it against a previous one. Use when the user wants to train locally or launch an experiment, e.g. "run exp196", "run a new experiment and compare with exp195", "train this config", "run batch_train_supervised".
---

# Run a supervised experiment on the laptop

Everything runs through `scripts/batch_train_supervised.py`. Results go to
`/home/fatemeh/Downloads/bird/results` (`save_path`).

**Ask permission before launching, unless the user just asked for the run.**
It costs GPU time. Never `git commit`.

## 1. Pick a free experiment number

Numbers are global, and reusing one silently overwrites its results:

```bash
ls -d /home/fatemeh/Downloads/bird/results/<n>_best.pth \
      /home/fatemeh/Downloads/bird/results/failed/<n>_* 2>/dev/null
```

Nothing listed means `<n>` is free.

## 2. Edit the config

`base_config` and `experiments` both live under `if __name__ == "__main__":`.
No argparse — edit in place and leave it uncommitted until the user commits.

- Bump `"exp"` and update the comment above `experiments` to say what the run tests.
- **`experiments` is a list: add more dicts to train several configs in one go**,
  sequentially, each overriding `base_config`. That is what the script is for.
- Augmentation is the `transforms` object in `main()`. Use the batched `bau.Batch*`
  transforms — `GpuBatches` hands them a channel-first `(N, C, T)` batch, so the
  per-sample classes will not work. `transforms = None` disables augmentation.

## 3. Launch in the background

~5 min for 4000 epochs with `BirdModelSmallDilated` on `starts.csv`. Log to
`tensorboard/<exp>.txt`, beside the tensorboard dir:

```bash
/home/fatemeh/miniconda3/envs/bird/bin/python scripts/batch_train_supervised.py \
  > /home/fatemeh/Downloads/bird/results/tensorboard/<exp>.txt 2>&1
```

Confirm it started before waiting on it — line 4 shows the split:

```bash
head -4 /home/fatemeh/Downloads/bird/results/tensorboard/<exp>.txt | tail -1
# Device: cuda:0, Train samples: 3,900, Validation samples: 438, ...
```

## 4. Read the results

Per run, under `/home/fatemeh/Downloads/bird/results/`:

- `failed/<exp>_<data stem>/app_loss_acc.txt` — the headline train/valid numbers
- `failed/<exp>_<data stem>/per_class_metrics_{,balanced_}{train,valid}.csv`
- `failed/<exp>_<data stem>/confusion_matrix_{train,valid}.png`
- `<exp>_best.pth` — best-validation checkpoint
- `tensorboard/<exp>/` — scalars; `tensorboard/<exp>.txt` — the stdout log

`failed/` is just where per-run metrics land. It does not mean the run failed.

## 5. Compare against a previous run

```bash
/home/fatemeh/miniconda3/envs/bird/bin/python exps/eval_labeled.py
/home/fatemeh/miniconda3/envs/bird/bin/python exps/eval_unlabeled.py
```

Both take the experiment pair from the config dict in their `__main__` — edit it,
then put it back to the documented pair when done. `eval_labeled.py`'s accuracy,
AP and loss must reproduce each run's `app_loss_acc.txt`; if they do not, the
split is wrong and nothing below it is trustworthy.

## 6. Update the docs (same change, always)

- `docs/experiments_log.md`: new entry **at the top**, terse, with the git hash.
- `docs/lesson_learned.md`: only if the run taught something that generalizes.
- `docs/descriptions.md`: only if script behavior changed.

## Gotchas

- **The headline val is optimistically biased.** `app_loss_acc.txt` comes from
  reloading `<exp>_best.pth` — the max over 4000 noisy epochs on a 438-sample
  split — so it sits above the plateau. Check
  `grep "valid: epoch" <exp>.txt | tail -500` before calling a peak real.
- **Seeds and data slices move validation by ~2.5 points.** Do not read a small
  gap between two runs as an effect. Manouvre and ExFlap have 16 and 4 valid
  bursts, so their per-class F1 swings on one or two samples.
- **Two runs compare only if one thing changed.** Diff the configs and record the
  hash each ran at.
