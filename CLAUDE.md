# Working notes

## Rules

- Commit messages: never mention Claude or Claude Code (no attribution, no
  Co-Authored-By line, no tool names).
- Always update the related documents in the same change (see Documentation).
- Never run `git commit`. Prepare the change and the message; the user commits.
- Ask for permission before running an experiment (training run). Running one
  costs GPU time and hours.
- Always call python by its full path: `/home/fatemeh/miniconda3/envs/bird/bin/python`.
  Not bare `python`, not `conda run -n bird`. VS Code puts one env's `bin` at the
  front of `PATH` for every tool call, and conda does not know it is there, so
  `conda run` and `conda activate` can silently run the wrong env. A full path
  cannot be shadowed.

## Workspace

- Scratch/temporary files: use `/home/fatemeh/Downloads/bird/claude`, not `/tmp`
  (which is wiped on shutdown). Clean up whatever is not worth keeping.
- Images worth keeping (figures for a presentation, screenshots): save to
  `/home/fatemeh/Downloads/bird/screenshots`.
- Settings/config for this project live in `~/dev/bird-behavior/.claude` (for
  example `settings.json`), never another repo's `.claude` in this VS Code
  workspace.

## Documentation

Keep these current whenever the code they describe changes:

- `docs/description.md`: overview of the data, models, and every script. Keep it
  polished.
- `docs/lesson_learned.md`: curated lessons that generalize, with the reason
  behind each.
- `docs/experiments_log.md`: terse per-run notebook. Held to a lower bar than
  the other two.

After a change: update the script's top docstring and `docs/description.md` for
new or changed behavior; add to `docs/lesson_learned.md` for a bug, surprise, or
design decision (with the reason); add to `docs/experiments_log.md` for a
finished or planned run.

## Experiments

- `scripts/batch_train_supervised.py`: supervised batch runner; the experiments
  are defined in `iter_batch_configs()`. It applies the rotation augmentation to
  the training set only. Current run is exp195: `BirdModelSmallDilated`
  (9 classes, `starts.csv`, 4000 epochs), the rotation A/B against exp194 (same
  model, no aug). Writes to `save_path` (`~/Downloads/bird/results`).
- Uses CUDA if available, else CPU. The small 3-conv model trains in ~20 min on a
  laptop GPU.
