# Presentation figures

The talk itself is `presentation.md`, and `update.md` is the meeting-update
deck, which is written by hand. This file is only about the mechanics: where
the figures and numbers come from, and how to remake them. Both decks render
the same way.

## Figures

Both live in `/home/fatemeh/Downloads/bird/screenshots/`.

| file | what it shows | how to remake |
|---|---|---|
| `behavior_classes.png` | one example burst per class, with its GPS speed | `behavior.utils.plot_one` on one burst per label; the window titles are matplotlib's, screenshotted as a 3x3 grid |
| `gps_burst_labeling_viz_app.png` | the labelling app: IMU trace, map, editable label | screenshot of `app/gps_burst_labeling_viz_app.py` |

Only `behavior_classes.png` is used in the current slides. The app screenshot is
kept for a slide about how the data is checked by eye, if that is ever wanted.

Also available but not currently used, per run in
`/home/fatemeh/Downloads/bird/results/failed/<exp>_starts/`:

- `confusion_matrix_{train,valid}.png`
- `per_class_metrics_{,balanced_}{train,valid}.csv`

Written automatically by `scripts/batch_train_supervised.py`; no extra step.

## Numbers

Every number in the talk comes from one of these. Rerun the script to refresh.

| slide | source |
|---|---|
| accuracy, AP, loss, per-class F1, robustness by orientation | `exps/eval_labeled.py` (`model_exps: [194, 195]` for the talk's orientation table, `[194, 196]` for per-class) |
| unlabelled comparison, confidence, impossible speed/place/rotation flip | `exps/eval_unlabeled.py` (`model_exps: [194, 196]`) |
| bad labels, 26 of 4,338 | `exps/find_label_noise.py` |
| training time 47 min → 5 min | wall clock in `~/Downloads/bird/results/tensorboard/{195,196}.txt` |
| valid/train accuracy per run | `~/Downloads/bird/results/failed/<exp>_starts/app_loss_acc.txt` |

Each of those scripts takes its settings from a config dict under
`if __name__ == "__main__":` — edit in place, no command-line flags.

The reasoning behind each conclusion is in `docs/lesson_learned.md`; the talk
gives only the conclusion.

## How to present this

`presentation.md` is written for **Marp**, which turns markdown into slides.
`---` starts a new slide and `![h:400](...)` sets image height in pixels. It is
not installed here.

Easiest, inside VS Code:

1. Install the extension **Marp for VS Code** (`marp-team.marp-vscode`).
2. Open `presentation.md`. The preview pane shows slides.
3. `Ctrl+Shift+P` then **Marp: Export Slide Deck** gives PDF, PPTX or HTML.

In Ubuntu’s Document Viewer (Evince), Press F5 to start presentation mode.

From the command line instead:

    npm install -g @marp-team/marp-cli
    marp presentation.md --pptx      # or --pdf, or --html

Images are embedded on export, so the absolute paths above do not have to travel
with the file.
