# Status

Where the project stands, as of 2026-11-17. Read this first. The results tables
live here only. The details of each run are in
[experiments_log.md](experiments_log.md), the reasons in
[lesson_learned.md](lesson_learned.md), and what each script does in
[descriptions.md](descriptions.md).

## Working models

- **Gulls: exp197.** `BirdModelSmallDilated` with 7 input channels (IMU x, y, z,
  GPS speed, and 3 magnitude channels, `add_magnitudes`), trained with a random
  3D rotation of the acceleration, on `starts.csv`. Valid accuracy 94.52% of 438
  valid bursts (`exps/eval_labeled.py`). It stays on the raw labels, not
  `starts_clean.csv`: see "Retraining on the cleaned labels" in
  `lesson_learned.md`.
- exp194, the same model with 4 channels and no rotation, has the highest valid
  accuracy, 96.35%, but falls to 13.05% when the tag is turned at random
  (`exps/eval_labeled.py`).
- **Spoonbills: none chosen yet.** Best so far is the exp197 recipe without
  rotation, split by fix (exp207, exp210, exp211): 93.27-94.65% of 505 valid
  segments at epoch 4000 (`exps/eval_spoonbill.py`).

## How the numbers are measured

Data and split:

- **Gulls.** `starts.csv`: 4,338 bursts of 20 records (1 s), 9 classes. In the
  gull text a burst is one model input: a segment in the Terms of
  `descriptions.md`. Split 90/10 within each class
  (`behavior/utils.py::stratified_split`, seed 32984): 3,900 train and 438 valid
  bursts. exp199-204 are scored on 437: the valid bursts that `starts.csv` and
  `starts_clean.csv` share.
- **Spoonbills.** `spoonbill_starts.csv`: 5,080 segments of 20 records, 14
  labels. Split by fix (`split_by` fix): 4,575 train segments from 702 fixes,
  505 valid segments from 79 fixes. The same birds are in train and valid.
- **Checkpoint.** `<exp>_best.pth` is the epoch with the best valid accuracy, so
  it can be a lucky epoch. From exp199 on, runs are compared at epoch 4000,
  `<exp>_4000.pth` (`save_final`).

Measures, all on the valid split:

- **Accuracy, %**: the share of valid bursts (or segments) given the right
  class.
- **F1 per class, 0-1**: 2TP / (2TP + FP + FN) over the valid bursts
  (`behavior/utils.py::per_class_statistics`).
- **Class-balanced F1 per class, 0-1**: the same, after weighting the confusion
  matrix so that every class has the same number of valid bursts, 438 / 9, about
  49 (`per_class_statistics_balanced`). ExFlap, with 4 valid bursts, is weighted
  12x.
- **Pooled F1**: the `Combined` row of the training CSVs, and `valid_F1` and
  `valid_F1_bal` of `exps/eval_labeled.py`. The plain one equals accuracy / 100.
  The class-balanced one equals the mean recall over the classes. Checked on
  exp194-197: for exp197, plain 0.95 against 94.52%, class-balanced 0.90
  against a mean recall of 0.898.
- **Accuracy at another orientation, %**: valid accuracy with the accelerometer
  axes changed: x and y swapped, tilted 20-90 degrees, or a random 3D rotation,
  mean of 20 (`exps/eval_labeled.py`).

Label-free checks, on the 115,266 bursts of device 6004, which has no labels
(`exps/eval_unlabeled.py`). Each counts predictions that contradict something
known without the model:

- **Speed conflict, % of bursts**: the class contradicts the GPS speed, by the
  Table S1 rules of `exps/find_label_noise.py`.
- **Place conflict, % of bursts**: walking or pecking at sea, floating or on a
  boat inland. Bursts within about 2 km of a coast are never counted; 53% of the
  bursts can be.
- **Unseen flip, % of pairs** of consecutive bursts in one fix: the class
  changes to one never seen next to it in the labeled data (9 of 36 pairs are
  seen at least 3 times, `exps/label_cooccurrence.py`).
- **Rotation flip, %**: the share of 10 random 3D rotations that change a
  burst's class, mean over the bursts.
- **Mean confidence, 0-1**: the mean softmax probability of the predicted class.

Spoonbills against the papers (`exps/eval_spoonbill.py`): F1, 0-1, per class
pooled as each paper pools, scored on records the papers' way. Each valid fix is
cut into 1 s windows from its first record, and every record is scored against
its own label: 602 windows, 12,040 records. How the papers score:
`descriptions.md`, How the papers score.

Across species (`exps/eval_cross_species.py`): each model on every segment of
the other species, all 5080 spoonbill segments or all 4338 gull bursts. What
counts as right comes from an ecologist's table (`descriptions.md`, Spoonbill
to gull labels); the spoonbill models are scored with it read backwards.

- **% expected**: the share of a class's segments given the class the table
  expects.
- **% expected or alternative**: or the class it gives as the alternative.

## Results

### Gulls, labeled

438 valid bursts, `<exp>_best.pth` (`exps/eval_labeled.py`).

| exp | what changed                  | accuracy % | class-balanced F1, pooled | accuracy %, x/y swapped | accuracy %, random rotation |
| --- | ----------------------------- | ---------- | ------------------------- | ----------------------- | --------------------------- |
| 194 | no augmentation, 4 channels   | 96.35      | 0.92                      | 71.46                   | 13.05                       |
| 195 | + random rotation in training | 92.24      | 0.84                      | 90.41                   | 90.67                       |
| 196 | exp195 rerun, rotation on GPU | 92.69      | 0.83                      | 93.15                   | 92.03                       |
| 197 | exp196 + 3 magnitude channels | 94.52      | 0.90                      | 94.52                   | 93.66                       |

### Gulls, label-free

Device 6004, 115,266 bursts, `<exp>_best.pth` (`exps/eval_unlabeled.py`).
exp195 was not scored.

| exp | mean confidence | speed conflict % | place conflict % | unseen flip % | rotation flip % |
| --- | --------------- | ---------------- | ---------------- | ------------- | --------------- |
| 194 | 0.89            | 1.45             | 1.18             | 6.66          | 82.76           |
| 196 | 0.91            | 0.09             | 1.45             | 4.92          | 5.37            |
| 197 | 0.95            | 0.11             | 1.09             | 4.04          | 1.72            |

### Gulls, raw against cleaned labels

Three seeds a side (32984, 1234, 5678), the split held at seed 32984, read at
epoch 4000. Accuracy on 437 valid bursts (`exps/eval_labeled.py`), the rest on
device 6004 (`exps/eval_unlabeled.py`). Only speed conflict separates the two:
its ranges do not overlap.

| labels                    | exp           | accuracy %          | speed conflict % | place conflict % | unseen flip %    | rotation flip %  |
| ------------------------- | ------------- | ------------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| raw, `starts.csv`         | 199, 201, 202 | 93.82, 94.05, 94.05 | 0.10, 0.15, 0.13 | 1.12, 1.08, 1.00 | 4.09, 3.97, 3.46 | 1.57, 1.62, 1.34 |
| clean, `starts_clean.csv` | 200, 203, 204 | 93.59, 93.36, 94.05 | 0.03, 0.06, 0.04 | 1.03, 1.07, 1.04 | 3.88, 3.86, 3.44 | 1.61, 1.53, 1.36 |

### Spoonbills

505 valid segments, epoch 4000, seeds 32984, 1234, 5678
(`exps/eval_spoonbill.py`).

| rotation in training | exp           | accuracy %          |
| -------------------- | ------------- | ------------------- |
| yes                  | 206, 208, 209 | 84.55, 84.55, 83.96 |
| no                   | 207, 210, 211 | 94.26, 93.27, 94.65 |

Against the papers: F1, 0-1, on records, no rotation, mean of the 3 seeds
(`exps/eval_spoonbill.py`). Their split shares fixes between train and test;
ours does not.

| F1, groups of Lok et al. 2025 | forage | walk | fly  | rest | beg  |
| ----------------------------- | ------ | ---- | ---- | ---- | ---- |
| Lok et al. 2025, 1.6 s        | 0.96   | 0.61 | 0.98 | 0.95 | 0.89 |
| ours, 1 s                     | 0.98   | 0.74 | 0.97 | 1.00 | 0.92 |

| F1, classes of Lok et al. 2023 | walk | search | prey ingestion |
| ------------------------------ | ---- | ------ | -------------- |
| Lok et al. 2023, best length   | 0.65 | 0.91   | 0.73           |
| ours, 1 s                      | 0.81 | 0.90   | 0.59           |

### Across species

The gull model, exp197 (`197_best.pth`), on all 5080 spoonbill segments
(`exps/eval_cross_species.py`). Other is not a class of the gull model, so a
label mapped to it scores only through its alternative.

| spoonbill label      | segments    | expected gull class (alternative) | % expected | % expected or alternative |
| -------------------- | ----------- | --------------------------------- | ---------- | ------------------------- |
| stand-rest           | 1357        | SitStand                          | 98.7       | 98.7                      |
| stand-preen          | 299         | Other (SitStand)                  | -          | 76.9                      |
| sit-alert            | 62          | SitStand                          | 100.0      | 100.0                     |
| stand-alert          | 416         | SitStand                          | 93.0       | 93.0                      |
| fly-flap             | 185         | Flap, ExFlap, Manouvre            | 98.4       | 98.4                      |
| fly-soar             | 52          | Soar                              | 98.1       | 98.1                      |
| walk                 | 177         | TerLoco                           | 67.2       | 67.2                      |
| for-search           | 1929        | Pecking (TerLoco, Other)          | 51.7       | 78.9                      |
| for-handle           | 14          | Pecking (Other)                   | 100.0      | 100.0                     |
| for-intake           | 115         | Pecking (Other)                   | 100.0      | 100.0                     |
| stand-other          | 15          | Other (SitStand)                  | -          | 53.3                      |
| stand-shake-feathers | 18          | Other (SitStand)                  | -          | 27.8                      |
| drink                | 7           | SitStand (Other, Pecking)         | 28.6       | 28.6                      |
| beg                  | 434         | not in the table                  | -          | -                         |
| **all scored**       | 4314 / 4646 |                                   | **75.8**   | **86.9**                  |

The spoonbill models on all 4338 gull bursts, epoch 4000, mean (min-max) of
three seeds: exp207, 210, 211 without rotation, exp206, 208, 209 with
(`exps/eval_cross_species.py`). Float and Boat have no spoonbill counterpart.

| gull class     | bursts | expected spoonbill label (alternative)                                           | no rotation, % expected | no rotation, % expected or alternative | rotation, % expected | rotation, % expected or alternative |
| -------------- | ------ | -------------------------------------------------------------------------------- | ----------------------- | -------------------------------------- | -------------------- | ----------------------------------- |
| Flap           | 643    | fly-flap                                                                         | 61.1 (46.2-84.1)        | 61.1 (46.2-84.1)                       | 71.7 (48.8-96.7)     | 71.7 (48.8-96.7)                    |
| ExFlap         | 38     | fly-flap                                                                         | 55.3 (31.6-78.9)        | 55.3 (31.6-78.9)                       | 64.0 (60.5-65.8)     | 64.0 (60.5-65.8)                    |
| Soar           | 537    | fly-soar                                                                         | 43.7 (37.2-52.1)        | 43.7 (37.2-52.1)                       | 76.4 (75.2-77.1)     | 76.4 (75.2-77.1)                    |
| SitStand       | 1502   | stand-rest, sit-alert, stand-alert, drink (stand-preen, -other, -shake-feathers) | 82.1 (72.3-88.2)        | 92.1 (85.2-95.7)                       | 85.6 (84.6-87.1)     | 95.2 (94.0-96.6)                    |
| TerLoco        | 337    | walk (for-search)                                                                | 22.7 (12.5-38.9)        | 29.1 (20.5-45.7)                       | 2.3 (1.5-3.0)        | 60.9 (60.5-61.1)                    |
| Manouvre       | 151    | fly-flap                                                                         | 12.4 (9.3-15.2)         | 12.4 (9.3-15.2)                        | 31.8 (11.3-58.9)     | 31.8 (11.3-58.9)                    |
| Pecking        | 225    | for-search, for-handle, for-intake (drink)                                       | 63.6 (60.4-66.7)        | 63.6 (60.4-66.7)                       | 68.1 (63.1-74.2)     | 68.1 (63.1-74.2)                    |
| **all scored** | 3433   |                                                                                  | 61.8 (60.7-63.5)        | 66.7 (65.8-67.4)                       | 69.6 (65.6-74.9)     | 79.6 (75.5-84.7)                    |

## Done

- Labeled data: one pipeline, `starts.csv`, 4,338 bursts
  (`scripts/data/prepare_labeled_data.py`).
- Unlabeled gull data: 24.6 million bursts from 186 devices, in Parquet
  (`exps/inspect_unlabeled_data.py`).
- `BirdModelSmallDilated` replaced `BirdModel` (exp194).
- Rotation augmentation (exp195), then batched on the GPU: 47 min to 5 min per
  run (exp196).
- Magnitude channels (exp197), the working model.
- Label-free checks on unlabeled data (`exps/eval_unlabeled.py`).
- Label noise: 26 bursts whose GPS speed contradicts their label, dropped in
  `starts_clean.csv` (4,312 bursts). Compared over 3 seeds (exp198-204), and for
  reliance on speed (`exps/speed_reliance.py`).
- IMU calibration estimated from the data, for tags without stored values
  (`behavior/data.py::acc_calibrate`). Ornitela bursts take the nearest GPS fix.
- Spoonbills: data counted, training file built, exp205-211, scored against the
  two papers.
- Across species: exp197 on spoonbill data, exp206-211 on gull data, scored
  with an ecologist's label map (`exps/eval_cross_species.py`).

## TODO

Pretraining:

- **A model for several species.** Pretrain a larger, longer-context model on
  unlabeled bursts of several species, then fine-tune only the last layer per
  species with few labels. Fixes are minutes apart, so treat the data as
  separate bursts, not one continuous signal. `exps/chronos2.py` and
  `exps/moment_test.py` extract frozen foundation-model features; no result is
  recorded yet.

Spoonbills:

- **Orientation test.** Run the orientation test of `exps/eval_labeled.py` on
  the runs without rotation, to see how much robustness to the tag's mounting
  they gave up. That script must first learn `split_by` and the spoonbill names.
- **Prey ingestion** (F1 0.59 on records, the paper 0.73). Try a shorter input
  (the 2023 paper is best at 0.4 s), or train on windows that hold two
  behaviours.
- **Class-balanced F1 without a class.** It is NaN when a class has no valid
  segment (drink). Skip such a class instead.
- **One model for both species**, next on branch `spoonbill_gull_training`.
  Train on gull and spoonbill data together and predict combined classes, as
  the label map says (`descriptions.md`, Spoonbill to gull labels). The
  ecologist hopes it splits gull SitStand into rest and alert.

Gulls:

- **Recover walking under rotation.** Estimate the gravity direction per burst,
  turn the burst so it points down, and train on the turned axes. Why: "Lengths
  cannot restore directions" in `lesson_learned.md`.
- **Magnitude channels without rotation**, a counterpart of exp194.
- **Revisit exp197 against the clean-label models** when a second logger, or
  more ExFlap data, can settle it.
- **Test on a real second logger**, not a synthetic rotation of the first.
- **Label about 200 bursts of a new bird**, the only way to get a real accuracy
  there.
- **Run the place and flicker checks on every new deployment.**
- **The other sources of variation**: where the tag sits on the bird, wind,
  water impact.
- **Decide the Ornitela x/y swap** for data that comes through the database
  (`docs/methods.md`, Also to know). `get_data` does not swap for now.

Bugs not fixed yet:

- `behavior/data.py::load_csv_pandas` timestamps are 1000x too small under
  pandas 3 (`lesson_learned.md`).
- `behavior/data.py::prepare_train_valid_dataset` gives one transform to both
  the train and the eval dataset, so an augmentation would also change the eval
  data.

## Dropped

The reasons are in `lesson_learned.md`.

- The model without rotation (exp194) as the working model: it is wrong on a tag
  mounted another way.
- The clean labels (exp200-204) for the working model.
- Jitter, scaling, time and magnitude warp (exp126-130): all below no
  augmentation.
- Label-noise rules on the accelerometer shape: too many false flags. Only the
  GPS-speed rules are kept.
- `BirdModelWideRF` (exp193): 1.14 points below `BirdModelSmallDilated`.
- Mean-entropy maximization (con3, con4): it lowers the accuracy on the common
  classes.
