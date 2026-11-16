# Lessons learned

Lessons that generalize, each with its reason, its number and the script that
printed it. The numbers of each run are in
[experiments_log.md](experiments_log.md); the results tables, and how each
number is measured, in [status.md](status.md); the data, model and scripts in
[descriptions.md](descriptions.md). A new lesson goes at the top of its group.

- **Data and labels**: spoonbill data, label noise and cleaning, IMU
  calibration, Ornitela GPS, the GPS speed cut.
- **Model and training**: rotation, the magnitude channels, model size.
- **Measuring**: how to score and compare models, with and without labels.
- **Engineering pitfalls**: traps in the code.
- **Early runs**: exp15-exp133, from the training logs of old scripts.

## Data and labels

### Spoonbill data: few birds per behaviour

The spoonbill data has 4718 segments of 20 records, about as many as the 4338
bursts of the gull `starts.csv`, but each behaviour comes from few birds
(`exps/spoonbill_data_stats.py`; terms in `descriptions.md`, Terms).

- **A class can be one bird, so a model can learn the bird.** for-search is 90%
  bird 760, stand-rest 94% bird 763, sit-alert 100% bird 584. 51 of the 53
  fly-soar segments come from the three migration birds (6283, 6287, 6291),
  which have no other label. Split by fix at least (Measuring, "Keep a fix
  whole").
- **Labels made by eye are a weaker test.** Those 51 soaring segments were
  labeled from plots, not from video. Of the 429 beg segments, all from juvenile
  6381, 329 were labeled from the signal alone. A model scored on them is scored
  against an annotator's reading of the signal, not against what the bird did.
- **Short behaviours are lost at 20 records.** 456 of 571 for-intake bouts are
  shorter than 20 records, so its 9459 records give 115 segments: 24% of the
  records used. for-handle keeps 35%. Over all labels, 857 of 2220 bouts are too
  short, and 81% of the records land in a segment.
- **Map labels before cutting segments.** The second paper maps beg_flap to beg.
  Mapped first, a beg_flap bout joins the beg bout next to it, so the records at
  the label change land in a segment instead of being cut off: 438 complete
  begging segments instead of 434 (`scripts/data/prepare_spoonbill_data.py`
  against `exps/spoonbill_data_stats.py`).
- **The second paper's `behaviour` column is model output.** In
  `gps.behav.data.csv` it is the paper's random-forest prediction, and the file
  has no x, y, z. It is not training data.

### IMU calibration from the data, for devices without calibration values

`behavior/data.py::get_data` used to return nothing for a device without
calibration values in the database. It now estimates them from the IMU data
(`estimate_calibration`, `acc_calibrate`), translated from Eldar Rakhimberdiev's
R code; the rules are in `docs/methods.md`, rule 1. The numbers below are from
`exps/check_imu_calibration.py`, on 15 Ornitela tags.

- **The translation matches R.** On the raw data behind
  `calibration_per_device_auto.csv`, the 15 scales agree within 4.5e-12. R was
  not installed, so this R output is the only check. It needs `q_static` 0.85,
  the value of the R example, and every row of the file, GPS rows too: each
  carries one IMU sample.
- **On a bird it is one scale, not one per axis.** The resting samples of all
  15 tags cover too few postures (coverage 0.0006-0.0119, cutoff 0.02), so
  `auto` falls back to one scale and no offset. The per-axis fit is tested only
  on synthetic data (`tests/test_data.py::test_acc_calibrate`: offsets within 1
  count, scales within 0.2%). For a per-axis fit, calibrate the tag before
  deployment (`docs/methods.md`, Calibrating a tag off the bird).
- **A short time range falls back to 1058.8, which is not exact.** With fewer
  than 300 static samples (a 60-sample burst is far below that),
  `estimate_calibration` uses no offset and the scale 1058.8. A scale fitted on
  one burst would set that burst's magnitude to 1 g, even in flight. But the 15
  tag scales run 993.7-1060.1 (mean 1029.6), so 1058.8 makes the data of the
  lowest tag 6.1% too small. Calibrate over a long range, as
  `scripts/data/get_data_gull_20.py` does.
- **`q_static` matters little.** 0.10 (the R default) against 0.85 moves the
  scale by at most 0.39%. 0.85 is kept because it reproduces
  `all_devices_calibrated.csv`.
- **Derivatives are numerical, as in R `optim`.** An exact gradient was tried.
  It stopped early under R's stopping rule, because the loss is very flat along
  the offsets.

### Ornitela GPS: match by nearest time, not by file position

`scripts/data/bird_behavior_app_data.py::prepare_calibrated_app_data` and
`scripts/data/merge_simon_data.py::load_calibrated_rows` gave each burst the GPS
row just above it in the file. Both now take the nearest valid fix of the same
device within 2 s (`docs/methods.md`, Ornitela data).

- **The row above is not always the nearest.** A fix 2 s away won over one 1 s
  away, and a burst was dropped when the row above was invalid (more than 2 s
  away, or at latitude and longitude 0), even with a good fix close by.
- **A bug that rarely fires stays hidden.** GPS rows nearly always sit right
  above their burst, so only 287 of 659,409 bursts got another fix, and 247
  dropped bursts came back (`exps/recompute_doc_numbers.py`; details in
  `experiments_log.md`, Ornitela GPS by nearest fix).
- **`merge_simon_data.py` also kept x and y unswapped.** It swaps them now, and
  its output matches `rose_data.csv` exactly on every shared burst.

### Retraining on the cleaned labels (exp198-exp204)

`starts_clean.csv` drops the 26 bursts whose GPS speed contradicts their label
(next lesson). Three training seeds a side, the split held still, and every run
read at the same epoch, 4000. The numbers are in `status.md`, Gulls, raw against
cleaned labels.

- **A label fix shows where it was aimed, and nowhere else.** The 26 bursts came
  from a GPS-speed rule, and speed conflict is the only check that moves: 2.5 to
  3.3 times lower at matched seeds, and the two ranges do not overlap
  (`exps/eval_unlabeled.py`). Accuracy could not move anyway: the 26 bursts are
  0.60% of the data, and 25 of them were in the train split, 1 in valid (3,900
  train and 438 valid bursts before, 3,875 and 437 after;
  `exps/eval_labeled.py`).
- **Cleaning one class can move another, through a shared input.** The 26
  dropped bursts are land bursts (SitStand, TerLoco, Pecking) at 3.24-17.78 m/s
  (`exps/find_label_noise.py`). Without them, no land or water burst is faster
  than 4.72 m/s, so in training, fast means flying. That moves ExFlap, a flight
  class that is usually slow (median 1.39 m/s, against 9.63 for Flap): the only
  ExFlap valid bursts that ever flip to Flap are the two at Flap-like speeds,
  15.84 and 4.99 m/s, and only in clean-label runs
  (`exps/recompute_doc_numbers.py`). Ask which input the dropped rows carried,
  not which class they had.
- **But the effect is small, and one seed made it look large.** On the ExFlap
  burst at 4.99 m/s, with its acceleration kept and its speed raised from 0 to
  20 m/s, P(ExFlap) fell from 0.45 to 0.06 for clean exp200 and from 0.79 to
  0.44 for raw exp199. Other seeds blur this: raw exp201 falls from 0.47 to
  0.19, clean exp204 from 0.84 to 0.52. Giving each of the 4338 bursts another
  burst's speed changes the class of 27.8-29.5% of bursts for the raw models and
  29.5-31.1% for the clean ones: about 2 points more, and the ranges touch
  (`exps/speed_reliance.py`).
- **Still keep exp197, trained on the raw data, as the working model.** This is
  a judgement, not a measurement, and it goes against the speed-conflict
  numbers. Revisit it when a second logger, or more ExFlap data, can settle it
  (TODO in `status.md`). Three reasons:
  - The only check where cleaning wins is speed conflict. But GPS speed also
    picked the 26 bursts, so that check cannot judge the change.
  - GPS speed can be badly wrong in real data. 16 of the 26 bursts are one
    device on one day, more likely a device fault than 16 labeling mistakes
    (`exps/find_label_noise.py`). The model should see such bursts in training.
  - Nothing else improves: accuracy, place, flicker and rotation stay inside the
    seed spread, and reliance on speed barely changes.

### Label noise: trust only the speed half of a label's definition

`exps/find_label_noise.py` flags bursts whose GPS speed contradicts their own
label's definition in Table S1 of the Judy supplement
(`/home/fatemeh/Downloads/bird/papers/Judy_features_supp1.pdf`): 26 of 4338
bursts, 0.60%. They are listed in `label_noise_candidates.csv` and dropped by
`data_processing.remove_label_noise`, giving `starts_clean.csv`, 4312 bursts.

- **Only the speed half of a definition is usable.** Rules on the accelerometer
  shape were tried and all dropped, after checking them against the plots in
  `~/Downloads/bird/results/gt2_starts/all`. An ODBA (overall dynamic body
  acceleration) rule, "Stationary means constant on all 3 axes", flagged sitting
  birds that shifted once mid-burst. A spectral rule, "Soar means no wing beat",
  flagged smooth glides: the FFT removes the mean but not the trend, so a
  glide's slow drift looks like a beat over 20 samples. Speed is a hard physical
  limit; shape is a description, and a description has a real tail.
- **Extreme for its class is not mislabeled.** Every dropped rule was set
  outside its class's 99th percentile and still flagged good bursts. That test
  finds the tail of a real distribution, not errors.
- **Never let the label under test train the judge.** exp194 reaches 99.44%
  train accuracy by learning the bad labels too: it predicts TerLoco at 0.92
  confidence for the burst labeled TerLoco at 17.78 m/s. exp196, trained with
  rotation, underfits, so it did not learn the labels by heart
  (`exps/find_label_noise.py`).
- **Compare a model's disagreement with its base rate.** exp196 disagrees with
  85% of the flagged bursts, which looks like strong support until you see it
  also disagrees with 8.3% of all bursts (exp194: 15% and 0.9%). The script
  prints both for that reason.
- **Check whether flags cluster by device and day.** 16 of the 26 are device
  805 on 2014-06-07: more likely a GPS or device fault than 16 separate labeling
  mistakes.
- **A constant signal at high speed is ambiguous.** Sitting on a fast boat and
  gliding both give a smooth signal at speed; only the GPS track or altitude
  tells them apart. The four device-806 bursts, labeled SitStand at 9.4 and 13.7
  m/s, are the clearest mislabels, with that caveat.

### Read GPS speed as truth only after the 30 m/s cut

Raw GPS speed has a tail of sensor errors: on the unlabeled gull data it reaches
521 m/s (`exps/inspect_unlabeled_data.py`, its `stats.txt` in
`/home/fatemeh/Downloads/bird/data/ssl/hist_ssl20`). So speeds of 30 m/s and
above are dropped first (`docs/methods.md`, rule 7). Any rule that reads GPS
speed as truth, such as `exps/find_label_noise.py`, assumes that cut.

## Model and training

### Spoonbill: rotation (exp205-exp211)

The exp197 recipe on the spoonbill data (`scripts/batch_train_supervised.py`,
numbers in `experiments_log.md`).

- **Rotation costs spoonbills far more than gulls.** With rotation the model
  underfits: valid 83.96-84.55 at epoch 4000 over 3 seeds, and sit-alert 0 of
  61 right even in training (exp206). Without it: 93.27-94.65, sit-alert 61 of
  61 (exp207). On gulls rotation cost 4 points. The spoonbill resting and
  alert classes differ by posture, sitting or standing, and posture is the
  gravity direction rotation scrambles.
- **The price is the same as for gulls.** Without rotation a model can also
  learn the tag's mounting. Train and valid hold the same birds, so 94% says
  nothing about a new bird or a tag mounted another way.

### Rotation-invariant input channels (exp197)

`data.add_magnitude_features` appends three channels to the 4 inputs: `mag`
(`||a||`), `dyn_mag` (`||a - mean_t a||`) and `jerk_mag` (`||a_t - a_{t-1}||`).
All three are lengths of an acceleration vector, so rotating the burst leaves
them unchanged. exp197 = exp196 + these three, +300 params (5,129 -> 5,429):

|        | valid acc | F1 balanced | SO(3) mean | rotation flip |
| ------ | --------- | ----------- | ---------- | ------------- |
| exp196 | 92.69     | 0.83        | 92.03      | 5.37%         |
| exp197 | **94.52** | **0.90**    | **93.66**  | **1.72%**     |

- **Giving a model an invariant beats making it learn one.** exp196 spends
  capacity learning rotation-invariance from the augmentation and underfits
  (train 91.56 < valid 92.69). exp197 flips regime, train 97.90 / valid 94.52,
  and recovers half the accuracy rotation cost — but only half: exp194 is still
  96.35, and it collapses to 13.05 under rotation.
- **A conv stack cannot build these itself.** It only takes linear combinations
  across channels, and `sqrt(x^2+y^2+z^2)` is not one.
- **Lengths cannot restore directions, and TerLoco proves it.** Rotation costs
  the *direction* of gravity, i.e. posture. Valid F1 exp196 -> exp197: Manouvre
  +0.12, Pecking +0.08, Boat +0.06, but **TerLoco does not move** (0.89 both,
  against .97 for exp194). Walking is separated by which way small movements
  point relative to gravity, so no invariant channel can reach it.
- **Only an orientation estimate can bring walking back.** Pecking and Manouvre
  are bursts of movement, so a size channel carries much of what marks them, and
  both win part of the rotation cost back: valid F1 .68 -> .76 and .67 -> .79
  (exp194 has .94 and .84, `exps/eval_labeled.py`). Walking is a posture plus a
  small rhythm — the body upright, the steps shaking it sideways. Both of those
  are directions relative to gravity, which every invariant channel throws away,
  so TerLoco stays at .89 however many such channels we add. The way in is to
  put the frame back: estimate the gravity direction per burst (the low-pass
  part of the acceleration), rotate the burst so it points down, and train on
  the aligned axes. Untested.
- **`mag` carries the gain, `jerk_mag` almost none.** Measured before training,
  by adding one channel at a time to a random forest on burst summaries and
  scoring it on randomly rotated data: `mag` +1.7 points, `dyn_mag` +0.6,
  `jerk_mag` +0.04, all three +2.3. `dyn_mag` and `jerk_mag` correlate at 0.96,
  so they are nearly one feature. In the *un*rotated frame all three together
  give +0.2 — the raw axes already separate these behaviours, so the channels
  buy invariance, not discrimination.
- **Compute them once at load time, before augmentation.** Being invariant they
  stay correct whatever rotation follows. Jitter, scaling and warp would break
  that and would have to be recomputed per batch.
- **`jerk_mag` is undefined at t=0; edge-replicate, do not zero-pad.** A zero is
  a "no motion" sample in every burst, and the model pools mean/max/std over
  time, so it biases the channel down for the high-jerk classes.
- Untested: these channels **without** rotation augmentation (an exp194
  counterpart). The forest says that arm gains ~0.2, but that is a proxy.

### Rotation augmentation (exp195)

Full SO(3) rotation of the IMU acc channels, train set only (`RandomRotation3D`).
The labeled data is one logger type mounted much the same way on every bird, so
the standard split holds no orientation variation at all. Full 3D is the realistic
model for a new logger, whose tilt and position on the bird are both unknown.

- **A standard-split A/B only measures the cost.** exp195 92.24 val vs exp194
  96.35 (-4.1) reads as "augmentation hurt". Re-evaluating both checkpoints on
  orientation-perturbed validation data inverts it
  (`exps/eval_labeled.py`):

  | valid under              | exp194 (no aug) | exp195 (rotation) |
  | ------------------------ | --------------- | ----------------- |
  | clean                    | **96.35**       | 92.24             |
  | x/y swap                 | 71.46           | **90.41**         |
  | pitch 20° / 70°          | 82.88 / 2.05    | **90.87 / 90.64** |
  | random SO(3), mean of 20 | 13.05           | **90.67**         |

  Off-orientation exp194 is not uncertain, it is confidently wrong: 2.05% is well
  under the 11% chance rate, so no confidence threshold will catch it.

- **The cost lands on the ground behaviours**, which need the gravity direction
  rotation scrambles: valid F1 TerLoco .97 -> .79, Pecking .94 -> .68. Balanced
  valid F1 0.92 -> 0.84, twice the accuracy drop
  (`exps/eval_labeled.py`). Plain and balanced numbers can disagree in
  sign on a rare class, so say which one a number came from.

- **Rotation replaces overfitting with underfitting**: exp194 99.44/96.35 vs
  exp195 90.21 (unaugmented train) / 92.24. At 5,129 params the capacity goes into
  invariance, not memorization.

- **A bigger receptive field did not pay for it**: BirdModel -> SmallDilated is
  +1.6 val without rotation (exp192 94.75 -> exp194 96.35) but +0.7 with
  (discarded trial 91.55 -> exp195 92.24).

### Rotation augmentation (background)

- The accelerometer reading mixes logger orientation, logger position on the bird
  (neck, back, wing), bird movement, external forces (wind, water, collision) and
  gravity. Only bird movement is the label. Known per-manufacturer conventions are
  a coordinate relabel, not an augmentation problem — Ornitela swaps x and y,
  undone in `scripts/data/bird_behavior_app_data.py`.
- **Static classes survive rotation, their members do not.** In the discarded
  BirdModel + rotation trial (val 91.55%) the orientation-dependent static classes
  held (val F1 SitStand .97, Float .97, Boat .92): a static behavior has |acc|≈1 g
  with near-zero temporal variance whatever the orientation, so rotation-invariant
  statistics still separate it. The loss went to the rare dynamic classes
  (Pecking .57, Manoeuvre .77, ExFlap .57 — prose record only, that run was not
  kept). exp195's bigger receptive field fixed exactly those three and the cost
  moved to TerLoco/Pecking instead.
- **Only augment the training set.** `prepare_train_valid_dataset` passes one
  transform object to both the train and eval `BirdDataset`s, which would rotate
  the evaluation data too — still a hazard for any script using it.
  `batch_train_supervised.py` no longer can: the transform is held by the
  `GpuBatches` train loader, and the eval `BirdDataset` has none.

### Model size

- **Bigger is worse on this small dataset.** Wider (30 -> 96 channels, 6.3k ->
  58k params) drops 94.5 -> 93.7 with clear overfitting (exp67). Deeper (3 -> 4
  layers) drops 94.5 -> 94.0 (exp70); a residual connection recovers most of it
  (exp71, 94.8).
- **Receptive field buys more than capacity does.** `BirdModelSmallDilated` is
  *smaller* than the 3-conv `BirdModel` (5,129 vs 6,309 params) yet gives the
  best `starts.csv` result so far: exp194 96.35 val, against 94.75 for BirdModel
  on the same data (exp192). The difference is the dilated k=5 stack — RF 25
  spans the whole 20-sample burst, where BirdModel's RF 7 sees only a third of
  it and cannot resolve the global flap sine-wave. Spend the budget on context,
  not channels. Widening the RF with plain k=7 convs instead of dilation
  (`BirdModelWideRF`, RF 19) costs 1.1 pts (exp193).
- **Multi-head attention helped** the transformer variants (exp85–87).
- The 3-conv small model (~6.3k params) is the historical reference and trains in
  ~20 min on a laptop RTX 3070; `BirdModelSmallDilated` has replaced it as the
  app inference model and the supervised baseline.
- **Compare parameter counts at the same class count.** `BirdModel(4,30,·)` is
  6,309 at 9 classes and 6,340 at 10, and the old notes quote the 10-class
  numbers throughout (their width pair 6,340 -> 58,282 is 10-class on both
  sides). Reading one against the other invents differences that are not there.
- The compact models overfit far harder than BirdModel (train ~99.5 vs ~96), so
  only the validation number is comparable between them.

## Measuring

How each number is defined, with its unit, is in `status.md`, How the numbers
are measured.

### Score the way the paper scores, or the comparison means nothing

The same spoonbill model (no rotation, mean of 3 seeds) gets prey ingestion F1
0.92 on our valid segments and 0.59 on records, the papers' way
(`exps/eval_spoonbill.py`).

- Our segments are whole 1 s bouts of one behaviour. The papers cut each fix
  into 1 s windows whatever the labels, so a window can hold two behaviours, for
  example 11 records of searching then 9 of ingesting. The model gives one
  answer for the 20 records, and each record is scored against its own label:
  answering ingest there gets 9 of 20 right.
- Prey ingestion suffers most because its bouts are short: see "Short
  behaviours are lost at 20 records" above.
- Their test is easier in one way: they split per segment, so train and test
  share fixes; ours share none. But theirs is the mean of 100 splits, ours one
  split of 79 fixes, and both test the training birds.

### Keep a fix whole when it holds several segments

A spoonbill fix holds up to 10 consecutive 1 s segments. Split per segment, 502
of 514 valid segments had a train segment from the same fix (exp205,
`exps/eval_spoonbill.py`). `split_by` fix keeps each fix whole. A split by bird
is not possible: sit-alert, stand-other, drink and beg each come from one bird
(`exps/spoonbill_data_stats.py`).

### The highest accuracy can hide a total failure

exp194 has the highest valid accuracy, 96.35%, and falls to 13.05% when the tag
is turned at random. exp197 is 1.83 points lower and holds 93.66%
(`exps/eval_labeled.py`). The labeled data is one logger type, mounted much the
same way on every bird, so the valid split cannot see a change of mounting.
Test the orientation too.

### The train-valid gap tells overfitting from underfitting

exp194 reaches 99.44% train and 96.35% valid accuracy: it memorises. exp196,
the same model trained with rotation, reaches 91.56% train and 92.69% valid: it
underfits. The augmentation decides which, not the parameter count
(`exps/eval_labeled.py`).

### Give plain and class-balanced F1 together, with the bursts per class

- exp196 has a pooled plain F1 of 0.93 and a pooled class-balanced F1 of 0.83:
  the errors are in the rare classes. Per class, exp194 to exp196: TerLoco 0.97
  to 0.89, Manouvre 0.84 to 0.67, Pecking 0.94 to 0.68; every other class moves
  0.03 or less (`exps/eval_labeled.py`).
- Small classes swing. One of the 16 Manouvre valid bursts moves Manouvre
  recall by 0.06. One of the 4 ExFlap bursts moves ExFlap recall by 0.25, and so
  the pooled class-balanced F1, the mean recall over 9 classes, by 0.03.
  `exps/eval_labeled.py` prints the bursts per class for this reason.
- Class-balanced F1 needs every class in valid. With no drink segment in the
  spoonbill valid split, `behavior/utils.py::per_class_statistics_balanced`
  divides by 0 and the whole table is NaN.

### One seed is not a result; compare the seed spread

At one seed, the clean-label model (exp200) was 0.23 points below the raw one
(exp199). Over three seeds a side the ranges overlap: 93.36-94.05% against
93.82-94.05% (`exps/eval_labeled.py`). Only a difference whose ranges do not
overlap counts. Of all the checks, only speed conflict passed (`status.md`,
Gulls, raw against cleaned labels).

### Compare runs at a fixed epoch, not at `<exp>_best.pth`

`<exp>_best.pth` is the best of 4000 epochs on about 438 valid bursts, so it
rewards a lucky epoch, and it lands somewhere else in each run: epoch 1870 for
exp197, 1227 for exp198, 1888 for exp199 and 3947 for exp200 (stored in each
checkpoint, printed by `scripts/batch_train_supervised.py`). Two runs read at
their best epochs differ in checkpoint age as well as in the change being
tested. `save_final` keeps epoch 4000; compare there.

### Pin the split when the data changes

`behavior/utils.py::stratified_split` draws every class from one random
generator, in class order. When one class changes size, every class after it is
reshuffled. Dropping the 26 noise bursts (SitStand, TerLoco, Pecking) left Flap
to Float as they were and reshuffled the rest: 177 of exp197's 438 valid bursts
fell in exp198's train split (`exps/recompute_doc_numbers.py`). `split_file`
borrows another data file's valid split, and `split_seed` holds the split while
`seed` moves the training. Without them, two runs on two data files are not
scored on the same bursts.

### Softmax confidence cannot compare models

On device 6004, exp194 and exp196 differ by 0.02 in mean confidence (0.89 and
0.91), while exp194 has 16 times more speed conflicts: 1.45% against 0.09% of
bursts (`exps/eval_unlabeled.py`). Confidence can at most rank which bursts to
look at. It cannot say whether a model works on a new bird.

### Count predictions that cannot be true

- Without labels, a prediction that contradicts something known without the
  model is wrong on physical grounds: SitStand at 13 m/s, TerLoco a kilometre
  out to sea. The four checks are defined in `status.md`.
- Rotation flip separates the models most: 82.76% for exp194 against 5.37% for
  exp196 (`exps/eval_unlabeled.py`).
- Not every check separates them, so report them all. exp196 is worse on place
  conflict, 1.45% against 1.18%, because it predicts more TerLoco, a land class:
  4.2% of bursts against 2.4%.
- Place conflict can only score the 53% of bursts clearly inland or offshore.
  The land mask has a cell of 1/100 degree (about 1.1 km), so bursts within
  about 2 km of a coast are skipped; a gull on a beach is ambiguous at that
  scale.

### A raw change rate is not an error rate

Counting every class change between consecutive bursts of a fix, exp194 and
exp196 look the same: 17.65% and 16.14% of pairs. Counting only changes never
seen between labeled behaviours: 6.66% and 4.92% (`exps/eval_unlabeled.py`).
In the labeled data only 10 of 36 behaviour pairs ever follow one another in a
fix, led by Soar-Manouvre (37 transitions), SitStand-Pecking (31) and
TerLoco-Pecking (23) (`exps/label_cooccurrence.py`, its
`label_transitions.csv`).

### Check a new evaluation against stored predictions

The app CSVs hold the predictions of the model that made them, so recomputing
them is a free check of new evaluation code. It was done once for
`exps/eval_unlabeled.py`; that code was not kept.

## Engineering pitfalls

### `load_csv_pandas` timestamps are 1000x too small under pandas 3

`bd.load_csv_pandas` builds its timestamp with `to_datetime(...).astype("int64") // 1_000_000_000`. Pandas 3 parses to `datetime64[us]`, not `[ns]`, so the
division yields seconds/1000 and every burst dates to 1970. The label, device
and index columns of `ldts` are fine; only column 2 is wrong. Training does not
read it, but anything naming a burst does, so `exps/plot_errors.py` takes the
datetime as text from `find_label_noise.load_bursts` instead. Not fixed yet:
the loader is shared with the training scripts.

### Rebuild the split on the same device as training

- **Rebuild the split on the same device as training.** `stratified_split` seeds
  a `torch.Generator(device=...)`, so passing a CPU label tensor gives a
  different split and silently scores the model on bursts it trained on — which
  showed up here as a suspiciously perfect 99.5% "valid" accuracy for exp194.

### Batched augmentation on the GPU

Per-sample transforms run inside `BirdDataset.__getitem__`, one Python call per
sample per epoch. On the 3900-sample full batch `RandomRotation3D` alone cost
~370 ms/epoch — that is why exp195 took 47 min against exp194's 12. The `Batch*`
transforms in `behavior/data_augmentation.py` build all N at once, and
`bd.GpuBatches` keeps the split on the GPU and applies them once per batch:
**the same exp195 config now runs in ~5.5 min**, below the un-augmented baseline.

- **Verify equivalence, not just speed.** exp196 is exp195 rerun through the new
  path: val 92.69 vs 92.24, valid F1 within ±0.01 on 7 of 9 classes, same
  orientation robustness — in 5m03s against 47m31s. The RNG stream differs, so
  reruns are not bit-identical; expect seed-level noise, not equality.
- `train_one_epoch` only iterates its loader, so any iterable works and
  `behavior/model.py` needed no change at all.
- Augmentation now lives on the train **loader**, not the dataset, which removes
  the shared-transform footgun described under "Rotation augmentation
  (background)" above.
- Batched transforms take channel-first `(N, C, T)`; the per-sample ones take
  `(T, C)`. They are not interchangeable, and the other scripts
  (`train.py`, `train_sup_contrastive.py`, `train_self_distill.py`, `simGCD.py`)
  still use the per-sample ones.

### Reproducibility / infrastructure bugs

- **BatchNorm caused a large train/val divergence** early on; fixed with
  `track_running_stats=False` (exp15 -> exp22/exp23). Per-conv BatchNorm layouts
  also shifted results (exp30/exp34).
- **Low batch size made runs non-reproducible** (exp29/exp31 could not reproduce
  earlier numbers until reverting to commit ed7d9a389). Pin the commit hash with
  each reported run.
- **`drop_last=True` breaks length-based accuracy**: accumulate the sample count
  per batch instead of using `len(loader.dataset)` (fixed in the train/eval
  loops of the contrastive scripts).
- A rounding subtlety in the old ÷20 mapping: Python rounds half to even (2.5 ->
  2.0), which shifted a few indices. Minor, but it explains small dataset diffs.
- **Recorded parameter counts go stale silently.** `MaskedAutoencoderViT` was
  logged at 9,546,756; the current code gives 9,557,508 at img_size=20, the gap
  being two 21x256 position embeddings that became learnable. `ResNet18_1D` was
  logged at 60,964 and now gives 61,289. Nothing errors — the number just stops
  being true. Recompute a count before quoting it, and say which constructor
  arguments produced it.

## Early runs (exp15-exp133)

From the training logs of old scripts; no script in the repo reruns them.
The run details are in `experiments_log.md`.

### Training recipe

- **AdamW with defaults is already the best.** The small 3-conv model trains
  best with AdamW, lr=3e-4, weight decay 1e-2, StepLR (step 2000, γ=0.1). SGD
  is ~5 points worse (exp42 90% vs exp45 95.4%).
- **Weight decay barely matters** at this scale: wd from 1e-3 to 3e-1 all land
  within noise of exp45 (exp53, exp54).
- **Dropout stabilizes training and prevents overfitting** (exp46 overfits
  after ~1000 epochs without it). But dropout lowers model capacity, so it can
  make train accuracy *lower* than validation (dropout is off in `model.eval()`);
  see the MAE note below.
- **Cosine annealing did not beat StepLR** on accuracy. It reduces overfitting
  (exp55 almost none) but going 3e-4 -> 2.5e-4 with cosine did not prevent the
  overfitting that motivated it.
- **Batch size**: smaller batches converge faster down to ~bs=25; bs=6 is worse
  and very slow. There is no accuracy gain from large batches here.

### Data

- **No augmentation gave the best supervised results.** exp125 (no augmentation,
  no sampling) is 96.94/95.36. Jitter/scaling (exp126), time/magnitude warp
  (exp129), and warping GPS all made things worse; holding GPS fixed during
  augmentation (exp130) beats warping it but still trails. **The best accuracy on
  `starts.csv` is exp194 (BirdModelSmallDilated, no augmentation, 96.35 val)**,
  and rotation (exp195, 92.24) is no exception — but it is kept anyway, for
  robustness rather than accuracy; see the rotation section.
- **The "sampling hurts" claim does not survive its own numbers.** exp127
  (sampling only, no augmentation) is 94.29/95.84: validation is *above* exp125's
  95.36 and only train is lower. The old notes call sampling worse; what the
  numbers show is that it costs train accuracy and helps rare classes. Do not
  cite exp127 against sampling without rerunning it.
- **GPS speed is needed for some classes**, mainly Boat and SitStand
  (IMU-only exp131 vs exp125; consistent with Judy's paper). Normalizing GPS by
  its max was a large early win (25% -> 53%, exp19).
- **Less data, larger drop**: 100% -> 20% of the data costs ~6 points
  (95.5 -> 89.4). More data past exp45 (exp78, +25%) did not help, so the
  dataset is near saturation for this model.
- **The data is noisy**: different data slices and seeds shift validation by
  ~2.5 points for reasons never fully explained (exp98, exp109–112). Report a
  fixed seed (32984) and be wary of reading small val differences as real.
- **Peck labels are noisy**; treat Pecking results with caution.

### Contrastive and clustering

- **Supervised contrastive loss** slightly improves the t-SNE structure and
  tracks the plain classifier (con1/con2 ≈ exp125).
- **Mean-entropy maximization hurts.** As a soft class-balancer it sacrifices
  majority-class accuracy and worsens the t-SNE (con3/con4). Class balancing in
  general (weights or resampling) trades overall accuracy for rare-class recall
  (exp28, exp34, exp127).

### MAE pretrain / finetune

- **Average-pool without a class token is the best MAE head** (exp96, 91.08 /
  train 94.55), beating class-token slicing and the ImageBind head. LayerNorm
  before the pool/classifier helps slightly (exp96 vs exp97).
- **Validation above training with the MAE encoder is a dropout artifact**, not a
  bug. Heavy dropout (.7) lowers train accuracy (capacity down) while eval turns
  dropout off; the gap shrinks with smaller dropout and higher accuracy (exp92),
  and disappears without dropout (exp87, exp96). Verified by varying validation
  size 10/30/50% (exp91/93/94): behavior unchanged.
- Finetuning the whole network beats training only the class head by a wide
  margin (f1 48.6 vs f2/f3 ~83).
