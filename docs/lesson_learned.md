# Lessons learned

Lessons that generalize, each with its reason, its number and the script that
printed it. The numbers of each run are in
[experiments_log.md](experiments_log.md); the results tables, and how each
number is measured, in [status.md](status.md); the data, model and scripts in
[descriptions.md](descriptions.md). A new lesson goes at the top of its group.

- **Data and labels**: spoonbill data, label noise and cleaning, IMU
  calibration, Ornitela GPS, the GPS speed cut.
- **Model and training**: rotation, the magnitude channels, the receptive
  field.
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
  which have no other label. Split by fix at least (Measuring: split per
  segment, most valid segments share a fix with train).
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

### Ornitela: the nearest GPS fix changes 287 of 659,409 bursts

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

### 26 of 4338 bursts have a GPS speed their label rules out

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

### Raw GPS speed reaches 521 m/s, so speeds of 30 m/s and up are dropped

Raw GPS speed has a tail of sensor errors: on the unlabeled gull data it reaches
521 m/s (`exps/inspect_unlabeled_data.py`, its `stats.txt` in
`/home/fatemeh/Downloads/bird/data/ssl/hist_ssl20`). So speeds of 30 m/s and
above are dropped first (`docs/methods.md`, rule 7). Any rule that reads GPS
speed as truth, such as `exps/find_label_noise.py`, assumes that cut.

## Model and training

### Fine heads per species keep accuracy, one class set does not (exp212-221)

One network on gulls and spoonbills (numbers in `status.md`, Both species).

- **Separate fine heads match the single-species models.** With a coarse head
  and one fine head per species, both species score within the seed ranges of
  their own models (`exps/eval_species_heads.py`). With one class set for both,
  sets or 23 classes, the gull lost 2.5 points or more (`exps/eval_joint.py`):
  one output made the species share or compete for its classes.
- **Accuracy is the same with the species token unknown.** It moves less than
  the seed spread: the signal already tells a gull from a spoonbill. So the
  model can also run on a species it was not trained on.
- **A translation can hold for one label and not another.** Through the
  spoonbill head, gull walking reads as spoonbill walk in all three runs
  without rotation, but the gull SitStand split into rest and alert flips in
  one of the three (exp221). Read a translation over several seeds, and check
  it with labels before using it.

### One class per label: rotation mixes the species more (exp214-215)

One model on both species with 23 classes, one per gull and spoonbill label
(`scripts/train_joint.py`; numbers in `experiments_log.md`, exp214-215).

- **The right behaviour of the other species counts as wrong.** exp215 gets
  87.90% of gull and 87.13% of spoonbill valid segments, the set model exp213
  91.32% and 92.08% (`exps/eval_joint.py`).
- **Without rotation the model keeps the species apart.** 30 of 438 gull and
  32 of 505 spoonbill valid segments get a class of the other species; with
  rotation 64 and 80 (`exps/eval_joint.py`). Why is a guess: the tag sits
  differently on the two species, and rotation hides that.
- **Read the cells between the species, not a forced choice.** Taking each
  segment's best class among the other species' classes only is arbitrary:
  training pushed all of them down. exp215 put most for-search segments in
  Soar (tried, no script kept).

### Set labels leave the split open: gull SitStand differs by run (exp212-213)

One model on both species, each label a set of joint classes
(`scripts/train_joint.py`; numbers in `experiments_log.md`, exp212-213).

- **Inside a set, the choice is not trained.** Any class of the set gives the
  same loss, so the split comes from what the model learned of the other
  species. exp213 calls 443 of 1502 gull SitStand bursts stand-alert and 62
  sit-alert; exp212, which differs only in rotation, 606 and 0
  (`exps/eval_joint.py`).
- **Accuracy cannot see that split.** Gull SitStand F1 is 0.96 for exp213
  and 0.95 for exp212, whatever the split. Read a split inside a set over
  several runs, and check it on a few labeled bursts.
- **Keep species-only classes out of a set.** A gull-only SitStand class next
  to stand-rest and stand-alert would take every gull resting burst at no
  cost, and nothing would carry over. A design choice, not tested.
- **One model for two species costs accuracy.** At one seed, exp213 is 2.5
  points below the gull models (93.84-94.06%) and 1.2 below the spoonbill
  models (93.27-94.65%) on their own valid splits (`exps/eval_joint.py`).

### Spoonbill models trained with rotation do better on gulls (exp206-211)

Each spoonbill model on all 4338 gull bursts, scored by an ecologist's label
map (`exps/eval_cross_species.py`; tables in `status.md`, Across species).

- **The worse model on its own species carries over better.** Of 3433 scored
  gull bursts, 75.5-84.7% get the expected or alternative spoonbill label with
  rotation, 65.8-67.4% without, over three seeds. On the spoonbill valid
  segments the order is reversed: 83.96-84.55% against 93.27-94.65%
  (`exps/eval_spoonbill.py`).
- **Without rotation a model reads posture, and posture differs between
  species.** The mean x of a walking gull is -0.41 g, of a standing spoonbill
  -0.38 g. exp207 calls 222 of 337 gull TerLoco bursts stand-rest or
  stand-preen; exp206, with rotation, 4 (`exps/eval_cross_species.py`).
- **Out of its training data, the seed decides.** Gull Flap gets fly-flap in
  46.2-84.1% of bursts without rotation and 48.8-96.7% with, over three seeds
  (`exps/eval_cross_species.py`). Judge a transfer over several seeds.

### Spoonbill: training with rotation lowers valid accuracy by about 10 points (exp205-exp211)

The exp197 recipe on the spoonbill data: 505 valid segments, 3 seeds a side
(`status.md`, Spoonbills).

- **Rotation costs spoonbills far more than gulls.** With rotation the model
  underfits: 83.96-84.55% valid accuracy at epoch 4000
  (`exps/eval_spoonbill.py`), and sit-alert is 0 of 61 right even in training
  (exp206, `per_class_metrics_train.csv` of
  `scripts/batch_train_supervised.py`). Without rotation: 93.27-94.65%, and
  sit-alert 61 of 61 (exp207). On gulls rotation cost 4.11 points (exp194
  against exp195). The spoonbill resting and alert classes differ by posture,
  sitting or standing, and posture is the gravity direction that rotation
  scrambles.
- **The price is the same as for gulls.** Without rotation a model can also
  learn the tag's mounting. Train and valid hold the same birds, so 94% says
  nothing about a new bird or a tag mounted another way.

### Three magnitude channels win back half the accuracy that rotation costs (exp197)

`behavior/data.py::add_magnitude_features` appends three channels to the 4
inputs: `mag` (`||a||`), `dyn_mag` (`||a - mean_t a||`) and `jerk_mag`
(`||a_t - a_{t-1}||`). Each is the length of an acceleration vector, so a
rotation leaves it unchanged. exp197 is exp196 plus these three: 5,429
parameters instead of 5,129. The results are in `status.md`, Gulls.

- **Giving a model an invariant beats making it learn one.** exp196 spends its
  capacity on ignoring the rotation and underfits: 91.56% train, 92.69% valid
  accuracy. exp197 reaches 97.90% train and 94.52% valid, and so wins back half
  of what rotation cost: exp194, without rotation, has 96.35%
  (`exps/eval_labeled.py`).
- **The network cannot compute them exactly.** Each convolution mixes the
  channels linearly before its nonlinearity, and a vector length is not a linear
  mix.
- **Lengths cannot restore directions, and TerLoco shows it.** From exp196 to
  exp197, per-class valid F1 rises for Manouvre (0.67 to 0.79), Pecking (0.68 to
  0.76) and Boat (0.88 to 0.94), but TerLoco stays at 0.89, against 0.97 without
  rotation (exp194, `exps/eval_labeled.py`). Pecking and Manouvre are bursts of
  movement, so a length carries much of what marks them. Walking is a posture
  plus a small rhythm: the body upright, the steps shaking it sideways. Both are
  directions relative to gravity, which every length throws away. Putting the
  frame back is a TODO in `status.md`.
- **`mag` carries most of the gain.** In a random forest on burst summaries,
  scored on randomly rotated data, adding one channel at a time raises valid
  accuracy by 1.69 points for `mag`, 0.59 for `dyn_mag` and 0.04 for
  `jerk_mag`, and all three by 2.33 (mean of 5 draws). `dyn_mag` and `jerk_mag`
  correlate at 0.96, so they are nearly one feature. On unrotated data all three
  add 0.18 points: the raw axes already separate the behaviours, so the channels
  buy invariance, not new information (`exps/magnitude_features_prior.py`).
- **Compute them once at load time, before augmentation.** Being invariant, they
  stay correct whatever rotation follows. Jitter, scaling and warp would break
  that, and the channels would have to be recomputed per batch.
- **Edge-replicate `jerk_mag` at t=0; do not zero-pad.** A zero is a "no motion"
  sample in every burst, and the model pools the mean, max and standard
  deviation over time, so a zero pulls the channel down for the classes that
  move most.

### Rotation augmentation keeps accuracy at any tag orientation, and costs 4 points on the standard split (exp195)

Each training burst gets a random 3D rotation of its acceleration, new every
epoch; the valid split is not turned. The accelerometer reading mixes the
logger's tilt, its place on the bird (neck, back, wing), the bird's movement,
external forces (wind, water, a collision) and gravity; only the movement is the
label. The labeled data is one logger type mounted much the same way on every
bird, so the valid split holds no change of mounting at all. A full 3D rotation
stands in for a new logger whose tilt and place are unknown. A known convention,
such as the Ornitela x/y swap, is a relabeling of the axes
(`scripts/data/bird_behavior_app_data.py`), not something to augment.

- **A test on the standard split only shows the cost.** exp195 has 92.24% valid
  accuracy against 96.35% for exp194, which reads as "augmentation hurt". Turned
  at random, exp194 falls to 13.05% and exp195 holds 90.67% (`status.md`, Gulls,
  labeled). exp194 is not unsure there, it is wrong: at a 70 degree tilt it
  scores 2.05%, far below the 11% of guessing among 9 classes
  (`exps/eval_labeled.py`).
- **The cost lands on the ground behaviours,** which need the gravity direction
  that rotation scrambles. Per-class valid F1 from exp194 to exp195: TerLoco
  0.97 to 0.79, Pecking 0.94 to 0.68. The pooled class-balanced F1 falls from
  0.92 to 0.84, twice the accuracy drop (`exps/eval_labeled.py`).
- **Static classes survive it.** With rotation, SitStand, Float and Boat keep a
  valid F1 of 0.98, 0.94 and 0.88, against 0.99, 0.97 and 0.91 without
  (`exps/eval_labeled.py`). A still bird reads about 1 g with almost no change
  over time, whatever the orientation, so lengths and spreads still separate it.

### A receptive field over the whole burst raises accuracy, most for Pecking (exp192-exp194)

- `BirdModelSmallDilated` is smaller than the 3-layer `BirdModel` (5,129 against
  6,309 parameters, 9 classes) and scores higher on `starts.csv`, both without
  augmentation: 96.35% valid accuracy (exp194) against 94.75% (exp192). The
  difference is the receptive field. Three kernel-5 convolutions with dilation
  1, 2 and 3 see 25 samples, the whole 20-sample burst; the three kernel-3
  convolutions of `BirdModel` see 7.
- Per-class valid F1, exp192 to exp194: Pecking 0.79 to 0.94, Boat 0.82 to 0.91,
  Manouvre 0.81 to 0.84. Flap was already high, 0.98 to 1.00, and Soar falls,
  0.94 to 0.91. The switch was made after a `BirdModel` run (exp125) read a
  clear wing beat as Manouvre in the app.
- Widening the receptive field with plain kernel-7 convolutions instead
  (`BirdModelWideRF`: receptive field 19, 6,889 parameters) gives 95.21%, 1.14
  points below (exp193).
- Both compact models fit their train split almost fully (99.44% and 99.67%
  train accuracy) where `BirdModel` reaches 95.95%, so compare them on valid
  only. All numbers here are in each run's `app_loss_acc.txt` and
  `per_class_metrics_valid.csv`, written by `scripts/batch_train_supervised.py`.

## Measuring

How each number is defined, with its unit, is in `status.md`, How the numbers
are measured.

### Spoonbill prey-ingestion F1 is 0.92 on segments and 0.59 on records

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

### Split per segment, 502 of 514 spoonbill valid segments share a fix with train

A spoonbill fix holds up to 10 consecutive 1 s segments. Split per segment, 502
of 514 valid segments had a train segment from the same fix (exp205,
`exps/eval_spoonbill.py`). `split_by` fix keeps each fix whole. A split by bird
is not possible: sit-alert, stand-other, drink and beg each come from one bird
(`exps/spoonbill_data_stats.py`).

### exp194 has the highest accuracy and falls to 13% at a random orientation

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

### Class-balanced F1 swings by 0.03 with one ExFlap burst

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

### A 0.23-point gap at one seed disappears over three seeds

At one seed, the clean-label model (exp200) was 0.23 points below the raw one
(exp199). Over three seeds a side the ranges overlap: 93.36-94.05% against
93.82-94.05% (`exps/eval_labeled.py`). Only a difference whose ranges do not
overlap counts. Of all the checks, only speed conflict passed (`status.md`,
Gulls, raw against cleaned labels).

### The best-valid checkpoint lands at a different epoch in each run

`<exp>_best.pth` is the best of 4000 epochs on about 438 valid bursts, so it
rewards a lucky epoch, and it lands somewhere else in each run: epoch 1870 for
exp197, 1227 for exp198, 1888 for exp199 and 3947 for exp200 (stored in each
checkpoint, printed by `scripts/batch_train_supervised.py`). Two runs read at
their best epochs differ in checkpoint age as well as in the change being
tested. `save_final` keeps epoch 4000; compare there.

### Dropping 26 bursts moved 177 of 438 valid bursts into train

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
bursts (`exps/eval_unlabeled.py`). Nor does confidence follow accuracy: turned
at random, exp194 falls from 96.35% to 13.05% valid accuracy, but its mean
confidence only from 0.964 to 0.735 (`exps/recompute_doc_numbers.py`).
Confidence can at most rank which bursts to look at. It cannot say whether a
model works on a new bird.

### Predictions that contradict speed or place give an error rate without labels

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

### Only class changes never seen in the labels separate the models

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

- Under pandas 3, the timestamp of `behavior/data.py::load_csv_pandas` (column 2
  of `ldts`) is 1000 times too small, so every burst dates to 1970:
  `to_datetime` now gives microseconds, and the code divides by 10^9 as if they
  were nanoseconds (lines 1430-1433). Training does not read it.
  `exps/plot_errors.py` takes the datetime as text from
  `find_label_noise.load_bursts` instead. Not fixed (TODO in `status.md`).
- `behavior/utils.py::stratified_split` seeds its generator on the device of the
  label tensor. A split rebuilt on the CPU differs from the GPU split of
  training and puts training bursts in valid. Rebuild it on the same device, as
  `exps/eval_labeled.py::build_splits` does.
- `behavior/data.py::prepare_train_valid_dataset` gives one transform object to
  both the train and the eval dataset, so an augmentation would change the eval
  data too (TODO in `status.md`). `scripts/batch_train_supervised.py` avoids
  it: the transform sits on the `GpuBatches` train loader, and the eval dataset
  has none.
- Rotating each burst in Python, one at a time in `BirdDataset.__getitem__`,
  made exp195 take 47 min. The batched transforms of
  `behavior/data_augmentation.py`, applied per batch by `bd.GpuBatches`, ran the
  same config in 5 min (exp196; start and end times in `tensorboard/195.txt` and
  `196.txt`). The results agree within seed noise: 92.69% against 92.24% valid
  accuracy, per-class F1 within 0.01 on 7 of 9 classes, and 92.03% against
  90.67% at a random orientation (`exps/eval_labeled.py`). The two paths draw
  different random numbers, so the runs are not identical.
- The batched transforms take `(N, C, T)`, the per-sample ones `(T, C)`; they
  are not interchangeable. `scripts/train.py`,
  `scripts/train_sup_contrastive.py`, `scripts/train_self_distill.py` and
  `exps/simGCD.py` still use the per-sample ones.
- With `drop_last=True`, dividing by `len(loader.dataset)` gives a wrong
  accuracy. Count the bursts batch by batch instead, as
  `scripts/train_sup_contrastive.py` does.
- A parameter count goes stale silently and depends on the class count.
  Recompute it before quoting it, and name the constructor arguments.
  `BirdModel(4, 30, ·)` has 6,309 parameters at 9 classes and 6,340 at 10; the
  old notes quote 10-class counts. `MaskedAutoencoderViT` was logged at
  9,546,756 and now has 9,557,508 at `img_size=20`, because two 21x256 position
  embeddings became learnable. `ResNet18_1D(9, dropout=0.3)` was logged at
  60,964 and now has 61,289.

## Early runs (exp15-exp133)

These come from the training logs of old scripts, mostly `BirdModel` on older
data files; no script in the repo reruns them. Accuracy is valid accuracy in %,
as recorded in `experiments_log.md`, which has the details. The data and the
seed alone moved it by up to several points (see below), so small differences
here are not reliable.

- AdamW (learning rate 3e-4, weight decay 1e-2, StepLR with step 2000 and factor
  0.1) beat SGD: 95.4% against 90% (exp45, exp42). Weight decay from 1e-3 to
  3e-1 changed little: 95.4% and 95.7% (exp53, exp54). Cosine schedules gave
  93.1-95.7%, no clear gain (exp47, exp55-58).
- Batch sizes from 1566 down to 25 gave 94.0-95.1% (exp59-63). A batch of 6 gave
  92.8% and was very slow (exp64).
- Without dropout, exp46 overfits after about 1000 epochs. Heavy dropout can put
  train accuracy below valid, because dropout is off at evaluation. An MAE
  encoder with dropout 0.7 reached 78.82% (exp91), with 0.1 86.93% and a smaller
  gap (exp92); without dropout the gap is gone (exp87, exp96). A valid share of
  10, 30 or 50% did not change this (exp91, exp93, exp94).
- A wider `BirdModel` (width 96, 58,282 parameters at 10 classes) gave 93.7%
  and overfit, against 94.5% at width 30 (exp67, exp45). Four layers gave 94.0%
  (exp70), and 94.8% with a residual connection (exp71).
- 8 attention heads gave the small transformer 85.8%, against 84.92% for exp85
  (exp87, embedding size 16).
- No augmentation was best: 95.36% for exp125 (`starts.csv`). Jitter and scaling
  with sampling (exp126), plus time and magnitude warp (exp129), or with the GPS
  speed held fixed (exp130), all gave 89.61%.
- Sampling alone did not hurt, against what the old notes say: exp127 has 95.84%
  valid against 95.36% for exp125; only its train accuracy is lower, 94.29%
  against 96.94%. Rerun it before citing it either way.
- GPS speed matters: without it, 93.76% against 95.36% (exp131, exp125); the
  old notes name Boat and SitStand. Dividing the speed by its maximum raised
  accuracy from 25% to 53% early on (exp19).
- Less data costs accuracy: 20% of the data gave 89.4% (exp66), against about
  95% for all of it (exp45). 25% more data than exp45 gave no gain, 94.3%
  (exp78).
- Small changes of the data or the seed moved valid accuracy by up to several
  points, never fully explained (exp98, exp109-112). exp45 alone has three
  recorded accuracies: 94.5%, 95.4% and 96.15%. Since then, runs fix the seed at
  32984 and compare seed spreads (Measuring).
- Supervised contrastive loss tracked the plain classifier (con1, con2).
  Mean-entropy maximization, a soft class balance, lowered the accuracy on the
  common classes (con3, con4). Balancing by weights or resampling traded overall
  accuracy for rare-class recall: 87% with balancing against 90% without (exp28,
  exp27).
- The best MAE head pools the tokens without a class token: 91.08% valid, 94.55%
  train (exp96); without LayerNorm before the pool, 89.47% (exp97). Fine-tuning
  the whole pretrained network beat training only the class head by far: 82.5%
  and 83.5% against 48.6% (f2, f3 against f1). This bears on the pretraining
  TODO in `status.md`, which plans to fine-tune only the last layer.
- BatchNorm caused a large train-valid gap until `track_running_stats=False`
  (exp15, fixed in exp22 and exp23). With a very low batch size, exp29 and exp31
  could not reproduce earlier numbers until the code went back to commit
  ed7d9a389; since then each run in the log carries its commit hash.
