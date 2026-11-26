---
marp: true
theme: default
paginate: true
size: 16:9
---

<!-- Slides. See README.md for where the figures come from and how to remake
them. Render with Marp: see "How to present this" at the end of README.md.
Valid accuracy, %: exp194 no augmentation (96.4), exp195/exp196 rotation (92.2/92.7, the same
config rerun, difference is noise), exp197 rotation + magnitude channels (94.5).
All BirdModelSmallDilated on starts.csv, seed 32984. Keep image height at 400;
taller pushes the caption off the slide. -->

# Reading gull behaviour from a tag

Nine behaviours from one second of accelerometer and GPS

A model small enough to run anywhere: **5,129 parameters**


---

## The data

![h:400](/home/fatemeh/Downloads/bird/screenshots/behavior_classes.png)

Nine behaviours. 1 second, 20 samples, 3 acceleration axes + GPS speed.
4,338 labelled bursts.

---

## What the classifier sees

- Three acceleration axes and one GPS speed, 20 samples each
- A small stack of convolutions, widened to span the whole second
  *(dilated 1-D CNN, receptive field 25)*
- **5,129 parameters**, fewer than before, yet 96.4% vs 94.8% accuracy
- Seeing the whole second helps pecking most: F1 0.79 → 0.94
  *(exp192 → exp194, scripts/batch_train_supervised.py)*

---

## The tag records its mounting, not only behaviour

The accelerometer measures in the **tag's own frame**, not the bird's. Every
reading mixes:

- **how the logger is tilted**, and where it sits — neck, back, wing
- **the environment** — wind, water impact, air resistance, a collision
- **gravity**, always present in the raw signal
- and, somewhere in there, what the bird is actually doing

Only the last one is the label. Our data is **one logger type, mounted much the
same way on every bird**, so the standard test split cannot see any of this.

---

## A turned tag: 13% accuracy, or 91% with rotation

| test bursts, accuracy % | no augmentation | with rotation |
|---|---|---|
| normal | **96.4** | 92.2 |
| x/y axes swapped *(Ornitela convention)* | 71.5 | **90.4** |
| tilted 70° | 2.1 | **90.6** |
| any random orientation | 13.1 | **90.7** |

Not merely wrong — **confidently** wrong. 2% is far below the 11% you would get
by guessing. *(exp194, exp195, 438 test bursts, exps/eval_labeled.py)*

---

## Rotation costs 4 points, holds 91% when turned

Train on the same bursts **rotated to every possible orientation**
*(random SO(3) rotation of the acceleration axes)*.

- Costs **4 points** of accuracy on the birds we have: 96.4% → 92.2%
- Holds **~91% accuracy at any mounting orientation**, instead of collapsing
- The cost lands on walking and pecking *(F1 0.97 → 0.79, 0.94 → 0.68)*,
  told apart by the *direction* of small movements relative to gravity,
  exactly what rotating destroys *(exp194 → exp195, exps/eval_labeled.py)*

Worth it whenever the model will meet a tag it was not trained on.

---

## Three magnitude channels win back half the cost

Turning the tag changes **which axis** records a movement, never **how big** it
is. So we give the model three sizes it cannot be confused about: how hard the
bird is accelerating, how much it moves once gravity is subtracted, and how
abruptly that changes *(rotation-invariant magnitudes)*.

- Costs **300 parameters**: 5,129 → 5,429
- Accuracy **92.7% → 94.5%**; at any orientation **92.0% → 93.7%**
- Rare behaviours recover most: class-balanced F1 **0.83 → 0.90**
- On a new bird it changes its answer **3x less often** when the tag is turned

*(exp196 → exp197; exps/eval_labeled.py, exps/eval_unlabeled.py)*

<!-- All five numbers are one A/B pair, same seed and split. -->


---

## How do we know it works on a new bird?

We have millions of unlabelled bursts and no answers for any of them.

**The obvious idea: trust the model's confidence when it is unsure.**

It does not work.

---

## Confidence stays at 0.74 when accuracy is 13%

Turned at random, exp194's accuracy fell from 96% to 13%:

- its mean confidence fell only from **0.96 to 0.74**
- 0.74 looks like a healthy model *(exps/recompute_doc_numbers.py)*

On 115,266 unlabelled bursts from a new bird, confidence rates the two models
**0.89 vs 0.91** — indistinguishable, while one makes 16x more impossible
predictions *(exps/eval_unlabeled.py)*.

Confidence is useful for **ranking** which bursts to look at, never for asking
whether the model is healthy.

---

## Counting impossible predictions needs no labels

No labels needed. Each check contradicts the model with something we know
independently.

- **Impossible speed** — "sitting still" at 13 m/s
- **Impossible place** — "walking" a kilometre out to sea *(land/sea mask)*
- **Impossible flicker** — behaviour changing every second in one GPS fix
- **Rotation flip** — a different answer when the tag is rotated

For flicker, "impossible" has to be learned: gulls really do walk, peck and walk
again. From the labelled data, only **10 of 36** behaviour pairs ever follow one
another within a fix. Everything else counts
*(label co-occurrence, exps/label_cooccurrence.py)*.

---

## New bird: fewer impossible speeds and flips

115,266 bursts, device 6004: exp194, 196, 197 *(exps/eval_unlabeled.py)*

| | mean confidence, 0–1 | impossible speed | impossible place | impossible flicker | rotation flip |
|---|---|---|---|---|---|
| no augmentation | 0.89 | 1.45% | 1.18% | 6.7% | 82.8% |
| with rotation | 0.91 | **0.09%** | 1.45% | 4.9% | 5.4% |
| + invariant channels | 0.95 | 0.11% | **1.09%** | **4.0%** | **1.7%** |


---

## 26 of 4,338 bursts contradict their own label

The same speed rule, pointed at the *labels* instead of the predictions.

- 10 SitStand, 11 TerLoco, 5 Pecking *(exps/find_label_noise.py)*
- Clearest: a bird labelled "sitting still" moving at **13.7 m/s**
- 16 of the 26: one device in about 20 minutes — might be a **broken GPS**
- Retrained without them: same accuracy, **impossible speed 2.5-3.3x rarer at
  every seed tried** *(exp199-204, exps/eval_unlabeled.py)*
- GPS speed both **flags** the labels and **judges** the fix, so we keep the raw
  model *(exp197)*

Mentioned in `behavior/data_processing.py::remove_label_noise`.

---

## Rotating on the GPU: 47 to 5 minutes per run

The augmentation was applied one burst at a time, in Python.

- Same result: 92.7% vs 92.2% accuracy, well inside run-to-run noise
- The trick: rotate the whole batch at once on the GPU, and stop rebuilding the
  data every epoch

Nine times more experiments per day, for no change in the science.
*(exp195 → exp196, exps/eval_labeled.py)*

---

## Spoonbills: few birds per behaviour

Data counts, no model run *(Lok et al. 2023, 2025; exps/eval_spoonbill.py)*:
5,080 labelled 1 s segments, 14 behaviours, 9 birds

| group: behaviours | segments | birds | share of the top bird |
|---|---|---|---|
| rest: stand, sit, preen, drink | 2,174 | 6 | 84% |
| forage: search, handle, ingest prey | 2,058 | 3 | 91% |
| beg: a chick asks to be fed | 434 | 1 | 100% |
| fly: flap, soar | 237 | 7 | 40% |
| walk | 177 | 3 | 69% |

Only 24% of prey-ingestion records fit in a whole 1 s segment
*(exps/spoonbill_data_stats.py)*

---

## Spoonbills: split by fix, not by segment

A fix: one GPS time of one bird, with up to 200 records (10 s). We cut it into
segments of 20 records (1 s), one model input each.

- **Split per segment** *(papers, our exp205)*: each segment goes to train or
  test on its own. **502 of the 514** test segments had a segment of the same
  fix in training *(exps/eval_spoonbill.py)*
- **Split per fix** *(exp206-211)*: all segments of a fix go to one side.
  Test: 505 segments from 79 fixes. Train: 4,575 segments from 702 fixes
- Same birds in train and test: new fixes, not new birds

---

## Spoonbills: rotation costs about 10 points

505 test segments, 3 seeds each, F1 the mean of the 3
*(exp206-211, exps/eval_spoonbill.py)*

| | with rotation | without rotation |
|---|---|---|
| accuracy, %, range | 83.96–84.55 | **93.27–94.65** |
| standing alert, F1 0–1 | 0.49 | **0.87** |
| walking, F1 0–1 | 0.42 | **0.78** |

- Rotated, sitting alert is 0 of 61 right, even in training
  *(exp206, scripts/batch_train_supervised.py)*
- Behaviours differ by **posture**, the gravity direction rotating destroys
- The price: not robust to how the tag sits. Not tested on a new bird

---

## Spoonbills: paper windows can mix behaviours

- **Segments** *(our training)*: cut where a behaviour starts, so each holds
  one behaviour. 505 test segments; a bout under 20 records gives none
- **Windows** *(the papers)*: cut every 20 records from the start of the fix,
  whatever the labels. 602 test windows, 133 hold two behaviours

One answer per window, checked against each of its 20 records
*(exp207, exps/eval_spoonbill.py)*:

| a test window's 20 records | model says | records right |
|---|---|---|
| 7 prey ingestion, then 13 search | search | 13 of 20 |
| 11 search, then 9 prey ingestion | prey ingestion | 9 of 20 |

---

## Spoonbills: level or better, except ingestion

Scored on windows, as the papers score: F1 0–1 per record. Ours: without
rotation, mean of 3 seeds *(exps/eval_spoonbill.py)*

| F1 | forage | walk | fly | rest | beg |
|---|---|---|---|---|---|
| Lok et al. 2025 *(random forest, 1.6 s)* | 0.96 | 0.61 | 0.98 | 0.95 | 0.89 |
| ours *(1 s, no rotation)* | 0.98 | **0.74** | 0.97 | **1.00** | **0.92** |

- Lok et al. 2023: walk 0.65 → **0.81**, search 0.91 → 0.90, prey ingestion
  *(for-intake)* 0.73 → 0.59: **worse, with rotation too** (0.51)
- Their split shares fixes between train and test; ours does not

---

## Gull model on spoonbills: most behaviours match

exp197 on all 5,080 spoonbill segments *(exps/eval_cross_species.py)*

| spoonbill behaviour | gull class given, % of the behaviour's segments |
|---|---|
| rest, alert | sit/stand 93–100 |
| preening | sit/stand 77 |
| flapping, soaring | flapping, soaring 98 |
| handling, swallowing prey | pecking 100 |
| walking | walking 67 |
| searching for food | pecking 52, walking 27, sit/stand 19 |

87% of 4,646 segments get the ecologist's first or second choice

---

## Spoonbill models on gulls: higher with rotation

Spoonbill models on all 4,338 gull bursts, 3 seeds per arm, epoch 4000
*(exp206-211, exps/eval_cross_species.py)*

| | with rotation | without rotation |
|---|---|---|
| ecologist's class, % of 3,433 bursts | 75.5–84.7 | 65.8–67.4 |

- Without rotation a walking gull reads as a standing spoonbill: 222 of
  337 *(exp207)*. Both tilt the tag forward alike: x −0.41 and −0.38 g
- Flapping gulls read as flapping spoonbills in 46–97% of bursts,
  depending on the seed

---

## One class list for both species: lower accuracy

Both species, no rotation, one seed, epoch 4000 *(exps/eval_joint.py)*

- **Sets** *(exp213)*: gull sit/stand may be any spoonbill rest or alert
- **23 classes** *(exp215)*: each behaviour of each species its own class

| valid accuracy, % | gull, 438 bursts | spoonbill, 505 segments |
|---|---|---|
| one model per species, 3 seeds | 93.84–94.06 | 93.27–94.65 |
| sets | 91.32 | 92.08 |
| 23 classes | 87.90 | 87.13 |

One model per species: gull exp199, 201, 202 *(rotation)*, spoonbill exp207,
210, 211. 23 classes: the other species' behaviour counts as wrong

---

## Coarse-fine model with a species token

- **One shared network** for gulls and spoonbills, 6,666 parameters
  *(scripts/train_species_heads.py)*
- **Coarse output**, the same 4 groups for both species: flying, water,
  still on land, active on land
- **Fine output per species**: 9 gull classes, 13 spoonbill classes
- **Species token**: gull, spoonbill or unknown. Unknown for half of the
  training, so the model can also run on another species

---

## Coarse-fine model, species token: same accuracy

Valid accuracy %, epoch 4000, mean of 3 seeds, on 438 gull bursts and 505
spoonbill segments *(exp216-221, exps/eval_species_heads.py)*. One model per
species: gull exp199, 201, 202 *(exp197's settings)*; spoonbill exp206, 208,
209 *(rotation)* and exp207, 210, 211 *(no rotation)*

| model | gull | spoonbill |
|---|---|---|
| one model per species, rotation | 93.99 | 84.36 |
| coarse-fine, rotation | 93.99 | 84.16 |
| one model per species, no rotation | – | 94.06 |
| coarse-fine, no rotation | 93.91 | 93.93 |
| coarse-fine, no rotation, token unknown | 94.37 | 94.06 |

---

## Gull rest or alert: not testable without labels

Gull bursts read through the spoonbill fine output, token unknown
*(exp216-221, exps/eval_species_heads.py)*

- Flying reads as flying; walking as walking in all 3 runs without
  rotation: 194–270 of 337 walking bursts
- Sit/stand → rest or alert: with rotation 3 runs agree, 846–890 of 1,502
  rest; without, one reverses it: 386 rest, 904 alert *(exp221)*
- Rest and alert: both a still bird with the same tag tilt, x −0.38 and
  −0.40 g *(exps/eval_cross_species.py)*
- **Ask:** 50–100 gull sit/stand bursts labelled rest or alert, from
  video or field notes

---

## Two species: what we learned

1. **Coarse behaviours transfer between species**: still, flying, active
   on land *(exp197, exp206-211, exps/eval_cross_species.py)*
2. **Rotation: about 10 points less on spoonbills, more expected classes on
   gulls** *(exp206-211, exps/eval_spoonbill.py, exps/eval_cross_species.py)*
3. **One class list for both species lowers accuracy** *(exp212-215,
   exps/eval_joint.py)*
4. **Coarse-fine model: the accuracy of one model per species, also with
   the token unknown** *(exp216-221, exps/eval_species_heads.py)*
5. **Fine behaviours, such as gull rest or alert, need labels of that
   species** *(exp216-221, exps/eval_species_heads.py)*

---

## What we learned

1. **A high test score can hide a total failure**: 96.35% valid accuracy,
   13.05% with the tag turned *(exp194, exps/eval_labeled.py)*
2. **Confidence is not a health check**: mean 0.74 at 13% accuracy
   *(exp194, exps/recompute_doc_numbers.py)*
3. **A model can be measured without labels**: count predictions that
   break physics *(exp194-197, exps/eval_unlabeled.py)*
4. **Check the labels too**: 26 of 4,338 bursts contradict their own label
   *(exps/find_label_noise.py)*
5. **Cleaner labels improved only the rule that chose them**: speed
   conflict *(exp199-204, exps/eval_unlabeled.py)*

---

## What is next

1. **Test on a real second logger**, not a synthetic rotation of the first
2. **Recover walking.** The new channels are sizes, so pecking and manoeuvring
   won part of their score back; walking won none. Walking is read from a
   *direction* — the body upright, the steps shaking it sideways — and a size
   cannot carry a direction. The way in is to estimate the tag's orientation on
   every burst *(gravity direction from the low-pass acceleration)*
3. **Automate the sea/land and flicker checks** on every new deployment
4. **Label ~200 bursts from a new bird**, the only way to get a real number
5. **The other sources of variation**: where the tag sits, wind, water impact

---

## Backup: numbers

- 4,338 labelled bursts, 9 classes, 20 Hz, 90/10 split, seed 32984
- Accuracy %, valid / train: exp194 no aug 96.35 / 99.44 — memorises
- exp196 rotation 92.69 / 91.56 — underfits, never memorises
- exp197 + 3 invariant channels (5,429 params) 94.52 / 97.90
- F1, 194→196: walking .97→.89, pecking .94→.68, manoeuvre .84→.67
- F1, 196→197: walking .89→.89, pecking .68→.76, manoeuvre .67→.79
- Class-balanced F1: 0.92 (exp194), 0.83 (exp196), 0.90 (exp197)
- Random orientation, accuracy %: 13.1, 90.7, 93.7 (exp194, 195, 197)
- Agreement on unlabelled bursts: exp194/196 87.5%, exp196/197 93.0%
  *(exps/eval_labeled.py, exps/eval_unlabeled.py)*

---

## Backup: numbers

- Label-noise check: 26 bursts dropped, `starts_clean.csv` 4,312 bursts
- Raw vs cleaned, 3 seeds each, same split and epoch: impossible speed
  0.10/0.15/0.13% → 0.03/0.06/0.04% of bursts, no overlap *(exp199-204)*
- Everything else — accuracy, place, flicker, rotation — sits inside the seed
  spread *(exps/eval_labeled.py, exps/eval_unlabeled.py)*
- Spoonbill, split by fix, 4,575/505 segments, accuracy % at epoch 4000:
  rotation 84.55 / 84.55 / 83.96, none 94.26 / 93.27 / 94.65
  *(exp206-211, exps/eval_spoonbill.py)*
- Spoonbill F1 0–1 on records, ranges over 3 seeds, no rotation: walk
  0.71–0.75, beg 0.91–0.93, ingest 0.56–0.63; paper 2025 at 0.8 s: walk 0.60,
  beg 0.86 *(exps/eval_spoonbill.py)*
