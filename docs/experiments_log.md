# Experiment log

Raw notebook of individual runs, terse style. Held to a lower bar than
[docs/lesson_learned.md](lesson_learned.md) (curated lessons) and
[docs/descriptions.md](descriptions.md) (data/model/script overview).

Accuracies are validation unless noted; `tr-val` gives train then valid. A
trailing `hash:...` is the git commit the run was made at. Numbering has gaps
(runs that were abandoned or folded elsewhere). Newest entries first.

## exp201-204: three seeds a side, and the ExFlap question

Four new runs: exp201/exp202 are exp199 (`starts.csv`) at seeds 1234 and 5678,
exp203/exp204 are exp200 (`starts_clean.csv`) at the same two. With exp199 and
exp200 at seed 32984 that is **3 raw runs (199, 201, 202) against 3 clean runs
(200, 203, 204)** — every triple below is in that order. `split_seed` is held
at 32984, so all six are scored on the same 437 bursts and the same 4 ExFlap
ones, at epoch 4000; only the trajectory moves. hash:2e4bc39 plus the
uncommitted changes. 4 runs, 5m each.

- **Speed conflict is the one real effect, and it is clean.** raw 0.10/0.15/0.13
  against clean 0.03/0.06/0.04 — the two ranges do not overlap, every clean run
  beats every raw run (`exps/eval_unlabeled.py`).
- **Every other check is inside the seed spread.** place 1.12/1.08/1.00 vs
  1.03/1.07/1.04, unseen flip 4.09/3.97/3.46 vs 3.88/3.86/3.44, rotation flip
  1.57/1.62/1.34 vs 1.61/1.53/1.36. Ranges overlap, no effect.
- **The accuracy gap does not survive either.** valid 93.82/94.05/94.05 vs
  93.59/93.36/94.05, means 93.97 and 93.67. The best clean seed ties the best
  raw seed, so the 0.23 points exp199 vs exp200 showed was one seed pair.
- **ExFlap: a shift in the odds, not a break.** On b0 (534, 2012-06-08
  18:18:19, 15.84 m/s) raw is right 3 of 3, clean 1 of 3; on b3 (782,
  2014-05-26 10:41:11, 4.99 m/s) raw 3 of 3, clean 2 of 3. But one clean seed
  (exp204) gets all 4 at 0.79-0.93 confidence, and one raw seed holds b0 at only
  0.41. Three seeds cannot carry more than a direction here.
- **The mechanism is GPS speed, measured.** ExFlap runs slow (median 1.39 m/s,
  p90 8.90) and Flap fast (median 9.63, p10 4.92). b0 at 15.84 and b3 at 4.99
  are the two ExFlap bursts with Flap-like speeds; b1 and b2 at 0.06 m/s are
  never wrong in any run. The 26 dropped bursts are the only fast-but-not-flying
  training examples there were (3.24-17.78 m/s, all SitStand/TerLoco/Pecking),
  so dropping them sharpens **fast -> Flap**. Sweeping the GPS channel with the
  accelerometer held fixed: exp200 P(ExFlap) on b3 falls 0.45 -> 0.06 from 0 to
  20 m/s where exp199 stays flat at 0.79 -> 0.44.
- Conclusion: **clean the labels.** The fix lands where it was aimed, 2-3x
  fewer physically impossible predictions, and costs nothing measurable else.
  Cleaning a land class can still move a flight class, because the classes share
  the GPS-speed feature, not because the behaviours are related.

## exp199/exp200: the clean-label A/B, made comparable

exp198 vs exp197 mixed three things: the cleaning, a reshuffled split and a
best-validation checkpoint from a different epoch. This pair removes the last
two. exp199 = exp197's config on `starts.csv`. exp200 = `starts_clean.csv` with
`split_file` pointing at `starts.csv`, so it validates on **the same bursts**.
Both keep epoch 4000 (`save_final`), and both are read there. exp197 untouched.
hash:2e4bc39 plus the uncommitted changes. 3900/438 and 3875/437, 5m each.

- **exp199 reproduces exp197**: best valid 94.52 at epoch 1996, same as exp197,
  final 93.84, plateau 93.89. The pipeline is near-deterministic at one seed, so
  what exp200 does differently is the data.
- The only difference between the two training sets is the **25 noise bursts**
  exp199 keeps (`exps/eval_labeled.py` prints this; the 26th is in the valid
  split of both).
- Labeled, epoch 4000, the same 437 bursts: valid **93.82 vs 93.59** — 1 burst.
  Plain F1 0.94 both, every class within 0.03 except ExFlap.
- **Orientation robustness is now level**, on both measures. Labeled SO(3)
  accuracy 93.94 vs 93.90 (`exps/eval_labeled.py`), where exp197 vs exp198 on
  their 260 shared bursts was 95.04 vs 94.31. Unlabeled rotation flip 1.57 vs
  1.61 (`exps/eval_unlabeled.py`), where exp197 vs exp198 was 1.72 vs 2.00.
  Two different measurements from two different scripts; both gaps fall to 0.04.
- Unlabeled (device 6004): speed conflict **0.10 vs 0.03**, place **1.12 vs
  1.03**, unseen flip **4.09 vs 3.88**, rotation flip 1.57 vs 1.61, mean
  confidence 0.95 vs 0.96. Agree on 97.2%. exp200 is better or level on every
  check — place and unseen flip **change sign** from the exp197/exp198 reading.
- The ExFlap bursts and the accuracy gap are followed up in exp201-204; one
  seed each settles neither.
- Conclusion: cleaning the labels is neutral to positive. The earlier
  regressions were the checkpoint, not the data.

## exp198: exp197 retrained on starts_clean.csv

exp197 with `data_file=starts_clean.csv`, the 26 label-noise bursts dropped.
Model, seed, augmentation and schedule unchanged, so it is a clean A/B on the
data. hash:2e4bc39 plus the uncommitted config change. 3877/435 split, 5m14s,
best epoch 1544.

- val **94.02** / train 95.98 vs exp197's 94.52 / 97.90. Valid F1 0.94 plain,
  0.84 balanced (exp197 0.95 / 0.90).
- The two runs do not share a valid split: dropping bursts reshuffles
  `stratified_split`, so 177 of exp197's 438 valid bursts are in exp198's train
  set. Neither number above is comparable; exp199/exp200 redo it properly.
- The balanced F1 gap is 2 bursts: ExFlap 2 of 4 wrong, both called Flap, where
  exp197 got 4 of 4. Every other class within 0.05 plain F1.
- Unlabeled (device 6004, `exps/eval_unlabeled.py`): speed conflict **0.03 vs
  0.11%**, place 1.16 vs 1.09, unseen flip 4.82 vs 4.04, rotation flip 2.00 vs
  1.72. Agree on 96.8%.
- **The 0.50 headline gap is checkpoint choice, not data.** Plateaus: exp197
  93.96 mean over the last 500 epochs (93.61–94.29, final 93.84), exp198 flat
  93.79. `<exp>_best.pth` is the best valid epoch — 1996 for exp197, 1544 for
  exp198 — and at matched epochs exp198 fits train harder (97.76 vs 97.13 at
  epoch 1996), so it is not the weaker run; its argmax landed earlier.
- The perturbation gaps likely follow the same thing: exp198's checkpoint is
  ~450 epochs younger and less confident (0.93 vs 0.95), and a less confident
  model flips more. Untested — no fixed-epoch checkpoint was saved.
- Conclusion: the same model, as expected. Plateaus differ by 0.17 points and
  only the check the cleaning targets moves.

## exp197: rotation-invariant magnitude channels

exp196 with `add_magnitudes=True`: input `[x, y, z, gps, mag, dyn_mag, jerk_mag]`, `BirdModelSmallDilated(7, 20, 9)`, 5,429 params. Data, seed, split,
augmentation and schedule unchanged, so it is a clean A/B on the features.
hash:202bf6f plus the uncommitted 7-channel change. 4m56s, best epoch 1870.

- val **94.52** / train 97.90 vs exp196's 92.69 / 91.56. Valid F1 0.95 plain,
  **0.90 balanced** (exp196 0.93 / 0.83).
- Not seed noise: last-500-epoch plateaus do not overlap, 92.29 (91.78–92.69) vs
  93.95 (93.61–94.29).
- Orientation: SO(3) mean 93.66 vs 92.03, 70° pitch 93.61 vs 91.55.
- Per-class valid F1: Manouvre +0.12, Pecking +0.08, Boat +0.06, TerLoco +0.00.
  ExFlap +0.14 is 4 bursts, ignore.
- Unlabeled (device 6004): rotation flip **1.72 vs 5.37%**, unseen flip 4.04 vs
  4.92, place 1.09 vs 1.45, speed 0.11 vs 0.09 (worse). Agree on 93.0%.
- Where the three weak classes go wrong (`exps/plot_errors.py`, 15 plots in
  `/home/fatemeh/Downloads/bird/results/errors_exp197`): TerLoco 3 of 34 wrong
  (2 to Pecking) and 5 wrongly predicted (3 from Pecking); Pecking 7 of 23 wrong
  (3 to TerLoco, 2 Soar, 2 SitStand); Manouvre 5 of 16 wrong (4 to Soar).
  Walking and pecking mostly trade with each other, Manouvre loses to Soar.

## exp196: exp195 repeated through the batched GPU path

Same model, data, seed and split as exp195; only the augmentation wiring differs
(`bd.GpuBatches` + `bau.BatchRandomRotation3D` instead of the per-sample
transform in `BirdDataset`). hash:520efd0. **5m03s against exp195's 47m31s, 9.4x.**

- val **92.69** (best, epoch 1885) / train 91.56, against exp195's 92.24 / 90.21.
  Plateau 92.24–92.47 over the last 500 epochs (exp195: ~91.1), final epoch 92.24.
- Valid F1 within ±0.01 of exp195 on 7 of 9 classes. The two that move are the
  smallest: TerLoco +0.10 (34 valid bursts) and Manouvre -0.14 (16 bursts, so
  1–2 samples). Class-balanced valid F1 0.84 -> 0.83.
- Orientation robustness holds: random SO(3) mean 92.03 (std 0.47) against
  exp195's 90.67 (std 1.00); x/y swap 93.15 vs 90.41.
- **Conclusion: same result, within the seed noise this dataset already shows.**
  The RNG stream differs between the two paths, so equality was never expected.

## exp195: rotation augmentation

BirdModelSmallDilated + full SO(3) IMU rotation (train set only), starts.csv,
9 class, 4000 epochs, hash:62e5b67. Clean A/B vs exp194: identical model
(`mid_channels=20`, `dropout=0.15`), split (3900/438) and seed (32984); only the
transform differs.

- val **92.24** (best, epoch 1308) / train 90.21 on unaugmented train data, vs
  exp194's 99.44/96.35. Valid plateaus at ~91.1 from epoch ~1300; 47m31s.
- Off-orientation exp194 collapses (x/y swap 71.46, 70° pitch 2.05, random SO(3)
  13.05) where exp195 holds 90.4–91.6. Table in
  [docs/lesson_learned.md](lesson_learned.md), from
  `exps/eval_labeled.py`.
- Valid F1 TerLoco .97 -> .79 and Pecking .94 -> .68 carry the loss; balanced valid
  F1 0.92 -> 0.84 (`exps/eval_labeled.py`).

## Configs for exp196-exp204

Kept here because `scripts/batch_train_supervised.py` holds only the run in
progress. Everything not listed comes from its `base_config`: seed 32984, 4000
epochs, `train_per` 0.9, `batch_size` None (one full batch), AdamW, StepLR with
`step_size` 2000 and `warmup_epochs` 1000, `max_lr` 3e-4, `weight_decay` 1e-2,
labels `[0, 1, 2, 3, 4, 5, 6, 8, 9]`. Augmentation is
`bau.BatchRandomRotation3D`, set in `main()`, train set only.

All nine use the same model entry, `in_channels` 7 except exp196, which is 4:

```python
"model": {
    "name": "BirdModelSmallDilated",
    "parameters": {
        "in_channels": 7,
        "mid_channels": 20,
        "out_channels": len(all_labels),
        "dropout": 0.15,
    },
},
```

A dash means the key is not set, so the `base_config` default applies.
`data_file` and `split_file` live in `/home/fatemeh/Downloads/bird/data/final/`.

| exp | data_file        | add_magnitudes | split_file | seed  | split_seed | save_final |
| --- | ---------------- | -------------- | ---------- | ----- | ---------- | ---------- |
| 196 | starts.csv       | False          | -          | 32984 | -          | -          |
| 197 | starts.csv       | True           | -          | 32984 | -          | -          |
| 198 | starts_clean.csv | True           | -          | 32984 | -          | -          |
| 199 | starts.csv       | True           | -          | 32984 | -          | True       |
| 200 | starts_clean.csv | True           | starts.csv | 32984 | -          | True       |
| 201 | starts.csv       | True           | -          | 1234  | 32984      | True       |
| 202 | starts.csv       | True           | -          | 5678  | 32984      | True       |
| 203 | starts_clean.csv | True           | starts.csv | 1234  | 32984      | True       |
| 204 | starts_clean.csv | True           | starts.csv | 5678  | 32984      | True       |

`split_seed` and `save_final` did not exist when exp196-exp198 ran.

## Reference

Label map:

```python
ind2name = {0: "Flap", 1: "ExFlap", 2: "Soar", 3: "Boat", 4: "Float",
            5: "SitStand", 6: "TerLoco", 7: "Other", 8: "Manouvre", 9: "Pecking"}
```

`s_data` burst counts (index): `{0:634, 1:38, 2:501, 3:176, 4:558, 5:894, 6:318, 7:25, 8:151, 9:210}`. `starts` (current pipeline, no Other): total 4338
`{5:1502, 4:729, 0:643, 2:537, 6:337, 9:225, 3:176, 8:151, 1:38}`.

Behavior grouping (peck labels are noisy):

```
Flight:  0 Flap 634, 1 ExFlap 38, 2 Soar 501 (thermal soaring), 8 Manoeuvre/Mixed 151
Float:   4 Float 558
SitStand:5 SitStand 894, 3 Boat 176
TerLoco: 6 TerLoco/Walk 318, 9 Pecking 210 (paper 209)
Other:   7 Other 25
```

Parameter counts, recomputed from the current code unless marked historical:

- `BirdModel(4, 30, ·)`: **6,309** at 9 classes, **6,340** at 10. The class head
  is in both — the 31-parameter gap is the extra class. The old notes quote the
  10-class numbers throughout, so their width pair is 6,340 -> 58,282 (both
  10-class); the 9-class width-96 model is 58,185.
- `BirdModelWideRF`: **6,889**, RF 19 (three k=7 convs). `BirdModelSmallDilated`:
  **5,129**, RF 25 (three k=5 convs, dilation 1/2/3). Both `mid_channels=20`.
- `ResNet18_1D(9, dropout=0.3)`: **61,289**. (`model_descriptions.txt` says
  60,964; that no longer reproduces — the stage/channel layout changed, and
  `ResNet1D` only builds 3 of the 4 stages in `layers`.)
- `MaskedAutoencoderViT` img_size=20: **9,557,508** (img_size=60: **9,577,988**).
  An earlier 9,546,756 circulated in these docs; it is stale by 10,752, the two
  21x256 position embeddings.
- `TransformerEncoderMAE` embed_dim 256, depth 6: **4,748,297**.
- Historical transformer sweep by embed_dim (from `model_descriptions.txt`,
  depth 1): 512 -> 3,170,816; 256 -> 798,976; "126" -> 196,686; 64 -> 52,288;
  32 -> 13,856; 16 -> 3,856; 8 -> 1,160. None reproduce exactly today (depth-1
  dim-256 is now 799,497) and 126 is not a legal embed_dim (must divide by
  num_heads), so treat the whole row as historical. Note it is a *different*
  depth from the 4,748,297 entry above, which is also embed_dim 256.

Data note: IMU clipped to [-2, 2] (exp190; only 782/86,760 rows were outside),
GPS 2D speed normalized by 22.3. Standard seed is 32984.

## Baselines and class subsets

Best runs:

- exp44 `[0,3,4,5,6]`: 99.2% (hash:dac5a78) `***`
- exp45 no-Other (9 class): 94.5% (hash:8b3b3d9); exp47 slightly better. Three
  numbers exist for this run in the old notes — 95.4% as first recorded, 94.5%
  on rerun, and 96.15% at seed 32984 (quoted as 96.1 in the exp78 line). The
  data slice differs between them; see the data note.
- exp69 all 10 class: 90.3% (hash:d2eb8fb)
- exp98: 94.7% (random split, so far the best)
- exp34 all 10 class, class-balanced: 89.1% (hash:850e071)
- exp73 `[8,9]`: 91.9% at train_per=.1 (324/37), whose val was too small and hit
  100% at epoch 32 on a lucky set; rerun at train_per=.5 (180/181) -> 97.2%
  (hash:d5ff6ef). Large overfit either way from little data.
- exp75 (= exp45, Soar+Manoeuvre merged): 94.5%, slight overfit; merge did not
  help Soar/Manoeuvre but Pecking improved.

Older subset runs: exp35 `[0,2,3,4,5,6]` 95%; exp36 `[0,3,4,5,6]` 98%;
exp42 no-Other 90% (hash:2765b7d); exp132 `[0,2,4,5,6,9]`, exp133 `[0,3,6]`,
exp134 `[2,3,4,6,9]` on starts.csv.

Leave-one/two-out sweeps: exp135–143 drop 1 of `[0..9]` (8-class, 9 models);
exp144–179 drop 2 (7-class, 36 models); exp180–184 and exp185–189 drop 1 of
`[0,2,4,5,6]` (4-class, 5 models each; the latter on balanced `balanced_02456.csv`).

## Dataset / data pipeline

- exp76: repeat exp45 from CSV -> 93.1 (hash:04ebe52); different data slice, hence lower.
- exp77: 10-point bursts instead of 20 (Nx20x4 -> 2Nx10x4) -> 90.37 (hash:26e2482), lower.
- exp78: more data (as exp45, no Other; 3928+437 vs exp45 3132+348) -> 94.3
  (hash:70d23b7). ~25% more data, no improvement.
- exp107/108: combined_unique 8742 rows. tr-val 97.2/94.8 and 96.8/94.7.
- exp109–112: s_data variants, tr-val 93.45/91.95 (exp109, mapped to ÷20; at seed
  1234 it is 94.89/92.82), 95.75/92.82 (exp110, s_data original), 95.31/92.53
  (exp111, s_data json), 95.7/**97.67** (exp112, combined_unique mapped to ÷20 —
  val above train, unexplained). Seeds shift val ~2.5 pts for reasons never
  explained; all reported numbers use seed 32984.
- exp113: combined_unique, IMU only -> tr-val 93.47/95.13.
- exp124: new pipeline, more data by shift-with-rules, 9 class -> tr/val/test
  96.5/?/95. Both source files record the val number as "9.4", which is not a
  plausible accuracy; 94.9 is the obvious typo repair but it is a guess, not a
  recorded value. Rerun if the number matters.
- **exp125**: new pipeline, `starts.csv` (start labels only), 9 class -> 96.94/95.36.
  Best of the exp124–exp131 family and the long-standing reference; beaten since
  by exp194 (96.35 val, same data) — see "Compact models and app reruns".
- exp131: as exp125 but no GPS -> tr-val 92.11/93.76. GPS mainly helps Boat and
  SitStand (consistent with Judy's paper).

Augmentation sweep on starts.csv (all worse than none):

- exp126: random crop + jitter + scaling (σ=0.05) + dataloader sampling -> 91.14/89.61.
  Jitter lowered overall but improved rare classes.
- exp127: no augmentation, sampling only -> 94.29/95.84. Better on rare classes.
  The old notes call this "lower than no sampling", but 95.84 val is *above*
  exp125's 95.36 — only the train number is lower. Unresolved; do not cite
  exp127 as evidence that sampling costs accuracy without rerunning it.
- exp129: exp126 + time & magnitude warp -> 91.42/89.61, very slow, slightly worse.
- exp130: exp126 but GPS held fixed -> 91.14/89.61. Better than warping GPS (exp126)
  but still below sampling-only (exp127).

Balanced data: exp114/115 `[0,2,4,5,6,9]` balanced (all data / s_data);
exp116–120 five random-balanced s_data splits; exp121 as exp116 but dim=16,
dropout 0.7 (~2000 params); exp122–123 same on shifted data.

## Compact models and app reruns (exp190–194)

All on the current pipeline. Numbers read from
`~/Downloads/bird/results/failed/<exp>/app_loss_acc.txt`.

| exp    | model / data                                   | train | valid     |
| ------ | ---------------------------------------------- | ----- | --------- |
| exp190 | BirdModel, starts.csv (IMU clipped to [-2, 2]) | 95.49 | 94.98     |
| exp191 | BirdModel, s_index (repeat exp45)              | 94.92 | 94.32     |
| exp192 | BirdModel, starts.csv (repeat exp125)          | 95.95 | 94.75     |
| exp193 | BirdModelWideRF, starts.csv                    | 99.67 | 95.21     |
| exp194 | BirdModelSmallDilated, starts.csv              | 99.44 | **96.35** |

- **exp194 is the best supervised run on `starts.csv` so far**, a point above
  exp125 (95.36) and 1.6 above exp192 (BirdModel on the same data). It is the
  app's configurable inference model and the no-aug baseline for exp195. Best in
  the canonical mounting orientation only — see exp195 at the top.
- exp193 (WideRF, RF 19) is 1.1 pts below exp194 (SmallDilated, RF 25) and was
  dropped as a tracked variant.
- exp192 did not reproduce exp125's 95.36 (94.75, -0.6) — within the ~2.5 pt
  seed/slice noise seen in exp109–112.
- Both compact models train to ~99.5% against BirdModel's ~96, so they overfit
  much harder; only the validation column is comparable across the rows.

## Rotation augmentation

See exp195 at the top. SmallDilated was picked over BirdModel for its larger
receptive field (RF 25 vs 7): it captures the global flap sine-wave, which the
shorter-context BirdModel confuses with Manoeuvre (seen by eye on unlabeled data
in `app/gps_burst_labeling_viz_app.py`).

- A discarded BirdModel + rotation trial (why the model was switched): val 91.55%,
  ~3.8 pts under the exp125 baseline it was run against (95.36) and ~4.8 under
  exp194. The cost did not fall on the orientation-dependent static classes
  (val F1 SitStand .97, Float .97, Boat .92, Flap .96 all held) but on the rare
  dynamic classes (Pecking .57, Manouvre .77, ExFlap .57) — consistent with the
  short receptive field confusing the flap pattern with Manoeuvre. Not kept as a
  tracked run.

## Data percentage

- setting exp36: exp37 50% 98.4%, exp38 20% 98.1% (hash:706176d) — no overfit,
  less generalization, small drop.
- setting exp30: exp39 50% 90.2%, exp40 20% 77.5% — 20% generalizes poorly.
- setting exp45: exp65 50% 91.4% (slight overfit), exp66 20% 89.4%.
- exp101: 10% train (392 tr / 3973 val), MAE -> tr 98.9 / val 83.2 (hash:13b3a7e).
- exp102: 20% (873 tr / 3492 val), MAE -> tr 94.5 / val 88.8.
- exp103/exp104: first-10%/second-10% (436/436) and ~1/40 data (100/100), to
  compare with finetune f9/f10-f11.

Takeaway: with less data, larger performance drop (95.5 -> 89.4 going 100% -> 20%).

## Model family

Width (setting exp45, width 30 = 6,340 params):

- exp67 width 96 (58,282) -> 93.7%, clear overfit.
- exp68 width 96 + harder dropout (l1 .25, l2–3 .5) -> 93.7%; dropout helped
  overfit a little, result no better.

Depth (setting exp45):

- exp70 4 layers (9,130) -> 94.0%, slightly worse + overfit (hash:15d7fd9).
- exp71 4 layers + residual conv1->conv4 (9,130) -> 94.8% (hash:9f49a11).

ResNet18 (setting exp45 + exp78 data): best exp79/exp82.

- exp79 no dropout -> 92.9% overfit (hash:253dffb, stopped early).
- exp80 dropout .7 -> 57.2% overfit; exp81 .5 -> 91.0%; exp82 .3 -> 92.22%;
  exp83 .3 + no downsample -> 91.4% (hash:a50bc2e).

Transformer / ImageBind (setting exp45 + exp78 data):

- exp72 ImageBind embed_dim 512 (3.17M) -> 84.8% overfit; exp84 512 -> 86.11%
  (hash:a940524), overfit but better than exp72.
- exp85 dim 16 -> 84.92%; exp86 dim 32 -> 83.59% (both underfit).
- exp87 dim 16, heads 8 -> 85.8% underfit.
- exp91 dim16/head8 dropout .7 (att&mlp) -> 78.82% (val > train, see lesson_learned).
- exp92 same, dropout .1 -> 86.93%, smaller train-val gap.

MAE encoder (setting exp45 + exp78 data, dim16/head8):

- exp88 avgpool + fc head -> 85.35% (hash:1940848), underfit (train 72%).
- exp89 cut-class-token + fc head -> 87.51% (hash:39aa268), very low train (51%).
- exp90 same + dropout .7 -> 71.54%; exp95 ImageBind head + dropout .7 -> 59.13%.
- **exp96** avgpool without class token -> 91.08% (train 94.55, hash:66d96fb), best MAE, no dropout.
- exp97 exp96 without layernorm before pool/classifier -> 89.47% (train 94.9), slightly lower.
- exp98 exp96 with random split -> val 94.7 / train 92.8, best overall.
- exp100 new dataloader: too slow (1 min -> 40 min), stopped.

## Optimizer / scheduler / lr / wd (setting exp45 unless noted)

- Optimizer: exp42 SGD 90% vs exp45 AdamW 95.4%.
- lr: exp48 lr=3e-4 step=1000 -> 92.8 (too small step); exp49 step=4000 -> 95.6;
  exp50 lr=1e-3 -> 94.2 (overfit @500); exp51 lr=1e-4 -> 95.4 (overfit @200);
  exp52 lr=6e-4 -> 94.8.
- wd: exp53 wd=3e-1 -> 95.4; exp54 wd=1e-3 -> 95.7. wd makes no real difference.
- Scheduler: exp47 CosineAnnealingWarmRestarts after 1000, min_lr 3e-5 -> 95.7;
  exp55 (accidental linear-warm + cosine) -> 93.4, no overfit; exp56 -> 93.1;
  exp57 cosine min_lr 1e-4 -> 95.1; exp58 cosine min_lr 2.5e-4 -> 95.4 (more overfit than step).

## Batch size (setting exp45, 3132/348 split)

- exp59 bs=1566 -> 94.8 (slightly more overfit); exp60 bs=783 -> 94.0;
  exp61 bs=391 -> 94.0; exp62 bs=97 -> 95.1; exp63 bs=25 -> 94.3;
  exp64 bs=6 -> 92.8 (very slow, stopped @2773).

## Supervised contrastive (small model)

Supervised contrastive loss slightly improves the t-SNE. Mean-entropy
maximization (a soft class balance) hurts: it lowers majority-class performance
and the t-SNE is worse.

- con1: class loss + supervised contrastive -> like exp125.
- con2: class + ¼·supcon -> like exp125.
- con3: class + supcon + 2·mean-entropy-max.
- con4: as con3 but weighted `(1-w)·2·me_max + w·(cls+supcon)`, w=0.35.

## Semi-supervised clustering

- 1discover_same_half_data: same split as clustering, 50/50 train/val.
- 1discover2: 90/10 train/val.

## MAE pretrain / finetune (small data, vs exp96/97 and exp104)

Pretrain: p1 all label data (emb/dec 16, head 8, drop .7, hash:f041d9f);
p2 80% data; p3 repeat p2 (lost to a crash); p4 = p3 run longer (12k iters,
lr step 6000). Last = p4.

Finetune (last = f11):

- f1 (p1, train class head only, frozen) -> tr 52 / val 48.6 (hash:2415ca9).
- f2 (f1, nothing frozen) -> 99 / 82.5; f3 (p2) -> 99 / 83.5.
- f4 (f3, no layernorm) -> 99 / 81.19 (norm slightly better).
- f5 (f4, head only) -> 57 / 52.75.
- f8 (repeat the two-head f7) -> 99.77 / 83.26; f9 (p3 all) -> 97.4 / 84;
  f10 (p3 all, vs exp104); f11 (p4 all, vs exp104).
- f6/f7 had a spurious second head layer of unknown origin — likely a mistake.

## App testing

- exp191 -> repeat exp45; exp192 -> repeat exp125. Accuracies in the
  "Compact models and app reruns" table.
- App showed a weird exp125 case (a clear Flap read as Manoeuvre at low
  confidence), not present for exp45; exp191/192 were the reruns to check it.
  This is the same short-receptive-field confusion that motivated moving to
  BirdModelSmallDilated for exp194/exp195.

## Early sanity / train-valid debugging

Small tmp data (1x4x20): exp3–exp9 probed conv/BN/relu wiring; exp15 traced a
train/val divergence to BatchNorm, fixed with `track_running_stats=False`
(exp22/exp23).

- exp19: normalizing GPS by max jumped 25% -> 53%.
- exp24 (`[0,2,4,5]` balanced) -> 97.5; **exp25_2** (90/10, 4 class) -> best 97%
  (hash:ed7d9a389).
- exp27 (90/10, all data, all classes, no balance) -> 90%; exp28 + balance -> 87%.
- exp30 (10 class, per-conv BN) -> 88.9 (hash:850e071); exp34 + balance -> 89%.
- exp33 (`[0,2,4,5]`, per-conv BN) -> 97% (hash:850e071).
- Replication was flaky at very low batch size (exp29/exp31 failed to reproduce
  until reverting to ed7d9a389 in exp30). exp32 bs=2 overfit fast and was very slow.
