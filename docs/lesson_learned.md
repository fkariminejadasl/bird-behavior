# Lessons learned

Curated lessons from the bird-behavior classification experiments. Terser,
per-run notes live in [docs/experiment log](experiments_log.md); the
data/model/script overview is in [docs/description](description.md).

## Batched augmentation on the GPU

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
  (background)" below.
- Batched transforms take channel-first `(N, C, T)`; the per-sample ones take
  `(T, C)`. They are not interchangeable, and the other scripts
  (`train.py`, `train_sup_contrastive.py`, `train_self_distill.py`, `simGCD.py`)
  still use the per-sample ones.

## Rotation augmentation (exp195)

Full SO(3) rotation of the IMU acc channels, train set only (`RandomRotation3D`).
The labeled data is one logger type mounted much the same way on every bird, so
the standard split holds no orientation variation at all. Full 3D is the realistic
model for a new logger, whose tilt and position on the bird are both unknown.

- **A standard-split A/B only measures the cost.** exp195 92.24 val vs exp194
  96.35 (-4.1) reads as "augmentation hurt". Re-evaluating both checkpoints on
  orientation-perturbed validation data inverts it
  (`exps/eval_rotation_robustness.py`):

  | valid under | exp194 (no aug) | exp195 (rotation) |
  |---|---|---|
  | clean | **96.35** | 92.24 |
  | x/y swap | 71.46 | **90.41** |
  | pitch 20° / 70° | 82.88 / 2.05 | **90.87 / 90.64** |
  | random SO(3), mean of 20 | 13.05 | **90.67** |

  Off-orientation exp194 is not uncertain, it is confidently wrong: 2.05% is well
  under the 11% chance rate, so no confidence threshold will catch it.
- **The cost lands on the ground behaviours**, which need the gravity direction
  rotation scrambles: valid F1 TerLoco .97 -> .79, Pecking .94 -> .68. Balanced
  valid F1 0.92 -> 0.84, twice the accuracy drop
  (`exps/compare_per_class_metrics.py`). Plain and balanced files can disagree in
  sign on a rare class, so say which one a number came from.
- **Rotation replaces overfitting with underfitting**: exp194 99.44/96.35 vs
  exp195 90.21 (unaugmented train) / 92.24. At 5,129 params the capacity goes into
  invariance, not memorization.
- **A bigger receptive field did not pay for it**: BirdModel -> SmallDilated is
  +1.6 val without rotation (exp192 94.75 -> exp194 96.35) but +0.7 with
  (discarded trial 91.55 -> exp195 92.24).

## Training recipe

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

## Model size

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

## Data

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
- **Clip IMU to [-2, 2]** (exp190): only 782 of 86,760 rows were outside, so the
  effect is small, but it bounds outliers.
- **Less data, larger drop**: 100% -> 20% of the data costs ~6 points
  (95.5 -> 89.4). More data past exp45 (exp78, +25%) did not help, so the
  dataset is near saturation for this model.
- **The data is noisy**: different data slices and seeds shift validation by
  ~2.5 points for reasons never fully explained (exp98, exp109–112). Report a
  fixed seed (32984) and be wary of reading small val differences as real.
- **Peck labels are noisy**; treat Pecking results with caution.

## Rotation augmentation (background)

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

## Contrastive and clustering

- **Supervised contrastive loss** slightly improves the t-SNE structure and
  tracks the plain classifier (con1/con2 ≈ exp125).
- **Mean-entropy maximization hurts.** As a soft class-balancer it sacrifices
  majority-class accuracy and worsens the t-SNE (con3/con4). Class balancing in
  general (weights or resampling) trades overall accuracy for rare-class recall
  (exp28, exp34, exp127).

## MAE pretrain / finetune

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

## Reproducibility / infrastructure bugs

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
