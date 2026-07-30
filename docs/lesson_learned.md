# Lessons learned

Curated lessons from the bird-behavior classification experiments. Terser,
per-run notes live in [docs/experiment log](experiments_log.md); the
data/model/script overview is in [docs/description](description.md).

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
  augmentation (exp130) beats warping it but still trails. **The current best on
  `starts.csv` is exp194 (BirdModelSmallDilated, no augmentation, 96.35 val) —
  that, not exp125, is the number the rotation augmentation must beat.**
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

## Rotation augmentation (current work)

- The accelerometer is measured in the **tag body frame**, whose orientation
  relative to the bird is not fixed. UvA-BiTS tags are mounted at different
  positions and angles (anywhere from horizontal to ~70° pitch; see the
  "Accelerometer calibration" section of the UvA-BiTS wiki,
  https://wiki.e-ecology.nl/index.php/UvA-BiTS_Tracking_Data), and Ornitela
  mounts x and y swapped relative to UvA-BiTS (the `fix_ornitella_axis_switch`
  branch and the `# In Ornitela devices x and y switched` note in
  `scripts/data/bird_behavior_app_data.py`). A model trained on one convention
  need not transfer to another.
- **Full random 3D rotation (SO(3)) of the three IMU acceleration channels**
  (`behavior.data_augmentation.RandomRotation3D`) makes the classifier invariant
  to that mounting orientation. It preserves the per-timestep acceleration
  magnitude and leaves the GPS speed channel untouched. A full rotation is a
  superset of the physical mounting variation (bounded pitch plus the x/y swap),
  so it covers the real cases as well as many that never occur.
- **It is an aggressive augmentation, and a short receptive field makes it
  worse.** A first BirdModel + rotation trial (discarded, val 91.55% — ~3.8 pts
  below the exp125 baseline it ran against, ~4.8 below exp194) did *not* lose
  accuracy where the a priori
  worry said it would. Scrambling the gravity direction left the
  orientation-dependent static classes intact (val F1 SitStand .97, Float .97,
  Boat .92): a static behavior has |acc|≈1 g with near-zero temporal variance
  whatever the orientation, so rotation-invariant statistics (magnitude,
  variance) still separate it. The loss instead concentrated in the rare dynamic
  classes (Pecking, Manoeuvre, ExFlap). That matches an independent observation
  on unlabeled data in `app/gps_burst_labeling_viz_app.py`: BirdModel's short
  receptive field (RF 7, only a third of the 20-sample burst) confuses the flap
  sine-wave with Manoeuvre, and rotation only adds to that confusion.
  **BirdModelSmallDilated has a larger receptive field (RF 25, spanning the full
  burst) and sees the global flap pattern, so it is the model to test rotation
  on** — exp195, still to be run, A/B against exp194's 96.35 val. If the accuracy
  cost still matters, a gravity-preserving yaw-only or bounded-pitch (~70°)
  rotation is the fallback; do not expect rotation to beat the no-aug baseline on
  the standard split.
- **Only augment the training set.** Both `prepare_train_valid_dataset` and the
  stratified-split path in `scripts/batch_train_supervised.py` pass one transform
  object to the train and eval `BirdDataset`s, which would rotate the evaluation
  data too. The rotation wiring in `batch_train_supervised.py` avoids this by
  giving the eval `BirdDataset` a `None` transform, and it clears
  `dataset.transform` before the final confusion-matrix evaluation (the train
  dataset is reused there and still holds the training transform).

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
