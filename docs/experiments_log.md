# Experiment log

Raw notebook of individual runs, terse style. Held to a lower bar than
[docs/lesson_learned.md](lesson_learned.md) (curated lessons) and
[docs/description.md](description.md) (data/model/script overview). 

Accuracies are validation unless noted; `tr-val` gives train then valid. A
trailing `hash:...` is the git commit the run was made at. Numbering has gaps
(runs that were abandoned or folded elsewhere). Newest entries first.

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
  `exps/eval_rotation_robustness.py`.
- Valid F1 TerLoco .97 -> .79 and Pecking .94 -> .68 carry the loss; balanced valid
  F1 0.92 -> 0.84 (`exps/compare_per_class_metrics.py`).

## Reference

Label map:

```python
ind2name = {0: "Flap", 1: "ExFlap", 2: "Soar", 3: "Boat", 4: "Float",
            5: "SitStand", 6: "TerLoco", 7: "Other", 8: "Manouvre", 9: "Pecking"}
```

`s_data` burst counts (index): `{0:634, 1:38, 2:501, 3:176, 4:558, 5:894,
6:318, 7:25, 8:151, 9:210}`. `starts` (current pipeline, no Other): total 4338
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

- exp44 `[0,3,4,5,6]`: 99.2% (hash:dac5a78) ***
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

| exp | model / data | train | valid |
|---|---|---|---|
| exp190 | BirdModel, starts.csv (IMU clipped to [-2, 2]) | 95.49 | 94.98 |
| exp191 | BirdModel, s_index (repeat exp45) | 94.92 | 94.32 |
| exp192 | BirdModel, starts.csv (repeat exp125) | 95.95 | 94.75 |
| exp193 | BirdModelWideRF, starts.csv | 99.67 | 95.21 |
| exp194 | BirdModelSmallDilated, starts.csv | 99.44 | **96.35** |

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
