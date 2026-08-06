---
marp: true
theme: default
paginate: true
size: 16:9
---

<!-- Slides. See README.md for where the figures come from and how to remake
them. Render with Marp: see "How to present this" at the end of README.md.
exp194 is the no-augmentation model. exp195 and exp196 are the same rotation
config, exp196 being the rerun through the faster wiring, so the orientation
table quotes exp195 (92.2) and the unlabelled comparison exp196 (92.7); the
difference is run-to-run noise. All BirdModelSmallDilated on starts.csv, seed
32984. Keep image height at 400; taller pushes the caption off the slide. -->

# Reading gull behaviour from a tag

Nine behaviours from one second of accelerometer and GPS

A model small enough to run anywhere: **5,129 parameters**

---

## The result in one sentence

# One model. One set of held-out bursts. **96% correct.**

# Turn the tag to a different mounting angle and feed it the very same bursts: **13%.**

Nothing about the bird or the behaviour changed — only which way the tag faces.
That fragility, not the 96%, is the finding.

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
- **5,129 parameters.** Smaller than the plain version it replaced, and better
- Context beat capacity: a wing beat is only visible if you see the whole second

---

## The catch: the tag records more than the behaviour

The accelerometer measures in the **tag's own frame**, not the bird's. Every
reading mixes:

- **how the logger is tilted**, and where it sits — neck, back, wing
- **the environment** — wind, water impact, air resistance, a collision
- **gravity**, always present in the raw signal
- and, somewhere in there, what the bird is actually doing

Only the last one is the label. Our data is **one logger type, mounted much the
same way on every bird**, so the standard test split cannot see any of this.

---

## What happens on a differently mounted tag

| test data | no augmentation | with rotation |
|---|---|---|
| normal | **96.4** | 92.2 |
| x/y axes swapped *(Ornitela convention)* | 71.5 | **90.4** |
| tilted 70° | 2.1 | **90.6** |
| any random orientation | 13.1 | **90.7** |

Not merely wrong — **confidently** wrong. 2% is far below the 11% you would get
by guessing.

---

## The fix, and what it costs

Train on the same bursts **rotated to every possible orientation**
*(random SO(3) rotation of the acceleration axes)*.

- Costs **4 points** on the birds we already handle: 96.4 → 92.2
- Holds **~91% at any mounting orientation**, instead of collapsing
- The cost lands on the ground behaviours — walking and pecking — which are
  told apart by the *direction* of small movements relative to gravity, exactly
  what rotating destroys

Worth it whenever the model will meet a tag it was not trained on.

---

## How do we know it works on a new bird?

We have millions of unlabelled bursts and no answers for any of them.

**The obvious idea: trust the model's confidence when it is unsure.**

It does not work.

---

## Confidence does not notice the model failing

On the tilted data where accuracy fell from 96% to 13%:

- confidence fell only from **0.96 to 0.74**
- 0.74 looks like a perfectly healthy model

On 115,266 unlabelled bursts from a new bird, confidence rates the two models
**0.89 vs 0.91** — indistinguishable, while one makes 16x more impossible
predictions.

Confidence is useful for **ranking** which bursts to look at, never for asking
whether the model is healthy.

---

## Instead: count predictions that cannot be true

No labels needed. Each check contradicts the model with something we know
independently.

- **Impossible speed** — "sitting still" at 13 m/s
- **Impossible place** — "walking" a kilometre out to sea *(land/sea mask)*
- **Impossible flicker** — behaviour changing every second within one recording
- **Unstable** — a different answer when the tag is rotated

For flicker, "impossible" has to be learned: gulls really do walk, peck and walk
again. From the labelled data, only **10 of 36** behaviour pairs ever follow one
another *(label co-occurrence within a fix)*. Everything else counts.

---

## The two models on a new bird, no labels

115,266 bursts, device 6004

| | confidence | impossible speed | impossible place | impossible flicker | unstable |
|---|---|---|---|---|---|
| no augmentation | 0.89 | 1.45% | **1.18%** | 6.7% | **82.8%** |
| with rotation | 0.91 | **0.09%** | 1.45% | **4.9%** | **5.4%** |

**16x fewer impossible predictions, and it stops changing its mind.**

Confidence, the one number you would have reached for, says they are the same.

Place is slightly worse — it predicts more walking, a land class. Reported, not
hidden.

---

## A side benefit: finding bad labels

The same speed rule, pointed at the *labels* instead of the predictions.

- **26 of 4,338 bursts** contradict their own label
- Clearest: a bird labelled "sitting still" moving at **13.7 m/s**
- 16 of the 26 are one device in one 20-minute window — that is a **broken GPS**,
  not 16 annotation mistakes

Kept separate, because the fix is different.

---

## Making it fast enough to iterate

The augmentation was applied one burst at a time, in Python.

- 47 minutes per run → **5 minutes**
- Same result: 92.7 vs 92.2, well inside run-to-run noise
- The trick: rotate the whole batch at once on the GPU, and stop rebuilding the
  data every epoch

Nine times more experiments per day, for no change in the science.

---

## What we learned

1. **A high score on your own test split can hide a total failure.** Ours could
   not see mounting orientation at all
2. **Confidence is not a health check.** A model can be certain and wrong
3. **You can measure a model without labels** — count predictions that break
   physics
4. **Robustness costs accuracy.** 4 points bought a model that survives a tag
   change
5. **Check the labels too.** Some of what we call error is bad ground truth

---

## What is next

1. **Test on a real second logger**, not a synthetic rotation of the first
2. **Recover the ground behaviours** — walking and pecking pay for the rotation
3. **Automate the sea/land and flicker checks** on every new deployment
4. **Label ~200 bursts from a new bird**, the only way to get a real number
5. **The other sources of variation**: where the tag sits, wind, water impact

---

## Backup: numbers

- 4,338 labelled bursts, 9 classes, 20 Hz, 90/10 split, seed 32984
- exp194 no aug: 96.35 valid / 99.44 train — memorises, overfits
- exp196 rotation: 92.69 valid / 91.56 train — underfits, never memorises
- Per-class cost of rotation: walking .97 → .79, pecking .94 → .68
- Class-balanced F1 0.92 → 0.84, twice the drop of plain accuracy
- Rotation robustness: 13.1% → 90.7% mean over 20 random orientations
- Unlabelled agreement between the two models: 87.5%
- Label-noise check: 26 bursts dropped, `starts_clean.csv` 4,312 bursts
