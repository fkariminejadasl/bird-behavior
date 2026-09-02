---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 22px; }
---


# Bird behaviour, updates

What changed since the last meeting.

The story itself is in `presentation.md`.

---

## 2026-09-02

**Done**

- Data: find incorrect label (speed)
- Eval:
    - Labeled
    - Unlabeled: sea/land, temporal, speed, rotation
    - Visual (via app)
- Experiments
    - Large receptive field for flap (194)
    - Rotation invariance (agumentation) (195, 196 faster)
    - Extra features as input [x,y,z,gps,mag, dyn_mag, jerk_mag] (197)

**Next**

- Eval on *boat

---
