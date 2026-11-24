"""Train one model on gulls and spoonbills: a coarse head, a fine head per species.

`behavior/model.py::BirdModelSpeciesHeads`: the trunk of
`BirdModelSmallDilated`, a species token, and three linear heads on the same
features:

- coarse: 4 groups shared by both species (`COARSE`). The one place where the
  two species must give the same answer, so it pushes the shared features to
  line up.
- gull: the 9 gull classes. spoonbill: the spoonbill classes. A fine
  prediction is always a class of the segment's own species.

The species token is two constant input channels, is-gull and is-spoonbill.
In training it is dropped, both 0, for a share `token_drop` of the segments,
so the model also works without knowing the species: the coarse head can then
serve a new species. The fine head is chosen by the true species either way.

Loss: cross-entropy of the coarse head plus cross-entropy of the segment's own
fine head, each a mean over the batch.

The coarse groups follow the cross-species tests (`docs/status.md`, Across
species and Both species in one model): walking and feeding are one group,
because gull Pecking and spoonbill for-search mix in every test. drink is left
out (7 segments, none in valid), after the split, so both valid sets stay
those of the single-species models: 438 gull bursts as exp197, 505 spoonbill
segments as exp206-211. Before training, the best checkpoints of exp197 and
exp207 must reproduce their valid accuracy on the rebuilt split
(`scripts/train_joint.py::load_species`).

The best checkpoint is the epoch with the highest mean fine valid accuracy of
the two species, token given. Writes, under `save_path`: `<exp>_best.pth`,
`<exp>_<no_epochs>.pth`, `tensorboard/<exp>/`, and in `failed/<exp>_heads/`:
`app_loss_acc.txt` (coarse and fine accuracy per species, split, checkpoint
and token, given or dropped), and per checkpoint, split and token a confusion
matrix, `counts_<ckpt>_<split>_<token>.png`, with its counts in a CSV beside
it. Compare with the single-species models in `exps/eval_species_heads.py`.

  /home/fatemeh/miniconda3/envs/bird/bin/python scripts/train_species_heads.py \\
    > /home/fatemeh/Downloads/bird/results/tensorboard/<exp>.txt 2>&1
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import train_joint as tj
from omegaconf import OmegaConf
from torch.utils import tensorboard

from behavior import data_augmentation as bau
from behavior import model as bm
from behavior import utils as bu

# Coarse groups, each label of both species in exactly one. drink: none.
COARSE = {
    "fly": ["Flap", "ExFlap", "Manouvre", "Soar", "fly-flap", "fly-soar"],
    "water": ["Float", "Boat"],
    "land inactive": [
        "SitStand",
        "stand-rest",
        "sit-alert",
        "stand-alert",
        "stand-preen",
        "stand-other",
        "stand-shake-feathers",
    ],
    "land active": [
        "TerLoco",
        "Pecking",
        "walk",
        "for-search",
        "for-handle",
        "for-intake",
        "beg",
    ],
}


def fine_classes(cfg, species):
    """Fine classes of one species, in label order, the dropped labels left out."""
    labels = cfg.species[species].labels
    return [labels[c] for c in sorted(labels) if labels[c] not in cfg.drop]


def load(cfg, device):
    """Per species and split: x, species index, fine and coarse targets, names.

    The split is that of `train_joint.load_species`, on all labels; the
    dropped labels are removed after it.
    """
    split_cfg = OmegaConf.create(
        {
            **OmegaConf.to_container(cfg),
            "model": OmegaConf.to_container(cfg.check_model),
            "joint_classes": tj.SEPARATE_CLASSES,
        }
    )
    data = tj.load_species(split_cfg, device)
    groups = list(cfg.coarse)
    to_group = {n: groups.index(g) for g, names in cfg.coarse.items() for n in names}
    out = {}
    for i, (species, d) in enumerate(data.items()):
        fine = fine_classes(cfg, species)
        assert all(n in to_group for n in fine), f"{species}: label without a group"
        out[species] = {}
        for split, part in d.items():
            keep = ~np.isin(part["names"], list(cfg.drop))
            names = part["names"][keep]
            n = len(names)
            out[species][split] = {
                "x": part["x"][torch.as_tensor(keep, device=device)],
                "names": names,
                "species": torch.full((n,), i, device=device),
                "fine": torch.tensor([fine.index(x) for x in names], device=device),
                "coarse": torch.tensor([to_group[x] for x in names], device=device),
            }
            dropped = int((~keep).sum())
            if dropped:
                print(
                    f"{species} {split}: {dropped} segments of {list(cfg.drop)} left out"
                )
    return out


def concat(parts):
    """One batch of several species' parts."""
    keys = ["x", "species", "fine", "coarse"]
    return {k: torch.cat([p[k] for p in parts]) for k in keys}


def loss_terms(out, batch, species_names):
    """Coarse and fine cross-entropy, each a mean over the batch."""
    coarse = F.cross_entropy(out["coarse"], batch["coarse"])
    fine = 0.0
    for i, name in enumerate(species_names):
        m = batch["species"] == i
        if m.any():
            fine = fine + F.cross_entropy(
                out[name][m], batch["fine"][m], reduction="sum"
            )
    return coarse, fine / len(batch["fine"])


@torch.no_grad()
def predict(model, part, species, token):
    """Coarse and fine predictions of one species' part, the fine from its head.

    token: "given" uses the species, "dropped" passes -1, unknown species.
    """
    model.eval()
    s = part["species"] if token == "given" else torch.full_like(part["species"], -1)
    out = model(part["x"], s)
    return out["coarse"].argmax(1), out[species].argmax(1)


def accuracies(model, part, species, token):
    """Coarse and fine accuracy in %."""
    coarse, fine = predict(model, part, species, token)
    return (
        100 * (coarse == part["coarse"]).float().mean().item(),
        100 * (fine == part["fine"]).float().mean().item(),
    )


def build_model(cfg, device):
    heads = {"coarse": len(cfg.coarse)}
    heads.update({s: len(fine_classes(cfg, s)) for s in cfg.species})
    return bm.BirdModelSpeciesHeads(heads=heads, **cfg.model.parameters).to(device)


def refuse_to_overwrite(cfg):
    """Stop if this experiment number already has results."""
    if cfg.overwrite:
        return
    save = Path(cfg.save_path)
    taken = [
        p
        for p in (
            save / f"{cfg.exp}_best.pth",
            save / "failed" / f"{cfg.exp}_heads",
            save / "tensorboard" / str(cfg.exp),
        )
        if p.exists()
    ]
    if taken:
        raise FileExistsError(f"exp {cfg.exp} already has results: {taken}")


def train(cfg, data, device):
    """Train; save the best and the last epoch. Returns the best epoch."""
    bu.set_seed(cfg.seed)
    save = Path(cfg.save_path)
    species_names = list(data)
    tr = concat([d["train"] for d in data.values()])
    n = len(tr["fine"])
    batch_size = n if cfg.batch_size is None else cfg.batch_size
    transform = bau.BatchRandomRotation3D() if cfg.augment else None

    model = build_model(cfg, device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.step_size, gamma=0.1
    )
    print(
        f"BirdModelSpeciesHeads, {params:,} parameters, rotation {cfg.augment}, "
        f"token dropped for {cfg.token_drop:.0%}, {n} train segments, start "
        f"{datetime.now()}"
    )

    best, best_epoch = -1.0, 0
    with tensorboard.SummaryWriter(save / f"tensorboard/{cfg.exp}") as writer:
        for epoch in range(1, cfg.no_epochs + 1):
            model.train()
            order = torch.randperm(n, device=device)
            sums = {"coarse": 0.0, "fine": 0.0}
            for start in range(0, n, batch_size):
                sel = order[start : start + batch_size]
                batch = {k: v[sel] for k, v in tr.items()}
                x = batch["x"] if transform is None else transform(batch["x"])
                token = batch["species"].clone()
                token[torch.rand(len(sel), device=device) < cfg.token_drop] = -1
                out = model(x, token)
                coarse, fine = loss_terms(out, batch, species_names)
                loss = cfg.coarse_weight * coarse + fine
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sums["coarse"] += coarse.item() * len(sel)
                sums["fine"] += fine.item() * len(sel)
            scheduler.step()
            for k, v in sums.items():
                writer.add_scalar(f"loss/train_{k}", v / n, epoch)
            fines = []
            for name, d in data.items():
                for token in ("given", "dropped"):
                    c, f = accuracies(model, d["valid"], name, token)
                    writer.add_scalar(f"coarse/valid_{name}_{token}", c, epoch)
                    writer.add_scalar(f"fine/valid_{name}_{token}", f, epoch)
                    if token == "given":
                        fines.append(f)
            mean = float(np.mean(fines))
            if mean > best:
                best, best_epoch = mean, epoch
                bm.save_model(save, cfg.exp, epoch, model, optimizer, scheduler, True)
            if epoch % cfg.print_every == 0 or epoch == 1:
                valid = ", ".join(f"{k} {a:.2f}" for k, a in zip(data, fines))
                print(
                    f"epoch {epoch}: train loss coarse {sums['coarse'] / n:.4f}, "
                    f"fine {sums['fine'] / n:.4f}; valid fine, token given: "
                    f"{valid}; best mean {best:.2f} at {best_epoch}"
                )
    bm.save_model(save, cfg.exp, cfg.no_epochs, model, optimizer, scheduler)
    print(f"end {datetime.now()}, best epoch {best_epoch}")
    return best_epoch


def by_group(cfg, labels):
    """`labels` ordered by coarse group, then as given."""
    groups = list(cfg.coarse)
    to_group = {n: groups.index(g) for g, names in cfg.coarse.items() for n in names}
    return sorted(labels, key=lambda n: (to_group[n], labels.index(n)))


def count_tables(cfg, model, data, split, token):
    """Per species: segments per label and coarse group, and per label and class.

    Labels and classes are ordered by coarse group (`by_group`).
    """
    groups = list(cfg.coarse)
    tables = {}
    for species, d in data.items():
        part = d[split]
        coarse, fine = predict(model, part, species, token)
        classes = np.array(fine_classes(cfg, species))  # in the head's order
        rows = by_group(cfg, list(classes))
        tables[species, "coarse"] = pd.crosstab(
            pd.Categorical(part["names"], rows),
            pd.Categorical(np.array(groups)[coarse.cpu().numpy()], groups),
            dropna=False,
        )
        tables[species, "fine"] = pd.crosstab(
            pd.Categorical(part["names"], rows),
            pd.Categorical(classes[fine.cpu().numpy()], rows),
            dropna=False,
        )
    return tables


def plot_tables(cfg, tables, title, path):
    """A 2 x 2 figure: per species, labels by coarse group and labels by class.

    The orange box is the right answer: the label's group, or the label.
    """
    species = list(cfg.species)
    sizes = [len(tables[s, "fine"]) for s in species]
    widths = [len(cfg.coarse), max(sizes)]
    fig, axes = plt.subplots(
        len(species),
        2,
        figsize=(0.5 * sum(widths) + 6, 0.36 * sum(sizes) + 4),
        height_ratios=sizes,
        width_ratios=widths,
        layout="constrained",
    )
    to_group = {n: g for g, names in cfg.coarse.items() for n in names}
    for row, s in enumerate(species):
        for col, level in enumerate(("coarse", "fine")):
            t = tables[s, level]
            labels, cols = list(t.index), list(t.columns)
            if level == "coarse":
                boxes = [(r, cols.index(to_group[lab])) for r, lab in enumerate(labels)]
            else:
                boxes = [(r, r) for r in range(len(labels))]
            ax = axes[row, col]
            im = tj.draw_counts(ax, t.values, labels, cols, boxes)
            right = sum(t.values[r, c] for r, c in boxes)
            total = t.values.sum()
            ax.set_title(
                f"{s}, {level}: {100 * right / max(total, 1):.2f}% of {total}",
                loc="left",
                fontsize=10,
            )
            ax.set_ylabel("true label (segments)" if col == 0 else "")
            ax.set_xlabel(
                f"predicted {'coarse group' if level == 'coarse' else 'class'}"
            )
    fig.colorbar(im, ax=axes, shrink=0.4, label="share of the row's segments, %")
    fig.suptitle(f"{title}\nnumber: segments; orange box: the right answer")
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def report(cfg, data, device, best_epoch):
    """Accuracies, counts and figures to `failed/<exp>_heads/`."""
    save = Path(cfg.save_path)
    out = save / "failed" / f"{cfg.exp}_heads"
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg, device)
    lines = []
    ckpts = {"best": f"{cfg.exp}_best.pth", "final": f"{cfg.exp}_{cfg.no_epochs}.pth"}
    for ckpt, file in ckpts.items():
        bm.load_model(save / file, model, device)
        epoch = best_epoch if ckpt == "best" else cfg.no_epochs
        for split in ("train", "valid"):
            for token in ("given", "dropped"):
                for name, d in data.items():
                    c, f = accuracies(model, d[split], name, token)
                    lines.append(
                        f"{ckpt} (epoch {epoch}), {name} {split}, token {token}: "
                        f"coarse accuracy {c:.2f}, fine accuracy {f:.2f}, "
                        f"segments {len(d[split]['names'])}"
                    )
                tables = count_tables(cfg, model, data, split, token)
                stem = f"counts_{ckpt}_{split}_{token}"
                pd.concat(
                    {f"{s} {lv}": t.stack() for (s, lv), t in tables.items()}
                ).rename("segments").to_csv(out / f"{stem}.csv")
                title = (
                    f"exp{cfg.exp}, {split} split, {ckpt} checkpoint (epoch "
                    f"{epoch}), species token {token}"
                )
                plot_tables(cfg, tables, title, out / f"{stem}.png")
    (out / "app_loss_acc.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main(cfg):
    refuse_to_overwrite(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = load(cfg, device)
    best_epoch = train(cfg, data, device)
    report(cfg, data, device, best_epoch)


if __name__ == "__main__":
    base_config = {
        "save_path": "/home/fatemeh/Downloads/bird/results",
        "exp": 216,
        "overwrite": False,  # True to replace the results of an experiment number
        "seed": 32984,
        "split_seed": 32984,  # both species keep the split of their own models
        "train_per": 0.9,
        "glen": 20,
        # data and split per species; to_joint is not used here
        "species": tj.species_config(tj.GULL_TO_SEPARATE, tj.SPOONBILL_TO_SEPARATE),
        "coarse": COARSE,
        "drop": ["drink"],  # labels left out, after the split
        "add_magnitudes": True,  # 7 channels, as exp197 and exp206-211
        "augment": True,  # random 3D rotation of the acceleration
        "token_drop": 0.5,  # share of training segments with the species unknown
        "coarse_weight": 1.0,  # loss = coarse_weight * coarse + fine
        # Training, as exp197: full batch, AdamW, StepLR
        "no_epochs": 4000,
        "batch_size": None,  # None is the whole train set in one batch
        "max_lr": 3e-4,
        "weight_decay": 1e-2,
        "step_size": 2000,
        "print_every": 100,
        "model": {
            "name": "BirdModelSpeciesHeads",
            "parameters": {
                "in_channels": 7,
                "n_species": 2,
                "mid_channels": 20,
                "dropout": 0.15,
            },
        },
        # the single-species model whose split check load() runs
        "check_model": {
            "name": "BirdModelSmallDilated",
            "parameters": {"in_channels": 7, "mid_channels": 20, "dropout": 0.15},
        },
    }
    # Rotation: exp216, 218, 220; no rotation: exp217, 219, 221; seeds 32984,
    # 1234, 5678. split_seed stays 32984, so all six share one valid split.
    experiments = [
        {"exp": 216},
        {"exp": 217, "augment": False},
        {"exp": 218, "seed": 1234},
        {"exp": 219, "seed": 1234, "augment": False},
        {"exp": 220, "seed": 5678},
        {"exp": 221, "seed": 5678, "augment": False},
    ]

    configs = [
        OmegaConf.merge(OmegaConf.create(deepcopy(base_config)), e) for e in experiments
    ]
    for cfg in configs:  # every number free before anything trains
        refuse_to_overwrite(cfg)
    for cfg in configs:
        print(f"\n# Experiment {cfg.exp}")
        main(cfg)
