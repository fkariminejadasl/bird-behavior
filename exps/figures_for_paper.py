import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from behavior import utils as bu


def make_grid_custom(
    paths,
    save_path,
    cols: int,
    cell_size=(512, 512),  # (width, height) of each tile
    padding: int = 0,
    bg="white",
):
    paths = list(paths)
    imgs = [Image.open(p).convert("RGB") for p in paths]

    cw, ch = cell_size
    tiles = []
    for im in imgs:
        im = ImageOps.contain(im, (cw, ch))  # keep aspect ratio
        im = ImageOps.pad(im, (cw, ch), color=bg)  # pad to exact cell size
        tiles.append(im)

    rows = math.ceil(len(tiles) / cols)
    grid_w = cols * cw + (cols - 1) * padding
    grid_h = rows * ch + (rows - 1) * padding
    canvas = Image.new("RGB", (grid_w, grid_h), bg)

    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        x = c * (cw + padding)
        y = r * (ch + padding)
        canvas.paste(tile, (x, y))

    canvas.save(save_path / "tsne_d0.png", dpi=(300, 300))
    canvas.save(save_path / "tsne_d0.pdf")  # convenient for LaTeX includegraphics

    return canvas


def make_grid_torchvision(
    paths, save_path, cols, cell_size=(512, 512), padding=0, bg="white"
):
    W, H = cell_size

    def to_cell(im):
        im = ImageOps.contain(im, (W, H))
        im = ImageOps.pad(im, (W, H), color=bg)
        return transforms.ToTensor()(im)

    imgs = [to_cell(Image.open(p).convert("RGB")) for p in paths]
    batch = torch.stack(imgs)
    grid = make_grid(batch, nrow=cols, padding=padding, pad_value=1.0)
    save_image(grid, save_path / "tsne_d0_tv.png")


# Example: all PNGs in a folder, sorted by name
fig_path = Path("/home/fatemeh/Downloads/bird/results/paper/tsne_d0")
save_path = Path("/home/fatemeh/Downloads/bird/results/paper")
paths = sorted(fig_path.glob("*.png"))
make_grid_custom(paths, save_path, cols=2, cell_size=(512, 512), padding=0, bg="white")
make_grid_torchvision(paths, save_path, cols=2, cell_size=(512, 512), padding=0)

exp_exclude = dict()
all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]
pairs = list(itertools.combinations(all_labels, 1))
for i, exclude in enumerate(pairs):
    exp = 135 + i
    exp_exclude[exp] = exclude[0]
    print(f"Experiment {exp}: Excluding label {exclude}")

# pairs = list(itertools.combinations(all_labels, 2))
# for i, exclude in enumerate(pairs):
#     exp = 144 + i
#     exp_exclude[exp] = exclude
#     print(f"Experiment {exp}: Excluding label {exclude}")

all_labels = [0, 2, 4, 5, 6]
pairs = list(itertools.combinations(all_labels, 1))
for i, exclude in enumerate(pairs):
    exp = 180 + i
    exp_exclude[exp] = exclude[0]
    print(f"Experiment {exp}: Excluding label {exclude}")

all_labels = [0, 2, 4, 5, 6]
pairs = list(itertools.combinations(all_labels, 1))
for i, exclude in enumerate(pairs):
    exp = 185 + i
    exp_exclude[exp] = exclude[0]
    print(f"Experiment {exp}: Excluding label {exclude}")


# exp_exclude = {135: 0, 136: 1, 137: 2, 138: 3, 139: 4, 140: 5, 141: 6, 142: 8, 143: 9, 180: 0, 181: 2, 182: 4, 183: 5, 184: 6, 185: 0, 186: 2, 187: 4, 188: 5, 189: 6}
# bu.ind2name = {0: 'Flap', 1: 'ExFlap', 2: 'Soar', 3: 'Boat', 4: 'Float', 5: 'SitStand', 6: 'TerLoco', 7: 'Other', 8: 'Manouvre', 9: 'Pecking'}

a = {
    135: (0.745, 0.958, 0.455, 643),
    136: (0.053, 0.526, -0.009, 38),
    137: (0.279, 0.164, -0.263, 537),
    138: (0.727, 0.716, 0.155, 176),
    139: (0.597, 0.646, 0.084, 729),
    140: (0.74, 0.747, 0.188, 1502),
    141: (0.323, 0.932, 0.131, 337),
    142: (0.589, 0.57, 0.029, 151),
    143: (0.4, 0.329, -0.132, 225),
}
a = {
    180: (0.969, 0.966, 0.556, 643),
    181: (0.41, 0.236, -0.282, 537),
    182: (0.524, 0.521, -0.044, 729),
    183: (0.737, 0.717, 0.121, 1502),
    184: (0.691, 0.682, 0.112, 337),
}
a = {
    185: (0.955, 0.961, 0.62, 337),
    186: (0.454, 0.338, -0.198, 337),
    187: (0.585, 0.579, 0.034, 337),
    188: (0.786, 0.757, 0.215, 337),
    189: (0.668, 0.62, 0.073, 337),
}

score_all_labels = {bu.ind2name[exp_exclude[k]]: v for k, v in a.items()}
labels = list(score_all_labels.keys())
sizes = np.array([v[-1] for v in score_all_labels.values()])
sep_scores = np.array([v[2] for v in score_all_labels.values()])
accuracies = np.array([v[0] for v in score_all_labels.values()])
accepted_inds = np.where(accuracies >= 0.4)[0]
rejected_inds = np.where(accuracies < 0.4)[0]
names = [f"{l}:{a}" for l, a in zip(labels, accuracies)]
plt.figure()
plt.plot(sizes[accepted_inds], sep_scores[accepted_inds], "g*")
plt.plot(sizes[rejected_inds], sep_scores[rejected_inds], "r*")
for x, y, name in zip(sizes, sep_scores, names):
    plt.text(x + 1, y, name, fontsize=8)
plt.show()
