from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.utils import n_classes, target_labels, target_labels_names

# import wandb
# wandb.init(project="uncategorized")

seed = 32984
train_per = 0.9
data_per = 1
exp = 45  # sys.argv[1]
save_name = f"{exp}"
width = 30
save_path = Path("/home/fatemeh/Downloads/bird/result/")
data_file = Path(
    "/home/fatemeh/Downloads/bird/data/final/corrected_combined_unique_sorted012.csv"
)
fail_path = save_path / f"failed/{save_name}"
fail_path.mkdir(parents=True, exist_ok=True)


bu.set_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.CrossEntropyLoss()

in_channel = 4  # next(iter(train_loader))[0].shape[1]
model = bm.BirdModel(in_channel, width, n_classes).to(device)
# model = bm1.TransformerEncoderMAE(
#     img_size=20,
#     in_chans=4,
#     out_chans=9,
#     embed_dim=16,
#     depth=1,
#     num_heads=8,
#     mlp_ratio=4,
#     drop=0.0,
#     layer_norm_eps=1e-6,
# ).to(device)
bm.load_model(save_path / f"{exp}_best.pth", model, device)
model.eval()

"""
train_dataset, eval_dataset = bd.prepare_train_valid_dataset(
    data_file, train_per, data_per, target_labels
)
train_loader = DataLoader(
    train_dataset,
    batch_size=len(train_dataset),
    shuffle=True,
    num_workers=1,
    drop_last=True,
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=len(eval_dataset),
    shuffle=False,
    num_workers=1,
    drop_last=True,
)

data, ldts = next(iter(train_loader))
bu.helper_results(
    data,
    ldts,
    model,
    criterion,
    device,
    fail_path,
    target_labels_names,
    n_classes,
    stage="train",
    SAVE_FAILED=False,
)

data, ldts = next(iter(eval_loader))
bu.helper_results(
    data,
    ldts,
    model,
    criterion,
    device,
    fail_path,
    target_labels_names,
    n_classes,
    stage="valid",
    SAVE_FAILED=False,
)
"""

print(device)
print(sum([p.numel() for p in model.parameters()]))

csv_igs, csv_ldts = bd.load_csv(data_file)
csv_igs, csv_ldts = bd.get_specific_labesl(csv_igs, csv_ldts, target_labels)
dataset = bd.BirdDataset(csv_igs, csv_ldts)
loader = DataLoader(
    dataset,
    batch_size=len(dataset),
    shuffle=False,
    num_workers=1,
    drop_last=False,
)
data, ldts = next(iter(loader))
bu.helper_results(
    data,
    ldts,
    model,
    criterion,
    device,
    fail_path,
    target_labels_names,
    n_classes,
    stage="all",
    SAVE_FAILED=False,
)

# bad classes: Other, Exflap (less data), Pecking (noisy), Manuver/Mix

# for i in range(len(target_labels)):
#     inds = torch.where(labels == 3)[0]
#     sel_labels = labels[inds]
#     sel_prob = prob[inds]
#     average_precision_score(sel_labels.cpu().numpy(), sel_prob.cpu().numpy())
#     average_precision_score(    labels.cpu().numpy(),     prob.cpu().numpy())


# y_true = np.array([0, 0, 1, 1, 2, 2])
# y_scores = np.array([
#     [0.7, 0.2, 0.1],
#     [0.4, 0.3, 0.3],
#     [0.1, 0.8, 0.1],
#     [0.2, 0.3, 0.5],
#     [0.4, 0.4, 0.2],
#     [0.1, 0.2, 0.7],
# ])
# average_precision_score(y_true, y_scores)
