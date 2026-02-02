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

seed = 32984
train_per = 0.9
data_per = 1
exp = 181
labels_to_use = [0, 4, 5, 6]
in_channel, width, n_classes = 4, 30, len(labels_to_use)
save_path = Path("/home/fatemeh/Downloads/bird/results/")
data_file = Path("/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv")

save_name = f"{exp}"
fail_path = save_path / f"failed/{save_name}"
fail_path.mkdir(parents=True, exist_ok=True)

label_names = [bu.ind2name[i] for i in labels_to_use]

bu.set_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.CrossEntropyLoss()

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

print(device)
print(sum([p.numel() for p in model.parameters()]))

igs, ldts = bd.load_csv_pandas(data_file, labels_to_use, glen=20)

# igs, ldts = bd.load_csv(data_file)
# igs, ldts = bd.get_specific_labesl(igs, ldts, labels_to_use)

dataset = bd.BirdDataset(igs, ldts)

# dataset = bd.BirdDataset(igs, ldts)
loader = DataLoader(
    dataset,
    batch_size=len(dataset),
    shuffle=False,
    num_workers=1,
    drop_last=False,
)
data, ldts = next(iter(loader))
probs, preds, labels, loss, accuracy = bu.evaluate(data, ldts, model, criterion, device)
bu.save_confusion_matrix_other_stats(
    probs,
    preds,
    labels,
    loss,
    accuracy,
    fail_path,
    label_names,
    n_classes,
    stage="all",
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
