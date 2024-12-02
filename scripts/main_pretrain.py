from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.utils import target_labels


def write_info_in_tensorboard(writer, epoch, loss, stage):
    loss_scalar_dict = dict()
    loss_scalar_dict[stage] = loss
    writer.add_scalars("loss", loss_scalar_dict, epoch)


def caculate_metrics(data, model, device):
    data = data.permute((0, 2, 1))  # NxCxL -> NxLxC
    data = data.to(device)
    loss, _, _ = model(data)  # NxC
    return loss


def train_one_epoch(loader, model, device, epoch, no_epochs, writer, optimizer):
    stage = "train"

    model.train()
    running_loss = 0
    for i, (data, _) in enumerate(loader):
        optimizer.zero_grad()

        loss = caculate_metrics(data, model, device)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    total_loss = running_loss / (i + 1)

    print(f"{stage}: epoch/total: {epoch}/{no_epochs}, total loss: {total_loss:.4f}")

    write_info_in_tensorboard(writer, epoch, total_loss, stage)


@torch.no_grad()
def evaluate(loader, model, device, epoch, no_epochs, writer):
    stage = "valid"

    model.eval()
    running_loss = 0
    for i, (data, _) in enumerate(loader):
        loss = caculate_metrics(data, model, device)

        running_loss += loss.item()

    total_loss = running_loss / (i + 1)
    print(f"{stage}: epoch/total: {epoch}/{no_epochs}, total loss: {total_loss:.4f}")
    write_info_in_tensorboard(writer, epoch, total_loss, stage)
    return total_loss


# import wandb
# wandb.init(project="uncategorized")

seed = 1234
bu.set_seed(seed)

save_path = Path("/home/fatemeh/Downloads/bird/result/")
exp = "p5"  # sys.argv[1]
no_epochs = 12000  # int(sys.argv[2])
save_every = 6000
train_per = 0.9
data_per = 1.0

# hyperparam
warmup_epochs = 1000
step_size = 6000
max_lr = 3e-4  # 1e-3
min_lr = max_lr / 10
weight_decay = 1e-2  # default 1e-2
# model
width = 30

all_measurements, label_ids = bd.load_csv(
    "/home/fatemeh/Downloads/bird/data/combined_s_w_m_j.csv"
)
all_measurements, label_ids = bd.get_specific_labesl(
    all_measurements, label_ids, target_labels
)
# use 80% data, the first 20% used for fine-tuning
all_measurements = all_measurements[872:]
label_ids = label_ids[872:]

n_trainings = int(all_measurements.shape[0] * train_per * data_per)
n_valid = all_measurements.shape[0] - n_trainings
train_measurments = all_measurements[:n_trainings]
valid_measurements = all_measurements[n_trainings : n_trainings + n_valid]
train_labels, valid_labels = (
    label_ids[:n_trainings],
    label_ids[n_trainings : n_trainings + n_valid],
)
print(
    len(train_labels),
    len(valid_labels),
    train_measurments.shape,
    valid_measurements.shape,
)

train_dataset = bd.BirdDataset(train_measurments, train_labels)
eval_dataset = bd.BirdDataset(valid_measurements, valid_labels)

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"data shape: {train_dataset[0][0].shape}")  # 3x20
in_channel = train_dataset[0][0].shape[0]  # 3 or 4
model = bm1.MaskedAutoencoderViT(
    img_size=20,
    in_chans=4,
    patch_size=1,
    embed_dim=16,
    depth=1,
    num_heads=8,
    decoder_embed_dim=16,
    decoder_depth=1,
    decoder_num_heads=8,
    mlp_ratio=4,
    layer_norm_eps=1e-6,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

len_train, len_eval = len(train_dataset), len(eval_dataset)
print(
    f"device: {device}, train: {len_train:,}, valid: {len_eval:,} \
    images, train_loader: {len(train_loader)}, eval_loader: {len(eval_loader)}"
)
best_loss = best_loss = float("inf")
with tensorboard.SummaryWriter(save_path / f"tensorboard/{exp}") as writer:
    for epoch in tqdm.tqdm(range(1, no_epochs + 1)):
        start_time = datetime.now()
        print(f"start time: {start_time}")
        train_one_epoch(
            train_loader, model, device, epoch, no_epochs, writer, optimizer
        )
        loss = evaluate(eval_loader, model, device, epoch, no_epochs, writer)
        end_time = datetime.now()
        print(f"end time: {end_time}, elapse time: {end_time-start_time}")

        scheduler.step()
        lr_optim = round(optimizer.param_groups[-1]["lr"], 6)
        lr_sched = scheduler.get_last_lr()[0]
        writer.add_scalar("lr/optim", lr_optim, epoch)
        writer.add_scalar("lr/sched", lr_sched, epoch)
        print(
            f"optim: {optimizer.param_groups[-1]['lr']:.6f}, sched: {scheduler.get_last_lr()[0]:.6f}"
        )

        if epoch % save_every == 0:
            bm.save_model(save_path, exp, epoch, model, optimizer, scheduler)
        # save best model
        if loss < best_loss:
            best_loss = loss
            # 1-based save for epoch
            bm.save_model(save_path, exp, epoch, model, optimizer, scheduler, best=True)
            print(f"best model loss: {best_loss:.2f} at epoch: {epoch}")

# 1-based save for epoch
bm.save_model(save_path, exp, epoch, model, optimizer, scheduler)
