"""
Non contrastive learning: DINO / iBOT / SimSiam / BYOL / SwAV / VICReg / BarlowTwins
Contrastive learning: SimCLR / MoCo

This script is used to train non-contrastive self-supervised learning models.
Other names can be non contrastive / self distillation / alignmet: train_self_distill.py, train_non_contrastive_ssl.py  

Based on: https://github.com/CVMI-Lab/SimGCD/blob/main/train_mp.py
"""

import gc
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
from omegaconf import OmegaConf
from scipy import io as mat_io
from torch.nn.functional import normalize
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import v2 as tvt2

from behavior import data as bd
from behavior import data_augmentation as bau
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.utils import new_label_inds


def inference(data, model, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        data = data.to(device)  # N x C x L
        outputs = model(data)  # N x C
        # prob = torch.nn.functional.softmax(outputs, dim=-1)  # N x C
        pred = torch.argmax(outputs.data, 1)  # N

    prob = outputs.cpu()  # .numpy()
    pred = pred.cpu().numpy()
    return prob, pred


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def info_nce_logits(features, n_views=2, temperature=1.0, device="cuda"):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def train(
    student, train_loader, optimizer, scaler, scheduler, cluster_criterion, epoch, args
):
    # loss_record = AverageMeter()

    student.train()
    for batch_idx, batch in enumerate(train_loader):
        images, class_labels, uq_idxs, mask_lab = batch
        mask_lab = mask_lab[:, 0]

        class_labels, mask_lab = (
            class_labels.cuda(non_blocking=True),
            mask_lab.cuda(non_blocking=True).bool(),
        )
        images = torch.cat(images, dim=0).cuda(non_blocking=True)

        with torch.amp.autocast(scaler is not None):
            student_proj, student_out = student(images)
            teacher_out = student_out.detach()

            # # clustering, sup
            # sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            # sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
            # cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            # # clustering, unsup
            # cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            # avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            # me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            # cluster_loss += args.memax_weight * me_max_loss

            # represent learning, unsup
            contrastive_logits, contrastive_labels = info_nce_logits(
                features=student_proj
            )
            contrastive_loss = torch.nn.CrossEntropyLoss()(
                contrastive_logits, contrastive_labels
            )

            # # representation learning, sup
            # student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            # student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
            # sup_con_labels = class_labels[mask_lab]
            # sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

            pstr = ""
            # pstr += f'cls_loss: {cls_loss.item():.4f} '
            # pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            # pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
            pstr += f"contrastive_loss: {contrastive_loss.item():.4f} "

            loss = 0

        # Train acc
        # loss_record.update(loss.item(), class_labels.size(0))
        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # if batch_idx % args.print_freq == 0 and dist.get_rank() == 0:
        #     args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
        #                 .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
    # Step schedule
    scheduler.step()

    # if dist.get_rank() == 0:
    #     args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))


def write_info_in_tensorboard(writer, epoch, loss, stage):
    loss_scalar_dict = dict()
    loss_scalar_dict[stage] = loss
    writer.add_scalars("loss", loss_scalar_dict, epoch)


def caculate_loss(data, model, device):
    model = model.to(device)
    data = torch.cat(data, dim=0).to(device)  # NxLxC
    # data = data.view(-1, data.shape[2], data.shape[1])  # NxCxL
    proj = model(data)  # NxC
    contrastive_logits, contrastive_labels = info_nce_logits(features=proj)
    loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
    return loss


def train_one_epoch(loader, model, device, epoch, no_epochs, writer, optimizer):
    stage = "train"

    model.train()
    running_loss = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()

        loss = caculate_loss(data, model, device)

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
    for i, data in enumerate(loader):
        loss = caculate_loss(data, model, device)

        running_loss += loss.item()

    total_loss = running_loss / (i + 1)
    print(f"{stage}: epoch/total: {epoch}/{no_epochs}, total loss: {total_loss:.4f}")
    write_info_in_tensorboard(writer, epoch, total_loss, stage)
    return total_loss


transform = tvt2.RandomChoice(
    [
        bau.RandomJitter(sigma=0.05),
        bau.RandomScaling(sigma=0.05),
        # bau.TimeWarp(sigma=0.05),
        # bau.MagnitudeWarp(sigma=0.05, knot=4),
    ]
)
transform = ContrastiveLearningViewGenerator(base_transform=transform, n_views=2)

cfg = dict(
    # # test_data_file=Path("/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv"),
    # data_path=Path("/home/fatemeh/Downloads/bird/data/ssl/parquetmini"),
    # # model_checkpoint=Path("/home/fatemeh/Downloads/bird/result/125_best.pth"),
    # model_checkpoint=Path("/home/fatemeh/Downloads/bird/snellius/p20_4_best.pth"),
    # save_path=Path("/home/fatemeh/Downloads/bird/result/"),
    data_path=Path("/home/fkarimineja/data/bird/ssl20parquet"),
    model_checkpoint=Path("/home/fkarimineja/exps/bird/runs/p20_4_best.pth"),
    save_path=Path("/home/fkarimineja/exps/bird/runs"),
    exp="self_distill_test",
    labels_to_use=[0, 1, 2, 3, 4, 5, 6, 8, 9],
    channel_first=False,
    # model parameters
    # # small model
    # in_channel=4,
    # mid_channel=30,
    # out_channel=9,
    # vit model
    g_len=20,  # 60, 20
    in_channel=4,
    out_channel=256,  # 6, 9
    embed_dim=256,  # 256, 16
    depth=6,  # 6, 1
    num_heads=8,
    decoder_embed_dim=256,  # 256, 16
    decoder_depth=6,  # 6, 1
    decoder_num_heads=8,
    mlp_ratio=4,
    drop=0.0,
    layer_norm_eps=1e-6,
    # General
    seed=1234,
    num_workers=1,  # 17 (a_100), 15 (h_100)
    no_epochs=1,
    save_every=200,
    # Data
    train_per=0.9,
    data_per=1.0,
    batch_size=1024,  # 4000
    # Training
    warmup_epochs=1000,
    step_size=2000,
    max_lr=3e-4,  # 1e-3
    min_lr=None,
    weight_decay=1e-2,  # default 1e-2
)
cfg = OmegaConf.create(cfg)
cfg.min_lr = cfg.max_lr / 10

# Convert the DictConfig to a standard dictionary
cfg_dict = OmegaConf.to_container(cfg, resolve=True)
import wandb

# wandb.init(project="bird-self-distill", config=cfg_dict)

bu.set_seed(cfg.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data
gimus = bd.load_gimu_data(cfg)
dataset = bd.BirdDatasetNoNorm(gimus, channel_first=False, transform=transform)
train_loader, eval_loader = bd.prepare_dataloaders(dataset, cfg)

# # dummy data loader
# df = pd.read_csv(cfg.test_data_file, header=None)
# df = df[df[3].isin(cfg.labels_to_use)]
# mapping = {l: i for i, l in enumerate(cfg.labels_to_use)}
# df[3] = df[3].map(mapping)
# all_measurements = df[[4, 5, 6, 7]].values.reshape(-1, 20, 4)
# label_ids = df[[3, 0, 0]].iloc[::20].values
# trainset = bd.BirdDataset(
#     all_measurements,
#     channel_first=cfg.channel_first,
#     transform=transform,
# )
# loader = torch.utils.data.DataLoader(
#     trainset, batch_size=len(trainset), shuffle=True, num_workers=1, drop_last=False
# )
# train_loader = deepcopy(loader)
# eval_loader = deepcopy(loader)

# Model
# model = bm.BirdModel(cfg.in_channel, cfg.mid_channel, cfg.out_channel)
# bm.load_model(cfg.model_checkpoint, model, device)
model = bm1.build_mae_vit_encoder_from_checkpoint(
    cfg.model_checkpoint, device, cfg, freeze_backbone=False
)

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=cfg.max_lr,
    weight_decay=cfg.weight_decay,
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=cfg.step_size, gamma=0.1
)

# Training loop
best_loss = float("inf")
with tensorboard.SummaryWriter(cfg.save_path / f"tensorboard/{cfg.exp}") as writer:
    for epoch in tqdm.tqdm(range(1, cfg.no_epochs + 1)):
        start_time = datetime.now()
        print(f"\nstart time: {start_time}")
        # get_gpu_memory()
        train_one_epoch(
            train_loader, model, device, epoch, cfg.no_epochs, writer, optimizer
        )
        loss = evaluate(eval_loader, model, device, epoch, cfg.no_epochs, writer)
        # get_gpu_memory()
        end_time = datetime.now()
        print(f"end time: {end_time}")
        print(f"elapse time: {end_time-start_time}")

        scheduler.step()
        lr_optim = round(optimizer.param_groups[-1]["lr"], 6)
        lr_sched = scheduler.get_last_lr()[0]
        writer.add_scalar("lr/optim", lr_optim, epoch)
        writer.add_scalar("lr/sched", lr_sched, epoch)
        print(
            f"optim: {optimizer.param_groups[-1]['lr']:.6f}, sched: {scheduler.get_last_lr()[0]:.6f}"
        )

        if epoch % cfg.save_every == 0:
            bm.save_model(cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler)
            # save best model
        if loss < best_loss:
            best_loss = loss
            # 1-based save for epoch
            bm.save_model(
                cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler, best=True
            )
            print(f"best model loss: {best_loss:.2f} at epoch: {epoch}")

# 1-based save for epoch
bm.save_model(cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler)
