from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm
from omegaconf import OmegaConf
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.utils import target_labels


@dataclass
class PathConfig:
    save_path: Path
    data_file: Path
    model_checkpoint: Path


cfg_file = Path(__file__).parents[1] / "configs/main_finetune.yaml"
cfg = OmegaConf.load(cfg_file)
cfg_paths = OmegaConf.structured(
    PathConfig(
        save_path=cfg.save_path,
        data_file=cfg.data_file,
        model_checkpoint=cfg.model_checkpoint,
    )
)
cfg = OmegaConf.merge(cfg, cfg_paths)

bu.set_seed(cfg.seed)

all_measurements, label_ids = bd.load_csv(cfg.data_file)
all_measurements, label_ids = bd.get_specific_labesl(
    all_measurements, label_ids, target_labels
)

# all = 4365
n_trainings = 100  # (10% data)# int(all_measurements.shape[0] * train_per * data_per)
n_valid = 100  # all_measurements.shape[0] - n_trainings
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

# ind_data = int(data_per * len(all_measurements))
# all_measurements, label_ids = all_measurements[:ind_data], label_ids[:ind_data]
# dataset = bd.BirdDataset(all_measurements, label_ids)
# train_size = int(train_per * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, eval_dataset = random_split(dataset, [train_size, val_size], generator)

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
model = bm1.TransformerEncoderMAE(
    img_size=cfg.g_len,
    in_chans=cfg.in_channel,
    out_chans=cfg.out_channel,
    embed_dim=cfg.embed_dim,
    depth=cfg.depth,
    num_heads=cfg.num_heads,
    mlp_ratio=cfg.mlp_ratio,
    drop=cfg.drop,
    layer_norm_eps=cfg.layer_norm_eps,
).to(device)

pmodel = torch.load(cfg.model_checkpoint, weights_only=True)["model"]
state_dict = model.state_dict()
for name, p in pmodel.items():
    if (
        "decoder" not in name and "mask" not in name
    ):  # and name!="norm.weight" and name!="norm.bias":
        state_dict[name].data.copy_(p.data)
        # dict(model.named_parameters())[name].requires_grad = False # freeze all layers except class head
print(
    f"fc: {model.fc.weight.requires_grad}, other:{model.blocks[0].norm2.weight.requires_grad}"
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=cfg.step_size, gamma=0.1
)


len_train, len_eval = len(train_dataset), len(eval_dataset)
print(
    f"device: {device}, train: {len_train:,}, valid: {len_eval:,} \
    images, train_loader: {len(train_loader)}, eval_loader: {len(eval_loader)}"
)
best_accuracy = 0
with tensorboard.SummaryWriter(cfg.save_path / f"tensorboard/{cfg.exp}") as writer:
    for epoch in tqdm.tqdm(range(1, cfg.no_epochs + 1)):
        # tqdm.tqdm(range(4001, no_epochs + 1)): # start from a checkpoint
        start_time = datetime.now()
        print(f"start time: {start_time}")
        bm.train_one_epoch(
            train_loader,
            model,
            criterion,
            device,
            epoch,
            cfg.no_epochs,
            writer,
            optimizer,
        )
        accuracy = bm.evaluate(
            eval_loader, model, criterion, device, epoch, cfg.no_epochs, writer
        )
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

        if epoch % cfg.save_every == 0:
            bm.save_model(cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler)
        # save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 1-based save for epoch
            bm.save_model(
                cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler, best=True
            )
            print(f"best model accuracy: {best_accuracy:.2f} at epoch: {epoch}")

# 1-based save for epoch
bm.save_model(cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler)
