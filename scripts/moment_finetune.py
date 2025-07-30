from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import DistributedType
from accelerate.utils import DistributedDataParallelKwargs
from momentfm import MOMENTPipeline
from momentfm.data.classification_dataset import ClassificationDataset
from momentfm.utils.masking import Masking
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import contingency_matrix
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.utils import get_gpu_memory, set_seed

# import wandb
# wandb.init(project="bird-moment-pt", config=cfg_dict)


def bird_collate_fn(batch, seq_len=32):
    """
    batch: list of (data, label) tuples where
      data is Tensor shaped (C, g_len)
      label is Tensor shaped (…)
    returns:
      x_flat:    Tensor of shape (B, C, seq_len)
      mask_flat: Tensor of shape (B, seq_len)
      y_batch:   Tensor of shape (B, …)
    """
    # unpack
    has_label = len(batch[0]) == 2
    if has_label:
        xs, ys = zip(*batch)
    else:
        xs = batch

    # Stack into (B, C, g_len)
    x = torch.stack(xs, dim=0)  # (B, C, g_len)
    B, C, g_len = x.shape

    # pad last dim L→seq_len
    if g_len < seq_len:
        # pad: (left, right) on final (L) dim
        x = F.pad(x, (0, seq_len - g_len), mode="constant", value=0)
    # now x is (B, C, seq_len)

    # flatten channels (B*C, 1, seq_len)
    # x_flat = x.reshape(B * C, 1, seq_len)
    x_flat = x  # (B, C, seq_len)

    # build a single-channel 2D mask:
    # 1 for real frames [0:g_len), 0 for padding [g_len:seq_len)
    base_mask = x.new_ones(seq_len, dtype=torch.long)
    base_mask[g_len:] = 0  # shape: (seq_len,)

    # repeat for every (sample,channel) row -> (B*C, seq_len)
    # mask_flat = base_mask.unsqueeze(0).repeat(B * C, 1)
    mask_flat = base_mask.unsqueeze(0).repeat(B, 1)  # (B, seq_len)

    if has_label:
        y = torch.stack(ys, dim=0)[:, 0]  # (B, 3) -> (B,)
        return x_flat, mask_flat, y
    return x_flat, mask_flat


def write_info_in_tensorboard(writer, epoch, loss, accuracy, stage):
    loss_scalar_dict = dict()
    acc_scalar_dict = dict()
    loss_scalar_dict[stage] = loss
    acc_scalar_dict[stage] = accuracy
    writer.add_scalars("loss", loss_scalar_dict, epoch)
    writer.add_scalars("accuracy", acc_scalar_dict, epoch)


def train_one_epoch(loader, model, device, epoch, no_epochs, writer, optimizer):
    stage = "train"

    model.train()
    running_loss = 0
    for i, (batch_x, batch_masks, batch_labels) in tqdm(
        enumerate(loader), total=len(loader)
    ):
        optimizer.zero_grad()
        batch_x = batch_x.to(device)  # [batch_size, n_channels, seq_len]
        batch_masks = batch_masks.to(device)  # [batch_size, seq_len]

        # Forward and compute loss
        output = model(x_enc=batch_x, input_mask=batch_masks, reduction=reduction)
        loss = criterion(output.logits, batch_labels)
        print(f"loss: {loss.item():.4f}")
        # bu.check_types(model)

        # Backward
        accelerator.backward(loss)
        # loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    total_loss = running_loss / (i + 1)

    print(f"{stage}: epoch/total: {epoch}/{no_epochs}, total loss: {total_loss:.4f}")
    write_info_in_tensorboard(writer, epoch, total_loss, total_loss, stage)


@torch.no_grad()
def evaluate(loader, model, device, epoch, no_epochs, writer):
    stage = "valid"

    model.eval()
    running_loss = 0
    for i, (batch_x, batch_masks, batch_labels) in tqdm(
        enumerate(loader), total=len(loader)
    ):
        batch_x = batch_x.to(device)  # [batch_size, n_channels, seq_len]
        batch_masks = batch_masks.to(device)  # [batch_size, seq_len]
        batch_labels = batch_labels.to(device)

        # Forward and compute loss
        output = model(x_enc=batch_x, input_mask=batch_masks, reduction=reduction)
        loss = criterion(output.logits, batch_labels)
        print(f"loss: {loss.item():.4f}")

        running_loss += loss.item()

    total_loss = running_loss / (i + 1)
    print(f"{stage}: epoch/total: {epoch}/{no_epochs}, total loss: {total_loss:.4f}")
    write_info_in_tensorboard(writer, epoch, total_loss, total_loss, stage)
    return total_loss


batch_size, n_channels, seq_len = 1000, 4, 32
g_len = 20
reduction = "mean"  # 'mean' or 'concat'

cfg = {
    "seed": 42,
    "exp": "mft_1",
    "data_path": Path("/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv"),
    "save_path": Path("/home/fatemeh/Downloads/bird/result"),
    "checkpoint_path": Path("/home/fatemeh/Downloads/bird/result/mpt_1_best.pth"),
    "no_epochs": 1,
    "init_lr": 1e-6,
    "max_lr": 1e-4,
    "PEFT": False,
    "save_every": 2,
    "num_workers": 0,
}
cfg = SimpleNamespace(**cfg)

set_seed(cfg.seed)
generator = torch.Generator().manual_seed(cfg.seed)  # for random_split

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-small",
    model_kwargs={
        "task_name": "classification",
        "n_channels": 4,
        "num_class": 9,
        "freeze_encoder": True,  # Freeze the transformer encoder
        "freeze_embedder": True,  # Freeze the patch embedding layer
        "freeze_head": False,  # The linear forecasting head must be trained
        ## NOTE: Disable gradient checkpointing to supress the warning when linear probing the model as MOMENT encoder is frozen
        "enable_gradient_checkpointing": False,
        # Choose how embedding is obtained from the model: One of ['mean', 'concat']
        # Multi-channel embeddings are obtained by either averaging or concatenating patch embeddings
        # along the channel dimension. 'concat' results in embeddings of size (n_channels * d_model),
        # while 'mean' results in embeddings of size (d_model)
        "reduction": "mean",
    },
)
model.init()
model.to(device)

# Load model
pmodel = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)[
    "model"
]
state_dict = model.state_dict()
for name, p in pmodel.items():
    if "head" not in name:
        state_dict[name].data.copy_(p.data)
del pmodel, name, p, state_dict
torch.cuda.empty_cache()
print("model is loaded")


train_dataset, val_dataset = bd.prepare_train_valid_dataset(
    cfg.data_path,
    0.9,
    1,
    [0, 1, 2, 3, 4, 5, 6, 8, 9],
    channel_first=True,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=min(batch_size, len(train_dataset)),
    shuffle=True,
    num_workers=cfg.num_workers,
    drop_last=False,
    collate_fn=lambda b: bird_collate_fn(b, seq_len=seq_len),
    # pin_memory=True,  # fast but more memory
)
val_loader = DataLoader(
    val_dataset,
    batch_size=min(batch_size, len(val_dataset)),
    shuffle=False,
    num_workers=cfg.num_workers,
    drop_last=False,
    collate_fn=lambda b: bird_collate_fn(b, seq_len=seq_len),
    # pin_memory=True,  # fast but more memory
)
len_data = len(train_dataset) + len(val_dataset)
print(f"All: {len_data}, Train: {len(train_dataset)}, valid: {len(val_dataset)}")


# Optimize Mean Squarred Error using your favourite optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.init_lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=cfg.max_lr, total_steps=cfg.no_epochs * len(train_loader)
)
print(
    f"optim: {optimizer.param_groups[-1]['lr']:.6f}, sched: {scheduler.get_last_lr()[0]:.6f}"
)

# Set up model ready for accelerate training
accelerator = Accelerator()
dist_type = accelerator.state.distributed_type
# # finetuning code seems don't have issue with unused parameters. If I use it I get warning for performance.
# if dist_type == DistributedType.MULTI_GPU:
#     print("Running on multiple GPUs!")
#     ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
#     accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
device = accelerator.device
model, optimizer, train_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, scheduler
)


# NB. Chage in the peft code to make the below code works.
# In lib/python3.x/site-packages/peft/tuners/tuners_utils.py, model_config.get("tie_word_embeddings") to model_config.t5_config["tie_word_embeddings"]
if cfg.PEFT:
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

best_loss = best_loss = float("inf")
with tensorboard.SummaryWriter(cfg.save_path / f"tensorboard/{cfg.exp}") as writer:
    for epoch in tqdm(range(1, cfg.no_epochs + 1)):
        start_time = datetime.now()
        print(f"start time: {start_time}")
        get_gpu_memory()
        train_one_epoch(
            train_loader, model, device, epoch, cfg.no_epochs, writer, optimizer
        )
        loss = evaluate(val_loader, model, device, epoch, cfg.no_epochs, writer)
        get_gpu_memory()
        end_time = datetime.now()
        print(f"end time: {end_time}, elapse time: {end_time-start_time}")

        lr_optim = round(optimizer.param_groups[-1]["lr"], 6)
        lr_sched = scheduler.get_last_lr()[0]
        writer.add_scalar("lr/optim", lr_optim, epoch)
        writer.add_scalar("lr/sched", lr_sched, epoch)
        print(
            f"optim: {optimizer.param_groups[-1]['lr']:.6f}, sched: {scheduler.get_last_lr()[0]:.6f}"
        )

        # to save it with pytorch. Otherwise it gets "module" in start of weight names
        unwrapped_model = accelerator.unwrap_model(model)
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

# # 1-based save for epoch
bm.save_model(cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler)
