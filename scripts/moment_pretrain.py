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

"""
This code is adapted from these MOMENT tutorials:
https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/anomaly_detection.ipynb
https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/imputation.ipynb

small: trainable params:   917,504 || all params:  36,259,016 || trainable%: 2.5304
base:  trainable params: 2,359,296 || all params: 112,000,904 || trainable%: 2.1065
large: trainable params: 6,291,456 || all params: 347,539,976 || trainable%: 1.8103
"""


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
        y = torch.stack(ys, dim=0)[:, 0]
        return x_flat, mask_flat, y
    return x_flat, mask_flat


def read_csv_file(csv_file):
    gimus = []
    dis = []
    timestamps = []
    df = pd.read_csv(csv_file, header=None)
    gimus.append(df[[4, 5, 6, 7]].values)
    dis.append(df[[0, 2]].values)
    timestamps.extend(df[1].tolist())
    gimus = np.concatenate(gimus, axis=0)
    dis = np.concatenate(dis, axis=0)
    timestamps = np.array(timestamps)
    return gimus


def read_csv_files(data_path):
    gimus = []
    dis = []
    timestamps = []
    for csv_file in data_path.glob("*.csv"):
        df = pd.read_csv(csv_file, header=None)
        gimus.append(df[[4, 5, 6, 7]].values)
        dis.append(df[[0, 2]].values)
        timestamps.extend(df[1].tolist())

    gimus = np.concatenate(gimus, axis=0)
    dis = np.concatenate(dis, axis=0)
    timestamps = np.array(timestamps)
    return gimus


def write_info_in_tensorboard(writer, epoch, loss, stage):
    loss_scalar_dict = dict()
    loss_scalar_dict[stage] = loss
    writer.add_scalars("loss", loss_scalar_dict, epoch)


def train_one_epoch(loader, model, device, epoch, n_epochs, writer, optimizer):
    stage = "train"

    model.train()
    running_loss = 0
    for i, (batch_x, batch_masks) in tqdm(enumerate(loader), total=len(loader)):
        optimizer.zero_grad()
        batch_x = batch_x.to(device)  # [batch_size, n_channels, seq_len]
        batch_masks = batch_masks.to(device)  # [batch_size, seq_len]
        # Randomly mask some patches of data
        mask = (
            mask_generator.generate_mask(x=batch_x, input_mask=batch_masks)
            .to(device)
            .long()
        )

        # Forward
        output = model(x_enc=batch_x, input_mask=batch_masks, mask=mask)
        # bu.check_types(model)

        # Compute loss
        recon_loss = criterion(output.reconstruction, batch_x)
        observed_mask = batch_masks * (1 - mask)
        masked_loss = observed_mask * recon_loss
        loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)
        print(f"{datetime.now().replace(microsecond=0)}: loss: {loss.item():.4f}")

        # Backward
        accelerator.backward(loss)
        # loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    total_loss = running_loss / (i + 1)

    print(f"{stage}: epoch/total: {epoch}/{n_epochs}, total loss: {total_loss:.4f}")
    write_info_in_tensorboard(writer, epoch, total_loss, stage)


@torch.no_grad()
def evaluate(loader, model, device, epoch, n_epochs, writer):
    stage = "valid"

    model.eval()
    running_loss = 0
    for i, (batch_x, batch_masks) in tqdm(enumerate(loader), total=len(loader)):
        batch_x = batch_x.to(device)  # [batch_size, n_channels, seq_len]
        batch_masks = batch_masks.to(device)  # [batch_size, seq_len]
        # Randomly mask some patches of data
        mask = (
            mask_generator.generate_mask(x=batch_x, input_mask=batch_masks)
            .to(device)
            .long()
        )

        # Forward
        output = model(x_enc=batch_x, input_mask=batch_masks, mask=mask)

        # Compute loss
        recon_loss = criterion(output.reconstruction, batch_x)
        observed_mask = batch_masks * (1 - mask)
        masked_loss = observed_mask * recon_loss
        loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)
        print(f"{datetime.now().replace(microsecond=0)}: loss: {loss.item():.4f}")

        running_loss += loss.item()

    total_loss = running_loss / (i + 1)
    print(f"{stage}: epoch/total: {epoch}/{n_epochs}, total loss: {total_loss:.4f}")
    write_info_in_tensorboard(writer, epoch, total_loss, stage)
    return total_loss


cfg = {
    # Result
    "seed": 42,
    "exp": "mpt_1",
    "save_path": Path("/home/fatemeh/Downloads/bird/results"),
    # "save_path": Path("/home/fkarimi/exps/bird"),
    # Data
    "data_path": Path("/home/fatemeh/Downloads/bird/data/ssl_mini"),
    # "data_path": Path("/home/fkarimi/data/bird/ssl_mini"), # #ssl/final/gull
    # Model
    "batch_size": 1000,
    "n_channels": 4,
    "seq_len": 32,
    "g_len": 20,
    # Training
    "n_epochs": 10,
    "init_lr": 1e-6,
    "max_lr": 1e-4,
    "PEFT": False,
    "save_every": None,
    "num_workers": 0,
    "WANDB": False,
    # System
    "n_cpus": None,
}
cfg = SimpleNamespace(**cfg)

if cfg.n_cpus is not None:
    torch.set_num_threads(cfg.n_cpus)
    torch.set_num_interop_threads(cfg.n_cpus)

batch_size, n_channels, seq_len = cfg.batch_size, cfg.n_channels, cfg.seq_len
g_len = cfg.g_len
if cfg.save_every is None:
    cfg.save_every = cfg.n_epochs

if cfg.WANDB:
    import wandb

    wandb.init(project="bird-moment-pt", config=cfg)

bu.print_enviroment_info()

set_seed(cfg.seed)
generator = torch.Generator().manual_seed(cfg.seed)  # for random_split

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-small",
    model_kwargs={
        "task_name": "reconstruction",
        "freeze_encoder": False,  # Freeze the transformer encoder
        "freeze_embedder": False,  # Freeze the patch embedding layer
        "freeze_head": False,  # The linear forecasting head must be trained} # For imputation, we will load MOMENT in `reconstruction` mode
        "enable_gradient_checkpointing": False,
    },
)
model.init()
model.to(device)


gimus = read_csv_files(cfg.data_path)
gimus = gimus.reshape(-1, g_len, n_channels)
"""
# test: small data
import gc
gimus = gimus[:2]
gc.collect()
"""
gimus = np.ascontiguousarray(gimus)
dataset = bd.BirdDataset(gimus, channel_first=True)
# Calculate the sizes for training and validation datasets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
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
print(f"All: {gimus.shape}, Train: {len(train_dataset)}, valid: {len(val_dataset)}")

# Optimize Mean Squarred Error using your favourite optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.init_lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=cfg.max_lr, total_steps=cfg.n_epochs * len(train_loader)
)
print(
    f"optim: {optimizer.param_groups[-1]['lr']:.6f}, sched: {scheduler.get_last_lr()[0]:.6f}"
)

# Set up model ready for accelerate training
accelerator = Accelerator()
dist_type = accelerator.state.distributed_type
if dist_type == DistributedType.MULTI_GPU:
    print("Running on multiple GPUs!")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
device = accelerator.device
model, optimizer, train_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, scheduler
)

mask_generator = Masking(mask_ratio=0.3)  # Mask 30% of patches randomly


# NB. Chage in the peft code to make the below code works.
# This is only when running `python script_name.py` not with `accelerate launch`.
# In accelerate launch, model get `module` namespace due to accelerate,
# e.g. model.module.encoder.final_layer_norm.weight[:10] instead of model.encoder.final_layer_norm.weight[:10]
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

best_loss = float("inf")
with tensorboard.SummaryWriter(cfg.save_path / f"tensorboard/{cfg.exp}") as writer:
    for epoch in tqdm(range(1, cfg.n_epochs + 1)):
        start_time = datetime.now().replace(microsecond=0)
        print(f"start time: {start_time}")
        get_gpu_memory()
        train_one_epoch(
            train_loader, model, device, epoch, cfg.n_epochs, writer, optimizer
        )
        loss = evaluate(val_loader, model, device, epoch, cfg.n_epochs, writer)
        get_gpu_memory()
        end_time = datetime.now().replace(microsecond=0)
        print(f"end time: {end_time}, elapse time: {end_time-start_time}")

        lr_optim = round(optimizer.param_groups[-1]["lr"], 6)
        lr_sched = scheduler.get_last_lr()[0]
        writer.add_scalar("lr/optim", lr_optim, epoch)
        writer.add_scalar("lr/sched", lr_sched, epoch)
        print(
            f"optim: {optimizer.param_groups[-1]['lr']:.6f}, sched: {scheduler.get_last_lr()[0]:.6f}"
        )

        # to save it with pytorch. Otherwise it gets "module" in start of weight names
        if cfg.PEFT:
            unwrapped_model = model.base_model.module  # accelerator.unwrap_model(model)
        else:
            unwrapped_model = model.module
        # print(list(dict(model.named_parameters()).keys())[-1])
        # print(list(dict(unwrapped_model.named_parameters()).keys())[-1])
        if epoch % cfg.save_every == 0:
            bm.save_model(
                cfg.save_path, cfg.exp, epoch, unwrapped_model, optimizer, scheduler
            )
            # save best model
        if loss < best_loss:
            best_loss = loss
            # 1-based save for epoch
            bm.save_model(
                cfg.save_path,
                cfg.exp,
                epoch,
                unwrapped_model,
                optimizer,
                scheduler,
                best=True,
            )
            print(f"best model loss: {best_loss:.2f} at epoch: {epoch}")

# # 1-based save for epoch
bm.save_model(cfg.save_path, cfg.exp, epoch, unwrapped_model, optimizer, scheduler)
