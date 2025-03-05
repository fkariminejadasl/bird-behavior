from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from omegaconf import OmegaConf
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset, random_split

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1


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


def caculate_metrics(data, model, device):
    data = data.to(device)  # NxLxC
    loss, _, _ = model(data)  # NxC
    return loss


def train_one_epoch(loader, model, device, epoch, no_epochs, writer, optimizer):
    stage = "train"

    model.train()
    running_loss = 0
    for i, data in enumerate(loader):
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
    for i, data in enumerate(loader):
        loss = caculate_metrics(data, model, device)

        running_loss += loss.item()

    total_loss = running_loss / (i + 1)
    print(f"{stage}: epoch/total: {epoch}/{no_epochs}, total loss: {total_loss:.4f}")
    write_info_in_tensorboard(writer, epoch, total_loss, stage)
    return total_loss


def get_gpu_memory():
    # Total memory currently allocated by tensors
    allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # in GB
    # Total memory reserved by the caching allocator (may be more than allocated_memory)
    reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)  # in GB
    # Total memory available on the GPU
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
    print(f"Allocated memory: {allocated_memory:.2f} GB")
    print(f"Reserved memory: {reserved_memory:.2f} GB")
    print(f"Total GPU memory: {total_memory:.2f} GB")
    # torch.cuda.empty_cache()


@dataclass
class PathConfig:
    save_path: Path
    data_path: Path
    model_checkpoint: Path


cfg_file = Path(__file__).parents[1] / "configs/pretrain_memory_load.yaml"
cfg = OmegaConf.load(cfg_file)
cfg_paths = OmegaConf.structured(
    PathConfig(
        save_path=cfg.save_path,
        data_path=cfg.data_path,
        model_checkpoint=cfg.model_checkpoint,
    )
)
cfg = OmegaConf.merge(cfg, cfg_paths)
cfg.min_lr = cfg.max_lr / 10

# Convert the DictConfig to a standard dictionary
cfg_dict = OmegaConf.to_container(cfg, resolve=True)
import wandb

wandb.init(project="bird-large-pt", config=cfg_dict)

cfg.save_path.mkdir(parents=True, exist_ok=True)

# Device
print(
    f"count: {torch.cuda.device_count()}, device type: {torch.cuda.get_device_name(0)}, property: {torch.cuda.get_device_properties(0)}"
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# nvidia-smi -i 0 # also show properties

# Random seed
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
generator = torch.Generator().manual_seed(cfg.seed)  # for random_split

# Data
gimus = read_csv_files(cfg.data_path)
print(gimus.shape)
gimus = gimus.reshape(-1, cfg.g_len, cfg.in_channel)
gimus = np.ascontiguousarray(gimus)
print(gimus.shape)
dataset = bd.BirdDataset(gimus, channel_first=False)
# Calculate the sizes for training and validation datasets
train_size = int(cfg.train_per * cfg.data_per * len(dataset))
val_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, val_size])

print(len(dataset), len(train_dataset), len(eval_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=min(cfg.batch_size, len(train_dataset)),
    shuffle=True,
    num_workers=cfg.num_workers,
    drop_last=False,
    pin_memory=True,  # fast but more memory
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=min(cfg.batch_size, len(eval_dataset)),
    shuffle=False,
    num_workers=cfg.num_workers,
    drop_last=False,
    pin_memory=True,
)

print(f"data shape: {train_dataset[0][0].shape}")  # 3x20
# in_channel = train_dataset[0][0].shape[0]  # 3 or 4
model = bm1.MaskedAutoencoderViT(
    img_size=cfg.g_len,
    in_chans=cfg.in_channel,
    patch_size=cfg.patch_size,
    embed_dim=cfg.embed_dim,
    depth=cfg.depth,
    num_heads=cfg.num_heads,
    decoder_embed_dim=cfg.decoder_embed_dim,
    decoder_depth=cfg.decoder_depth,
    decoder_num_heads=cfg.decoder_num_heads,
    mlp_ratio=cfg.mlp_ratio,
    layer_norm_eps=cfg.layer_norm_eps,
).to(device)
if cfg.model_checkpoint != Path("."):
    bm.load_model(cfg.model_checkpoint, model, device)
    print(f"{cfg.model_checkpoint} used")
else:
    print("NO Check point is used")

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
print(f"number of paratmeters: {sum(i.numel() for i in model.parameters()):,}")

best_loss = best_loss = float("inf")
with tensorboard.SummaryWriter(cfg.save_path / f"tensorboard/{cfg.exp}") as writer:
    for epoch in tqdm.tqdm(range(1, cfg.no_epochs + 1)):
        start_time = datetime.now()
        print(f"start time: {start_time}")
        get_gpu_memory()
        train_one_epoch(
            train_loader, model, device, epoch, cfg.no_epochs, writer, optimizer
        )
        loss = evaluate(eval_loader, model, device, epoch, cfg.no_epochs, writer)
        get_gpu_memory()
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
        if loss < best_loss:
            best_loss = loss
            # 1-based save for epoch
            bm.save_model(
                cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler, best=True
            )
            print(f"best model loss: {best_loss:.2f} at epoch: {epoch}")

# 1-based save for epoch
bm.save_model(cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler)
