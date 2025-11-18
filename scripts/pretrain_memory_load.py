import gc
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
from behavior.utils import get_gpu_memory, set_seed


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


class BirdDataset(Dataset):
    def __init__(
        self,
        all_measurements: np.ndarray,  # NxLxC
        ldts: np.ndarray = None,  # Nx3
        transform=None,
        channel_first=True,
    ):
        """
        dtype: all_measurements np.float32
        dtype: ldts np.int64 or None (if no labels are provided)
        :param channel_first: If True, data is returned in CxL format (channel-first). Otherwise, LxC (channel-last).
        """
        # data: NxLxC C=4
        self.data = np.ascontiguousarray(all_measurements, dtype=np.float32)

        self.has_label = ldts is not None  # Check if labels are provided
        if self.has_label:
            self.ldts = np.ascontiguousarray(ldts, dtype=np.int64)  # Nx3

        self.transform = transform
        self.channel_first = channel_first  # Flag for channel arrangement

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ind):
        data = self.data[ind]  # LxC

        data = torch.from_numpy(data)  # torch
        if self.transform:
            data = self.transform(data)

        # Rearrange channels if channel_first is True
        if self.channel_first:
            data = data.transpose(1, 0)  # LxC -> CxL

        if self.has_label:
            ldt = torch.from_numpy(self.ldts[ind])  # 3 torch
            # ldt = self.ldts[ind]  # 3 numpy
            return data, ldt  # Return both data and label
        else:
            return data


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
set_seed(cfg.seed)
generator = torch.Generator().manual_seed(cfg.seed)  # for random_split

# Data
gimus = []
parquet_files = cfg.data_path.glob("*.parquet")
for parquet_file in parquet_files:
    df = pd.read_parquet(parquet_file)
    data = np.vstack(df["gimu"].apply(lambda x: x.reshape(-1, 20, 4)))
    print(parquet_file.stem, data.shape)
    gimus.append(data)
gimus = np.vstack(gimus)

# free memory
del df, data
gc.collect()
# df = pd.read_parquet(cfg.data_path)
# gimus = np.vstack(df["gimu"].apply(lambda x: x.reshape(-1, 20, 4)))
print(gimus.shape)
# gimus = read_csv_files(cfg.data_path)
# gimus = gimus.reshape(-1, cfg.g_len, cfg.in_channel)
gimus = np.ascontiguousarray(gimus)
print(gimus.shape)
dataset = BirdDataset(gimus, channel_first=False)

# free memory
del gimus
gc.collect()

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
    , train_loader: {len(train_loader)}, eval_loader: {len(eval_loader)}"
)
print(f"number of paratmeters: {sum(i.numel() for i in model.parameters()):,}")

best_loss = float("inf")
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
