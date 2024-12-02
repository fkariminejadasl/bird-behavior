from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset, random_split

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.utils import n_classes, new_label_inds, target_labels


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


def read_csv_files(directory):
    gimus = []
    dis = []
    timestamps = []
    for csv_file in directory.glob("*.csv"):
        df = pd.read_csv(csv_file, header=None)
        gimus.append(df[[4, 5, 6, 7]].values)
        dis.append(df[[0, 2]].values)
        timestamps.extend(df[1].tolist())

    gimus = np.concatenate(gimus, axis=0)
    dis = np.concatenate(dis, axis=0)
    timestamps = np.array(timestamps)
    return gimus


class BirdDataset(Dataset):
    def __init__(self, gimus: np.ndarray, transform=None):
        self.data = gimus.copy()  # NxLxC C=4
        # normalize gps speed by max
        self.data[:, :, 3] = self.data[:, :, 3] / 22.3012351755624
        self.data = self.data.astype(np.float32)

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ind):
        data = self.data[ind].transpose((1, 0))  # LxC -> CxL

        if self.transform:
            data = self.transform(data)

        return data


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


import wandb

wandb.init(project="bird-large-pt")

print(
    f"count: {torch.cuda.device_count()}, device type: {torch.cuda.get_device_name(0)}, property: {torch.cuda.get_device_properties(0)}"
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# nvidia-smi -i 0 # also show properties

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
generator = torch.Generator().manual_seed(seed)  # for random_split

# gimus = read_csv_file("/home/fatemeh/Downloads/bird/data/combined_s_w_m_j_no_others.csv")
# gimus = read_csv_file("/home/fatemeh/Downloads/bird/ssl/tmp3/304.csv")
data_file = Path("/home/fatemeh/Downloads/bird/data/final/combined_unique.csv")
directory = Path("/home/fatemeh/Downloads/bird/data/ssl/final")
# directory = Path("/gpfs/home4/fkarimineja/data/bird/ssl")
save_path = Path("/home/fatemeh/Downloads/bird/result/")
# save_path = Path("/gpfs/home4/fkarimineja/exp/bird/runs")
save_path.mkdir(parents=True, exist_ok=True)

exp = "f_mem1"
# model_checkpoint = "/gpfs/home4/fkarimineja/exp/bird/runs/p_mem6_500.pth"
model_checkpoint = "/home/fatemeh/Downloads/bird/result/p_mem5_1.pth"
num_workers = 15  # 17, 15
no_epochs = 1  # 500, 1
save_every = 200
train_per = 0.9
data_per = 1.0

# hyperparam
warmup_epochs = 1000
step_size = 2000
max_lr = 3e-4  # 1e-3
min_lr = max_lr / 10
weight_decay = 1e-2  # default 1e-2
# model
g_len = 60  # 60, 20
in_channel = 4
out_channel = 9
patch_size = 1
embed_dim = 16  # 256, 16
depth = 1  # 6, 1
num_heads = 8
decoder_embed_dim = 16  # 256, 16
decoder_depth = 1  # 6, 1
decoder_num_heads = 8
mlp_ratio = 4
drop = 0.0


train_dataset, eval_dataset = bd.prepare_train_valid_dataset(
    data_file, train_per, data_per, target_labels
)


print(len(train_dataset) + len(eval_dataset), len(train_dataset), len(eval_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=min(4000, len(train_dataset)),
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
    pin_memory=True,  # fast but more memory
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=min(4000, len(eval_dataset)),
    shuffle=False,
    num_workers=num_workers,
    drop_last=True,
    pin_memory=True,
)

print(f"data shape: {train_dataset[0][0].shape}")  # 3x20
# in_channel = train_dataset[0][0].shape[0]  # 3 or 4
model = bm1.TransformerEncoderMAE(
    img_size=g_len,
    in_chans=in_channel,
    out_chans=out_channel,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    drop=drop,
    layer_norm_eps=1e-6,
).to(device)

# bm.load_model(model_checkpoint, model, device)
pmodel = torch.load(model_checkpoint, weights_only=True)["model"]
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
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

len_train, len_eval = len(train_dataset), len(eval_dataset)
print(
    f"device: {device}, train: {len_train:,}, valid: {len_eval:,} \
    images, train_loader: {len(train_loader)}, eval_loader: {len(eval_loader)}"
)
print(f"number of paratmeters: {sum(i.numel() for i in model.parameters()):,}")
best_accuracy = 0
with tensorboard.SummaryWriter(save_path / f"tensorboard/{exp}") as writer:
    for epoch in tqdm.tqdm(range(1, no_epochs + 1)):
        start_time = datetime.now()
        print(f"start time: {start_time}")
        get_gpu_memory()
        bm.train_one_epoch(
            train_loader, model, criterion, device, epoch, no_epochs, writer, optimizer
        )
        accuracy = bm.evaluate(
            eval_loader, model, criterion, device, epoch, no_epochs, writer
        )
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

        if epoch % save_every == 0:
            bm.save_model(save_path, exp, epoch, model, optimizer, scheduler)
        # save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 1-based save for epoch
            bm.save_model(save_path, exp, epoch, model, optimizer, scheduler, best=True)
            print(f"best model accuracy: {best_accuracy:.2f} at epoch: {epoch}")

# 1-based save for epoch
bm.save_model(save_path, exp, epoch, model, optimizer, scheduler)
