import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.amp
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import DistributedType
from accelerate.utils import DistributedDataParallelKwargs
from momentfm import MOMENTPipeline
from momentfm.data.classification_dataset import ClassificationDataset
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import contingency_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu


# This code is adapted from these MOMENT tutorials:
# https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/anomaly_detection.ipynb
# https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/imputation.ipynb
from momentfm.utils.masking import Masking

device = "cuda" if torch.cuda.is_available() else "cpu"

mask_generator = Masking(mask_ratio=0.25)

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
model.train()

# Optimize Mean Squarred Error using your favourite optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

from torch.utils.data import DataLoader, TensorDataset

batch_size, n_channels, seq_len = 2, 4, 32
# inputs = torch.rand(
#     (batch_size * 5, n_channels, seq_len), device=device, dtype=torch.float32
# )
# masks = torch.ones((inputs.shape[0], inputs.shape[2]), device=device)
# dataset = TensorDataset(inputs, masks)
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
generator = torch.Generator().manual_seed(seed)  # for random_split

import torch
import torch.nn.functional as F


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

    # stack into (B, C, g_len)
    x = torch.stack(xs, dim=0)  # (B, C, g_len)

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


data_path = Path("/home/fatemeh/Downloads/bird/data/ssl_mini")
g_len = 20
seq_len = 32
in_channel = 4
gimus = read_csv_files(data_path)
print(gimus.shape)
gimus = gimus.reshape(-1, g_len, in_channel)
gimus = np.ascontiguousarray(gimus)
print(gimus.shape)
dataset = bd.BirdDataset(gimus, channel_first=True)
# Calculate the sizes for training and validation datasets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, val_size])
batch_size = 1000
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False,
    collate_fn=lambda b: bird_collate_fn(b, seq_len=32),
)

# Set up model ready for accelerate finetuning
accelerator = Accelerator()
dist_type = accelerator.state.distributed_type
if dist_type == DistributedType.MULTI_GPU:
    print("Running on multiple GPUs!")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
device = accelerator.device
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

mask_generator = Masking(mask_ratio=0.3)  # Mask 30% of patches randomly

for batch_x, batch_masks in tqdm(train_loader):
    optimizer.zero_grad()
    batch_x = batch_x.to(device)  # [batch_size, n_channels, seq_len]
    batch_masks = batch_masks.to(device)  # [batch_size, seq_len]
    # Randomly mask some patches of data
    mask = (
        mask_generator.generate_mask(x=batch_x, input_mask=batch_masks)
        .to(device)
        .long()
    )

    # from collections import OrderedDict
    # seen: "OrderedDict[torch.nn.Module, bool]" = OrderedDict()

    # def make_hook(name):
    #     def hook(module, inp, out):
    #         # Only log the *first* time this module runs
    #         if not seen[name]:
    #             seen[name] = True
    #             if isinstance(out, torch.Tensor):
    #                 print(f"[{name:30s}] output dtype = {out.dtype}")
    #     return hook

    # # Initialize seen flags and register hooks
    # for name, module in model.named_modules():
    #     seen[name] = False
    #     module.register_forward_hook(make_hook(name))

    # Forward
    output = model(x_enc=batch_x, input_mask=batch_masks, mask=mask)

    # Compute loss
    recon_loss = criterion(output.reconstruction, batch_x)
    observed_mask = batch_masks * (1 - mask)
    masked_loss = observed_mask * recon_loss
    loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)
    print(f"loss: {loss.item()}")

    # Backward
    accelerator.backward(loss)
    # loss.backward()
    optimizer.step()
