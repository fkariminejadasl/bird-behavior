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

# This code is adapted from these MOMENT tutorials:
# https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/anomaly_detection.ipynb
# https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/imputation.ipynb
from momentfm.utils.masking import Masking
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import contingency_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu

device = "cuda" if torch.cuda.is_available() else "cpu"

mask_generator = Masking(mask_ratio=0.25)

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
model.train()


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
        y = torch.stack(ys, dim=0)[:, 0]  # (B, 3) -> (B,)
        return x_flat, mask_flat, y
    return x_flat, mask_flat


train_dataset, eval_dataset = bd.prepare_train_valid_dataset(
    "/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv",
    0.9,
    1,
    [0, 1, 2, 3, 4, 5, 6, 8, 9],
    channel_first=True,
)
g_len = 20
seq_len = 32
in_channel = 4
reduction = "mean"  # 'mean' or 'concat'
# Calculate the sizes for training and validation datasets
batch_size = 1000
train_loader = DataLoader(
    train_dataset,
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
# Optimize Mean Squarred Error using your favourite optimizer
epoch = 5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=epoch * len(train_loader)
)
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

mask_generator = Masking(mask_ratio=0.3)  # Mask 30% of patches randomly


for batch_x, batch_masks, batch_labels in tqdm(train_loader):
    optimizer.zero_grad()
    batch_x = batch_x.to(device)  # [batch_size, n_channels, seq_len]
    batch_masks = batch_masks.to(device)  # [batch_size, seq_len]

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

    # Forward and compute loss
    output = model(x_enc=batch_x, input_mask=batch_masks, reduction=reduction)
    loss = criterion(output.logits, batch_labels)

    print(f"loss: {loss.item()}")

    # Backward
    accelerator.backward(loss)
    # loss.backward()
    optimizer.step()
