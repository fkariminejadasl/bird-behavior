from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import tqdm
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.utils import n_classes, new_label_inds, target_labels


@dataclass
class Config:
    seed: int = 32984
    save_path: Path = Path("/home/fatemeh/Downloads/bird/result/")
    exp: int = 114  # Experiment number
    no_epochs: int = 4000
    save_every: int = 2000
    train_per: float = 0.9
    data_per: float = 1.0
    # Hyperparameters
    warmup_epochs: int = 1000
    step_size: int = 2000
    max_lr: float = 3e-4
    min_lr: float = None
    weight_decay: float = 1e-2  # Default 1e-2
    # Model parameters
    width: int = 30
    model_name: str = (
        "BirdModel"  # Options: 'BirdModel', 'ResNet18_1D', 'BirdModelTransformer', etc.
    )
    # Criterion parameters
    use_weighted_loss: bool = False  # Whether to use weighted CrossEntropyLoss
    # Optimizer parameters
    optimizer_name: str = "AdamW"
    # Scheduler parameters
    scheduler_name: str = "StepLR"

    def __post_init__(self):
        # Set min_lr based on max_lr
        self.min_lr = self.max_lr / 10


cfg = Config()
# from omegaconf import OmegaConfig
# cfg = OmegaConfig.load(yaml_file)

# import wandb
# wandb.init(project="small-bird", config=cfg)

# Set seed and device
bu.set_seed(cfg.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prepare datasets
train_dataset, eval_dataset = bd.prepare_train_valid_dataset(
    cfg.train_per, cfg.data_per, target_labels
)
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

# Model setup
# Number of input channels
in_channel = train_dataset[0][0].shape[0]  # 3 or 4


# Select model based on configuration
if cfg.model_name == "BirdModel":
    model = bm.BirdModel(in_channel, cfg.width, n_classes).to(device)
elif cfg.model_name == "ResNet18_1D":
    model = bm.ResNet18_1D(n_classes, dropout=0.3).to(device)
elif cfg.model_name == "BirdModelTransformer":
    model = bm.BirdModelTransformer(n_classes, embed_dim=16, drop=0.7).to(device)
elif cfg.model_name == "BirdModelTransformer":
    model = bm1.TransformerEncoderMAE(
        img_size=20,
        in_chans=4,
        out_chans=9,
        embed_dim=16,
        depth=1,
        num_heads=8,
        mlp_ratio=4,
        drop=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ).to(device)
elif cfg.model_name == "BirdModelTransformer_":
    model = bm.BirdModelTransformer_(in_channel, n_classes).to(device)
else:
    raise ValueError(f"Unknown model name: {cfg.model_name}")

# bm.load_model(save_path / f"{exp}_4000.pth", model, device) # start from a checkpoint


# Loss function and optimizer
if cfg.use_weighted_loss:
    weights = bd.get_labels_weights(new_label_inds)
    criterion = torch.nn.CrossEntropyLoss(torch.tensor(weights).to(device))
else:
    criterion = torch.nn.CrossEntropyLoss()

# Select optimizer based on configuration
if cfg.optimizer_name == "AdamW":
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay
    )
else:
    raise ValueError(f"Unknown optimizer name: {cfg.optimizer_name}")

# Select scheduler based on configuration
if cfg.scheduler_name == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.step_size, gamma=0.1
    )
elif cfg.scheduler_name == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.no_epochs, eta_min=cfg.min_lr
    )
elif cfg.scheduler_name == "CosineAnnealingWarmRestarts":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, cfg.warmup_epochs, eta_min=cfg.min_lr
    )
elif cfg.scheduler_name == "SequentialLR":
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1, total_iters=cfg.warmup_epochs
    )
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.no_epochs - cfg.warmup_epochs, eta_min=cfg.min_lr
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lr_scheduler, main_lr_scheduler],
        milestones=[cfg.warmup_epochs],
    )
else:
    raise ValueError(f"Unknown scheduler name: {cfg.scheduler_name}")

# Print dataset sizes and device
len_train, len_eval = len(train_dataset), len(eval_dataset)
print(
    f"Device: {device}, Train samples: {len_train:,}, Validation samples: {len_eval:,}, "
    f"Train loader batches: {len(train_loader)}, Eval loader batches: {len(eval_loader)}"
)

# Training loop
best_accuracy = 0
with tensorboard.SummaryWriter(cfg.save_path / f"tensorboard/{cfg.exp}") as writer:
    for epoch in tqdm.tqdm(range(1, cfg.no_epochs + 1)):
        # tqdm.tqdm(range(4001, no_epochs + 1)): # start from a checkpoint
        start_time = datetime.now()
        print(f"Start time: {start_time}")

        # Train for one epoch
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
        print(f"End time: {end_time}, Elapsed time: {end_time - start_time}")

        # Update scheduler and log learning rates
        scheduler.step()
        lr_optim = round(optimizer.param_groups[-1]["lr"], 6)
        lr_sched = scheduler.get_last_lr()[0]
        writer.add_scalar("lr/optim", lr_optim, epoch)
        writer.add_scalar("lr/sched", lr_sched, epoch)
        print(
            f"Optimizer LR: {optimizer.param_groups[-1]['lr']:.6f}, "
            f"Scheduler LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save model at intervals
        if epoch % cfg.save_every == 0:
            bm.save_model(cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler)
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 1-based save for epoch
            bm.save_model(
                cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler, best=True
            )
            print(f"Best model accuracy: {best_accuracy:.2f}% at epoch: {epoch}")


# Save the final model
# 1-based save for epoch
bm.save_model(cfg.save_path, cfg.exp, model, optimizer, scheduler)

bm.load_model(cfg.save_path / f"{cfg.exp}_best.pth", model, device)
model.eval()
fail_path = cfg.save_path / f"failed/{cfg.exp}"
fail_path.mkdir(parents=True, exist_ok=True)

data, ldts = next(iter(train_loader))
bu.helper_results(
    data,
    ldts,
    model,
    criterion,
    device,
    fail_path,
    bu.target_labels_names,
    n_classes,
    stage="train",
    SAVE_FAILED=False,
)

data, ldts = next(iter(eval_loader))
bu.helper_results(
    data,
    ldts,
    model,
    criterion,
    device,
    fail_path,
    bu.target_labels_names,
    n_classes,
    stage="valid",
    SAVE_FAILED=False,
)

"""
from copy import deepcopy
model = bm.BirdModel(3, 30, 10)
model.load_state_dict(torch.load("/home/fatemeh/test/14_700.pth")["model"])
orig = deepcopy(dict(model.named_parameters()))
'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'fc.weight', 'fc.bias', 'bn.weight', 'bn.bias'


def compare_tensors(orig, other):
    for key in orig.keys():
        if not orig[key].equal(other[key]):
            print(key)

compare_tensors(orig, dict(model.state_dict()))
compare_tensors(orig, dict(model.named_parameters()))

# The difference is in training on the batchnorm buffers (not trained values), bn.running_mean, bn.running_var, bn.num_batches_tracked.

# for unit test (normalizing training data)
# (array([0.45410261, 0.42281342, 0.49202435]), array([0.07290404, 0.04372777, 0.08819486]), array([0., 0., 0.]), array([1., 1., 1.]))
"""

"""
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# https://discuss.pytorch.org/t/register-forward-hook-after-every-n-steps/60923/3
model.requires_grad_(False)
activation = {}
model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))
model.conv3.register_forward_hook(get_activation('conv3'))
model.fc.register_forward_hook(get_activation('fc'))
output = model(data)

mm = activation['conv1'].permute(1,0,2).flatten(1)
fig, axs = plt.subplots(10,1);[axs[i].plot(mm[j], "*") for i,j in enumerate(range(20,30))];plt.show(block=False)
"""
