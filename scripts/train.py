from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import tqdm
from omegaconf import OmegaConf
from torch.utils import tensorboard
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as tvt2

from behavior import data as bd
from behavior import data_augmentation as bau
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.utils import new_label_inds

models = {
    "BirdModel": bm.BirdModel,
    "ResNet18_1D": bm.ResNet18_1D,
    "BirdModelTransformer": bm.BirdModelTransformer,
    "TransformerEncoderMAE": bm1.TransformerEncoderMAE,
    "BirdModelTransformer_": bm.BirdModelTransformer_,
}


@dataclass
class PathConfig:
    save_path: Path


cfg_file = Path(__file__).parents[1] / "configs/train.yaml"
cfg = OmegaConf.load(cfg_file)
cfg_paths = OmegaConf.structured(PathConfig(save_path=cfg.save_path))
cfg = OmegaConf.merge(cfg, cfg_paths)
cfg.min_lr = cfg.max_lr / 10

# Convert the DictConfig to a standard dictionary
cfg_dict = OmegaConf.to_container(cfg, resolve=True)
import wandb

# wandb.init(project="small-bird", config=cfg_dict)


def main(cfg):
    # Set seed and device
    bu.set_seed(cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Use one of the augmentation. transforms.Compose use all the augmentations.
    transforms = tvt2.RandomChoice(
        [
            bau.RandomJitter(sigma=0.05),
            bau.RandomScaling(sigma=0.05),
            # bau.TimeWarp(sigma=0.05),
            # bau.MagnitudeWarp(sigma=0.05, knot=4),
        ]
    )
    transforms = None

    # Prepare datasets
    if cfg.valid_file is not None:
        # The transform is not implemented here.
        train_dataset = bd.get_bird_dataset_from_csv(
            cfg.data_file, cfg.labels_to_use, channel_first=True
        )
        eval_dataset = bd.get_bird_dataset_from_csv(
            cfg.valid_file, cfg.labels_to_use, channel_first=True
        )
    else:
        print("No validation file provided, using train dataset for evaluation.")
        train_dataset, eval_dataset = bd.prepare_train_valid_dataset(
            cfg.data_file,
            cfg.train_per,
            cfg.data_per,
            cfg.labels_to_use,
            channel_first=True,
            transforms=transforms,
        )

    # Build the sampler: inversely weight by class frequency
    labels = train_dataset.ldts[:, 0]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]  # one weight per sample
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    batch_size = len(train_dataset) if cfg.batch_size is None else cfg.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # sampler=sampler,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=len(eval_dataset),
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    # Model setup
    # Number of input channels
    in_channel = train_dataset[0][0].shape[0]  # 3 or 4

    if cfg.model.name not in models:
        raise ValueError(f"Unknown model name: {cfg.model_name}")
    model = models[cfg.model.name](**cfg.model.parameters).to(device)
    print(cfg.model.name)

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
    # """
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
            # if epoch % cfg.save_every == 0:
            #     bm.save_model(cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler)
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # 1-based save for epoch
                bm.save_model(
                    cfg.save_path,
                    cfg.exp,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best=True,
                )
                print(f"Best model accuracy: {best_accuracy:.2f}% at epoch: {epoch}")

    # Save the final model
    # 1-based save for epoch
    # bm.save_model(cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler)
    # """

    bm.load_model(cfg.save_path / f"{cfg.exp}_best.pth", model, device)
    model.eval()
    name = f"{cfg.exp}_{Path(cfg.data_file).stem}"
    fail_path = cfg.save_path / f"failed/{name}"
    fail_path.mkdir(parents=True, exist_ok=True)

    datasets = dict()
    if cfg.valid_file is not None:
        data_files = {
            "train": cfg.data_file,
            "valid": cfg.valid_file,
            "test": cfg.test_file,
        }
        for stage, data_file in data_files.items():
            if data_file is not None:
                # del eval_loader, train_loader, train_dataset, eval_dataset
                dataset = bd.get_bird_dataset_from_csv(
                    data_file, cfg.labels_to_use, channel_first=True
                )
                datasets[stage] = dataset

    else:
        datasets = {"train": train_dataset, "valid": eval_dataset}

    for stage, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
        )
        label_names = [bu.ind2name[i] for i in cfg.labels_to_use]
        data, ldts = next(iter(loader))
        bu.helper_results(
            data,
            ldts,
            model,
            criterion,
            device,
            fail_path,
            label_names,
            len(cfg.labels_to_use),
            stage=stage,
            SAVE_FAILED=False,
        )


def get_config():
    return cfg


if __name__ == "__main__":
    # cfg.no_epochs = 2000
    # exclude = {} #{1, 8}
    # all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]
    # cfg.labels_to_use = sorted(set(all_labels) - set(exclude))
    # cfg.model.parameters.out_channels = len(cfg.labels_to_use)
    # cfg.exp = "tmp"
    main(cfg)


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
