from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    "BirdModel": bm.BirdModelWithEmb,
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

# # Convert the DictConfig to a standard dictionary
# cfg_dict = OmegaConf.to_container(cfg, resolve=True)
# import wandb
# wandb.init(project="small-bird", config=cfg_dict)


# =============================
class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        return x_proj, logits


class DistillLoss(nn.Module):
    def __init__(
        self,
        warmup_teacher_temp_epochs,
        nepochs,
        ncrops=2,
        warmup_teacher_temp=0.07,
        teacher_temp=0.04,
        student_temp=0.1,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16') # Nx768
# projector = DINOHead(in_dim=768, out_dim=10, nlayers=3)
# model = nn.Sequential(backbone, projector).cuda()
# images = torch.rand(1, 3, 100, 200)
# student_proj, student_out = model(images.cuda()) # Nx256, Nx10

# cluster_criterion = DistillLoss(30, 200, 2, 0.07, 0.04)
# student_out = torch.randn(6, 10)  # N=3, views=2
# teacher_out = torch.randn(6, 10)  # Example teacher output
# cluster_loss = cluster_criterion(student_out, teacher_out, 1)

# # representation learning, sup
# labels = torch.tensor([5, 5, 4, 5, 4, 4]) #torch.randint(0, 10, (6,))  # Example labels for supervised contrastive loss
# student_proj = torch.randn(6, 1, 256)  # Example student projection
# student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
# sup_con_loss = SupConLoss(contrast_mode="one")(student_proj, labels)


def _caculate_metrics(data, ldts, model, criteria, device):
    labels = ldts
    if ldts.dim() == 2:
        labels = ldts[:, 0]
    labels = labels.to(device)
    data = data.to(device)  # N x C x L

    outputs = model(data)  # N x C
    if len(outputs) == 2:  # for models with embeddings
        embs, outputs = outputs
        embs = embs.unsqueeze(1)  # N x 1 x C
        embs = torch.nn.functional.normalize(embs, dim=-1)
        losses = dict()
        loss = 0
        loss_text = ""
        for key, criterion in criteria.items():
            if key == "cls":
                losses[key] = criterion(outputs, labels)  # 1
            if key == "sup_con":
                losses[key] = criterion(embs, labels)  # 1]
        for key, val in losses.items():
            loss += val
            loss_text += f"{key}: {val.item():.4f}, "
        loss_text += f"Total Loss: {loss.item():.4f}"
        print(f"{loss_text}")
    else:
        loss = criteria(outputs, labels)  # 1

    corrects = (torch.argmax(outputs.data, 1) == labels).sum().item()
    return loss, corrects


def train_one_epoch(
    loader, model, criterion, device, epoch, no_epochs, writer, optimizer
):
    stage = "train"

    model.train()
    running_loss = 0
    running_corrects = 0
    # Fixed: data_len is computed incrementally per batch instead of len(loader.dataset)
    # since drop_last=True can make the total length incorrect
    data_len = 0
    for i, (data, ldts) in enumerate(loader):
        optimizer.zero_grad()

        loss, corrects = _caculate_metrics(data, ldts, model, criterion, device)

        loss.backward()
        optimizer.step()

        running_corrects, running_loss = bm._calculate_batch_stats(
            running_loss, running_corrects, loss, corrects
        )
        data_len += data.shape[0]

    total_loss, total_accuracy = bm._calculate_total_stats(
        running_loss, running_corrects, data_len, i
    )
    bm._print_final(
        epoch, no_epochs, data_len, running_corrects, total_loss, total_accuracy, stage
    )
    bm.write_info_in_tensorboard(writer, epoch, total_loss, total_accuracy, stage)


@torch.no_grad()
def evaluate(loader, model, criterion, device, epoch, no_epochs, writer):
    stage = "valid"

    model.eval()
    running_loss = 0
    running_corrects = 0
    # data_len = len(loader.dataset) is wrong if drop_last=True in the dataloader
    data_len = 0
    for i, (data, ldts) in enumerate(loader):
        loss, corrects = _caculate_metrics(data, ldts, model, criterion, device)

        running_corrects, running_loss = bm._calculate_batch_stats(
            running_loss, running_corrects, loss, corrects
        )
        data_len += data.shape[0]

    total_loss, total_accuracy = bm._calculate_total_stats(
        running_loss, running_corrects, data_len, i
    )
    bm._print_final(
        epoch, no_epochs, data_len, running_corrects, total_loss, total_accuracy, stage
    )
    bm.write_info_in_tensorboard(writer, epoch, total_loss, total_accuracy, stage)
    return total_accuracy


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
    criterion = dict()
    if cfg.use_weighted_loss:
        weights = bd.get_labels_weights(new_label_inds)
        criterion["cls"] = torch.nn.CrossEntropyLoss(torch.tensor(weights).to(device))
    else:
        criterion["cls"] = torch.nn.CrossEntropyLoss()
    criterion["sup_con"] = SupConLoss(contrast_mode="one")

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

            train_one_epoch(
                train_loader,
                model,
                criterion,
                device,
                epoch,
                cfg.no_epochs,
                writer,
                optimizer,
            )
            accuracy = evaluate(
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
            # criterion2=sup_con
        )


# def get_config():
#     return cfg


if __name__ == "__main__":
    # cfg.no_epochs = 2000
    exclude = {}  # {1, 8}
    all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]
    cfg.labels_to_use = sorted(set(all_labels) - set(exclude))
    cfg.model.parameters.out_channels = len(cfg.labels_to_use)
    cfg.exp = "con3"
    main(cfg)
