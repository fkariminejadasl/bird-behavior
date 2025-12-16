"""
Non contrastive learning: DINO / iBOT / SimSiam / BYOL / SwAV / VICReg / BarlowTwins
Contrastive learning: SimCLR / MoCo

This script is used to train non-contrastive self-supervised learning models.
Other names can be non contrastive / self distillation / alignmet: train_self_distill.py, train_non_contrastive_ssl.py  

Based on: https://github.com/CVMI-Lab/SimGCD/blob/main/train_mp.py
"""

import gc
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
from omegaconf import OmegaConf
from scipy import io as mat_io
from torch.nn.functional import normalize
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets.folder import default_loader
from torchvision.transforms import v2 as tvt2

from behavior import data as bd
from behavior import data_augmentation as bau
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.utils import new_label_inds

cifar_10_root = "/home/fatemeh/Downloads/data"


def inference(data, model, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        data = data.to(device)  # N x C x L
        outputs = model(data)  # N x C
        # prob = torch.nn.functional.softmax(outputs, dim=-1)  # N x C
        pred = torch.argmax(outputs.data, 1)  # N

    prob = outputs.cpu()  # .numpy()
    pred = pred.cpu().numpy()
    return prob, pred


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


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


def info_nce_logits(features, n_views=2, temperature=1.0, device="cuda"):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


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


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def train(
    student, train_loader, optimizer, scaler, scheduler, cluster_criterion, epoch, args
):
    # loss_record = AverageMeter()

    student.train()
    for batch_idx, batch in enumerate(train_loader):
        images, class_labels, uq_idxs, mask_lab = batch
        mask_lab = mask_lab[:, 0]

        class_labels, mask_lab = (
            class_labels.cuda(non_blocking=True),
            mask_lab.cuda(non_blocking=True).bool(),
        )
        images = torch.cat(images, dim=0).cuda(non_blocking=True)

        with torch.amp.autocast(scaler is not None):
            student_proj, student_out = student(images)
            teacher_out = student_out.detach()

            # # clustering, sup
            # sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            # sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
            # cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            # # clustering, unsup
            # cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            # avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            # me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            # cluster_loss += args.memax_weight * me_max_loss

            # represent learning, unsup
            contrastive_logits, contrastive_labels = info_nce_logits(
                features=student_proj
            )
            contrastive_loss = torch.nn.CrossEntropyLoss()(
                contrastive_logits, contrastive_labels
            )

            # # representation learning, sup
            # student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            # student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
            # sup_con_labels = class_labels[mask_lab]
            # sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

            pstr = ""
            # pstr += f'cls_loss: {cls_loss.item():.4f} '
            # pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            # pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
            pstr += f"contrastive_loss: {contrastive_loss.item():.4f} "

            loss = 0

        # Train acc
        # loss_record.update(loss.item(), class_labels.size(0))
        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # if batch_idx % args.print_freq == 0 and dist.get_rank() == 0:
        #     args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
        #                 .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
    # Step schedule
    scheduler.step()

    # if dist.get_rank() == 0:
    #     args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))


def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(
                mask,
                np.array(
                    [
                        True if x.item() in range(len(args.train_classes)) else False
                        for x in label
                    ]
                ),
            )

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    # all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
    #                                                 T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
    #                                                 args=args)

    # return all_acc, old_acc, new_acc


def get_transform(transform_type="imagenet", image_size=32, args=None):

    if transform_type == "imagenet":

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        train_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size / crop_pct), interpolation),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size / crop_pct), interpolation),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    else:

        raise NotImplementedError

    return (train_transform, test_transform)


def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(
        range(len(dataset)),
        replace=False,
        size=(int(prop_indices_to_subsample * len(dataset)),),
    )

    return subsample_indices


class MergedDataset(Dataset):
    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[
                item - len(self.labelled_dataset)
            ]
            labeled_or_not = 0

        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)


class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):

        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(
            cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),)
        )
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_cifar_10_datasets(
    train_transform,
    test_transform,
    train_classes=(0, 1, 8, 9),
    prop_train_labels=0.8,
    split_train_val=False,
    seed=0,
):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCIFAR10(
        root=cifar_10_root, transform=train_transform, train=True
    )

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(
        deepcopy(whole_training_set), include_classes=train_classes
    )
    subsample_indices = subsample_instances(
        train_dataset_labelled, prop_indices_to_subsample=prop_train_labels
    )
    train_dataset_labelled = subsample_dataset(
        train_dataset_labelled, subsample_indices
    )

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(
        deepcopy(train_dataset_labelled), train_idxs
    )
    val_dataset_labelled_split = subsample_dataset(
        deepcopy(train_dataset_labelled), val_idxs
    )
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(
        train_dataset_labelled.uq_idxs
    )
    train_dataset_unlabelled = subsample_dataset(
        deepcopy(whole_training_set), np.array(list(unlabelled_indices))
    )

    # Get test set for all classes
    test_dataset = CustomCIFAR10(
        root=cifar_10_root, transform=test_transform, train=False
    )

    # Either split train into train and val or use test set as val
    train_dataset_labelled = (
        train_dataset_labelled_split if split_train_val else train_dataset_labelled
    )
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        "train_labelled": train_dataset_labelled,
        "train_unlabelled": train_dataset_unlabelled,
        "val": val_dataset_labelled,
        "test": test_dataset,
    }

    return all_datasets


get_dataset_funcs = {
    "cifar10": get_cifar_10_datasets,
}


def get_datasets(dataset_name, train_transform, test_transform, args):
    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(
        train_transform=train_transform,
        test_transform=test_transform,
        train_classes=args.train_classes,
        prop_train_labels=args.prop_train_labels,
        split_train_val=False,
    )
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(
        labelled_dataset=deepcopy(datasets["train_labelled"]),
        unlabelled_dataset=deepcopy(datasets["train_unlabelled"]),
    )

    test_dataset = datasets["test"]
    unlabelled_train_examples_test = deepcopy(datasets["train_unlabelled"])
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


# Test losses
# --------------------
"""
image_size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
interpolation = 3
crop_pct = 0.875
train_transform = transforms.Compose(
    [
        transforms.Resize(int(image_size / crop_pct), interpolation),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
)
train_transform = ContrastiveLearningViewGenerator(
    base_transform=train_transform, n_views=2
)
trainset = torchvision.datasets.CIFAR10(
    root="/home/fatemeh/Downloads/data", train=True, download=True, transform=train_transform
)
loader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=1, drop_last=True
)
a = next(iter(loader))

feat_dim = 768
num_mlp_layers = 3
num_clusters = mlp_out_dim = 10
batch_size = 8
device = "cuda"

# embeddings for contrastive loss
student_proj = torch.randn(batch_size, feat_dim, device=device)  
student_out = torch.randn(batch_size, num_clusters, device=device)  # logits for student
teacher_out = torch.randn(batch_size, num_clusters, device=device)  # logits for teacher
class_labels = torch.randint(0, num_clusters, (batch_size // 2,), device=device)

backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
projector = DINOHead(in_dim=feat_dim, out_dim=mlp_out_dim, nlayers=num_mlp_layers)
model = nn.Sequential(backbone, projector).to(device)

# clustering, sup
# student_proj, student_out = student(images)
sup_logits = torch.cat([f for f in (student_out / 0.1).chunk(2)], dim=0)
sup_labels = torch.cat([class_labels for _ in range(2)], dim=0)
cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

# clustering, unsup
cluster_criterion = DistillLoss(30, 200, 2)
cluster_loss = cluster_criterion(student_out, teacher_out, epoch=1)
avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
me_max_loss = -torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(
    float(len(avg_probs))
)
memax_weight = 2
cluster_loss += memax_weight * me_max_loss

# represent learning, unsup
contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

# representation learning, sup
student_proj = torch.cat([f.unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
sup_con_labels = class_labels
sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)
"""

"""
# Check data and mask_lab
# --------------------
args = OmegaConf.create()
# args.image_size = 224
args.feat_dim = 768
args.num_mlp_layers = 3
args.image_size = 32
args.train_classes = list(range(5))
args.unlabeled_classes = list(range(5, 10))
args.num_labeled_classes = len(args.train_classes)
args.num_unlabeled_classes = len(args.unlabeled_classes)
args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
args.n_views = 2
args.dataset_name = "cifar10"
args.num_workers = 1
args.interpolation = 3
args.crop_pct = 0.875
args.transform = "imagenet"
args.prop_train_labels = 0.5
args.batch_size = 128
train_transform, test_transform = get_transform(
    args.transform, image_size=args.image_size, args=args
)
train_transform = ContrastiveLearningViewGenerator(
    base_transform=train_transform, n_views=args.n_views
)
# --------------------
# DATASETS
# --------------------
train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
    args.dataset_name, train_transform, test_transform, args
)

label_len = len(train_dataset.labelled_dataset)
unlabelled_len = len(train_dataset.unlabelled_dataset)
sample_weights = [
    1 if i < label_len else label_len / unlabelled_len
    for i in range(len(train_dataset))
]
sample_weights = torch.DoubleTensor(sample_weights)
train_sampler = torch.utils.data.WeightedRandomSampler(
    sample_weights, num_samples=len(train_dataset)
)
train_loader = DataLoader(
    train_dataset,
    num_workers=args.num_workers,
    batch_size=args.batch_size,
    shuffle=False,
    sampler=train_sampler,
    drop_last=True,
    pin_memory=True,
)

feat_dim = 768
num_mlp_layers = 3
num_clusters = mlp_out_dim = 10
batch_size = 8
device = "cuda"
backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
projector = DINOHead(in_dim=feat_dim, out_dim=mlp_out_dim, nlayers=num_mlp_layers)
student = nn.Sequential(backbone, projector).to(device)
student.train()
for batch_idx, batch in enumerate(train_loader):
    images, class_labels, uq_idxs, mask_lab = batch
    mask_lab = mask_lab[:, 0]
    print(images[0].shape)
    # represent learning, unsup
    class_labels, mask_lab = (
        class_labels.cuda(non_blocking=True),
        mask_lab.cuda(non_blocking=True).bool(),
    )
    images = torch.cat(images, dim=0).cuda(non_blocking=True)
    student_proj, student_out = student(images)
    contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
    contrastive_loss = torch.nn.CrossEntropyLoss()(
        contrastive_logits, contrastive_labels
    )
    break
"""

# Test on my own data
# --------------------
train_transform = tvt2.RandomChoice(
    [
        bau.RandomJitter(sigma=0.05),
        bau.RandomScaling(sigma=0.05),
        # bau.TimeWarp(sigma=0.05),
        # bau.MagnitudeWarp(sigma=0.05, knot=4),
    ]
)
train_transform = ContrastiveLearningViewGenerator(
    base_transform=train_transform, n_views=2
)

cfg = dict(
    test_data_file=Path("/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv"),
    model_checkpoint=Path("/home/fatemeh/Downloads/bird/result/125_best.pth"),
    labels_to_use=[0, 1, 2, 3, 4, 5, 6, 8, 9],
    channel_first=False,
    in_channel=4,
    mid_channel=30,
    out_channel=9,
    seed=123,
    batch_size=4338,
)
cfg = OmegaConf.create(cfg)

bu.set_seed(cfg.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(cfg.test_data_file, header=None)
df = df[df[3].isin(cfg.labels_to_use)]
mapping = {l: i for i, l in enumerate(cfg.labels_to_use)}
df[3] = df[3].map(mapping)
all_measurements = df[[4, 5, 6, 7]].values.reshape(-1, 20, 4)
label_ids = df[[3, 0, 0]].iloc[::20].values
trainset = bd.BirdDataset(
    all_measurements,
    label_ids,
    channel_first=cfg.channel_first,
    transform=train_transform,
)
loader = torch.utils.data.DataLoader(
    trainset, batch_size=len(trainset), shuffle=False, num_workers=4, drop_last=False
)
a = next(iter(loader))
model = bm.BirdModel(cfg.in_channel, 30, cfg.out_channel)
bm.load_model(cfg.model_checkpoint, model, device)

b1 = inference(a[0][0].transpose(2, 1), model, device)[0][3]  # 8
b2 = inference(a[0][1].transpose(2, 1), model, device)[0][3]  # 8
b2 = inference(a[0][0].transpose(2, 1), model, device)[0][4]  # 6
b2 = inference(a[0][0].transpose(2, 1), model, device)[0][5]  # 8


logits0, preds0 = inference(a[0][0].transpose(2, 1), model, device)
logits1, preds1 = inference(a[0][1].transpose(2, 1), model, device)
idxs6 = np.where(label_ids == 6)[0]
idxs8 = np.where(label_ids == 8)[0]
idxs = idxs6
b1 = logits0
b2 = logits1
vals = []
for i in range(len(idxs)):
    val = normalize(b1[idxs[i]], dim=-1, p=2).dot(normalize(b2[idxs[i]], dim=-1, p=2))
    val = round(val.item(), 2)
    vals.append(val)
# preds[idxs[np.argsort(vals)]]
# np.array(vals)[np.argsort(vals)]
cosim = torch.nn.functional.cosine_similarity(b1[idxs], b2[idxs], dim=1)
sidxs = cosim.sort()[1].numpy()
preds0[idxs[sidxs]]
preds1[idxs[sidxs]]
cosim[sidxs]
print("Done")

"""
bu.plot_one(df.iloc[:20, 4:7].values)
bu.plot_one(a[0][0][0])
bu.plot_one(a[0][1][0])
a[0][0].shape # torch.Size([64, 20, 4])
np.array(df.iloc[::20,3])[:10] # equal label_ids[:10,0]
"""

"""
image_size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
interpolation = 3
crop_pct = 0.875
train_transform = transforms.Compose([
    transforms.Resize(int(image_size / crop_pct), interpolation),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=torch.tensor(mean),
        std=torch.tensor(std))
])
train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=2)
trainset = torchvision.datasets.CIFAR10(root='/home/fatemeh/data', train=True,download=True, transform=train_transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
a = next(iter(loader))
"""
