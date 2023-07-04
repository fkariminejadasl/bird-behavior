# N x C x L = N x 4 x 20

import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)


class BirdModel(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels, mid_channels, kernel_size=3, padding=1
        )
        self.conv2 = torch.nn.Conv1d(
            mid_channels, mid_channels, kernel_size=3, padding=1
        )
        self.conv3 = torch.nn.Conv1d(
            mid_channels, mid_channels, kernel_size=3, padding=1
        )
        self.fc = torch.nn.Linear(mid_channels, out_channels)
        # self.bn = torch.nn.BatchNorm1d(mid_channels)
        self.bn = torch.nn.BatchNorm1d(mid_channels, track_running_stats=False)
        self.relu = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        x = self.relu(self.bn(self.conv3(x)))
        # x = self.relu(self.conv1(x))
        # x = self.conv1(x)
        x = self.avgpool(x).flatten(1)
        # x = self.relu(self.fc(x))
        x = self.fc(x)
        return x


class BirdModel_(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(
            in_channels, mid_channels, kernel_size=3, padding=1
        )
        self.conv2 = torch.nn.Conv1d(
            mid_channels, mid_channels, kernel_size=3, padding=1
        )
        self.conv3 = torch.nn.Conv1d(
            mid_channels, mid_channels, kernel_size=3, padding=1
        )
        self.fc = torch.nn.Linear(mid_channels, out_channels)
        self.bn = torch.nn.BatchNorm1d(mid_channels)
        self.relu = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) and m.bias is not None:
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            if (
                isinstance(m, (torch.nn.BatchNorm1d, torch.nn.Linear))
                and m.bias is not None
            ):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        x = self.relu(self.bn(self.conv3(x)))
        x = self.avgpool(x).flatten(1)
        x = self.fc(x)
        return x


"""
# a trick to use vision transformer for a sequence of 20x4 (20 size, 4 dim)
# 1x4x20->1x3x14x14 by convtranspose2d followed by VisionTransformer(img_size=14, patch_size=1, ...
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial

x = torch.rand(1,4,20).reshape(1,4,20,1)
upsample = torch.nn.ConvTranspose2d(4, 3, 1, stride=(10,1))
model = VisionTransformer(img_size=14, patch_size=1, embed_dim=384, depth=1, num_heads=1, mlp_ratio=4, qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
model.head = torch.nn.Linear(in_features=384, out_features=10, bias=True)
model(upsample(x, output_size=torch.Size([196, 1])).reshape(1,3,14,14))

This differs from ImageBind patchfiy
https://github.com/facebookresearch/ImageBind/blob/main/models/multimodal_preprocessors.py#L667
>>> x = torch.rand(1, 6, 2000)
>>> m = torch.nn.Linear(48,512)
>>> l = torch.nn.LayerNorm(normalized_shape=512)
>>> l(m(x.unfold(-1,8,8).permute(0,2,1,3).reshape(1,250,-1))).shape
torch.Size([1, 250, 512])
>>> x.unfold(-1,8,8).shape
torch.Size([1, 6, 250, 8])
>>> x.unfold(-1,8,8).permute(0,2,1,3).reshape(1,250,-1).shape
torch.Size([1, 250, 48])
"""


# since there is one model, I put all engine stuff here, not to spreed the stuff.


def write_info_in_tensorboard(writer, epoch, loss, accuracy, stage):
    loss_scalar_dict = dict()
    acc_scalar_dict = dict()
    loss_scalar_dict[stage] = loss
    acc_scalar_dict[stage] = accuracy
    writer.add_scalars("loss", loss_scalar_dict, epoch)
    writer.add_scalars("accuracy", acc_scalar_dict, epoch)


def _count_data(loader):
    data_len = len(loader.dataset)
    batch_size = next(iter(loader))[-1].shape[0]
    no_batches = int(np.ceil(data_len / batch_size))
    start_time = time.time()
    return data_len, batch_size, no_batches, start_time


def _print_per_batch(
    batch_ind, batch_size, no_batches, start_time, loss, corrects, stage, save_every=100
):
    if batch_ind % save_every == 0 and batch_ind != 0:
        print(f"batch: {batch_ind}, time: {time.time()-start_time:.1f}s")
        print(
            f"{stage}: batch/total: {batch_ind}/{no_batches}, total loss: {loss.item():.4f}, \
                accuracy: {corrects * 100/batch_size:.2f}, no. correct: {corrects}, bs:{batch_size}"
        )
        start_time = time.time()
    return start_time


def _print_final(
    epoch, no_epochs, data_len, running_corrects, total_loss, total_accuracy, stage
):
    print(
        f"{stage}: epoch/total: {epoch}/{no_epochs}, total loss: {total_loss:.4f}, accuracy: {total_accuracy:.2f}, \
        no. correct: {running_corrects}, length data:{data_len}"
    )


def _calculate_batch_stats(running_loss, running_corrects, loss, corrects):
    running_corrects += corrects
    running_loss += loss.item()
    return running_corrects, running_loss


def _calculate_total_stats(running_loss, running_corrects, data_len, i):
    total_loss = running_loss / (i + 1)
    total_accuracy = running_corrects / data_len * 100
    return total_loss, total_accuracy


def _caculate_metrics(data, labels, model, criterion, device):
    labels = labels.to(device)
    data = data.to(device)  # N x C x L
    outputs = model(data)  # N x C
    loss = criterion(outputs, labels)  # 1
    corrects = (torch.argmax(outputs.data, 1) == labels).sum().item()
    return loss, corrects


def train_one_epoch(
    loader, model, criterion, device, epoch, no_epochs, writer, optimizer
):
    stage = "train"
    data_len, batch_size, no_batches, start_time = _count_data(loader)

    model.train()
    running_loss = 0
    running_corrects = 0
    for i, (data, labels) in enumerate(loader):
        optimizer.zero_grad()

        loss, corrects = _caculate_metrics(data, labels, model, criterion, device)

        loss.backward()
        optimizer.step()

        running_corrects, running_loss = _calculate_batch_stats(
            running_loss, running_corrects, loss, corrects
        )

        # daata = data.permute(0, 2, 1).reshape(-1, data.shape[1])
        # print(
        #     f"batch: {i}, data: {data.shape}, min: {daata.min(0)}, max: {daata.max(0)}"
        # )
        # start_time = _print_per_batch(
        #     i, batch_size, no_batches, start_time, loss, corrects, stage
        # )

    total_loss, total_accuracy = _calculate_total_stats(
        running_loss, running_corrects, data_len, i
    )
    _print_final(
        epoch, no_epochs, data_len, running_corrects, total_loss, total_accuracy, stage
    )
    write_info_in_tensorboard(writer, epoch, total_loss, total_accuracy, stage)


@torch.no_grad()
def evaluate(loader, model, criterion, device, epoch, no_epochs, writer):
    stage = "valid"
    data_len, batch_size, no_batches, start_time = _count_data(loader)

    model.eval()
    running_loss = 0
    running_corrects = 0
    for i, (data, labels) in enumerate(loader):
        loss, corrects = _caculate_metrics(data, labels, model, criterion, device)

        running_corrects, running_loss = _calculate_batch_stats(
            running_loss, running_corrects, loss, corrects
        )

        # daata = data.permute(0, 2, 1).reshape(-1, data.shape[1])
        # print(
        #     f"batch: {i}, data: {data.shape}, min: {daata.min(0)}, max: {daata.max(0)}"
        # )
        # start_time = _print_per_batch(
        #     i, batch_size, no_batches, start_time, loss, corrects, stage
        # )

    total_loss, total_accuracy = _calculate_total_stats(
        running_loss, running_corrects, data_len, i
    )
    _print_final(
        epoch, no_epochs, data_len, running_corrects, total_loss, total_accuracy, stage
    )
    write_info_in_tensorboard(writer, epoch, total_loss, total_accuracy, stage)
    return total_accuracy


def load_model(checkpoint_path, model, device) -> None:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model"])
    return model


def save_model(
    checkpoint_path, exp, epoch, model, optimizer, scheduler, best=False
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "date": datetime.now().isoformat(),
    }
    name = f"{exp}_{epoch}.pth"
    if best:
        name = f"{exp}_best.pth"
    torch.save(checkpoint, checkpoint_path / name)
