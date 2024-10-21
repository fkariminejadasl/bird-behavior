# N x C x L = N x 4 x 20

import time
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

from behavior import utils as bu

seed = 1234
bu.set_seed(seed)


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
        # self.conv4 = torch.nn.Conv1d(
        #     mid_channels, mid_channels, kernel_size=3, padding=1
        # )
        self.fc = torch.nn.Linear(mid_channels, out_channels)
        self.bn1 = torch.nn.BatchNorm1d(mid_channels)
        self.bn2 = torch.nn.BatchNorm1d(mid_channels)
        self.bn3 = torch.nn.BatchNorm1d(mid_channels)
        # self.bn4 = torch.nn.BatchNorm1d(mid_channels)
        self.relu = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.25)

        # for m in self.modules():
        #     if isinstance(m, torch.nn.Conv1d) and m.bias is not None:
        #         torch.nn.init.kaiming_normal_(
        #             m.weight, mode="fan_out", nonlinearity="relu"
        #         )
        #     if (
        #         isinstance(m, (torch.nn.BatchNorm1d, torch.nn.Linear))
        #         and m.bias is not None
        #     ):
        #         torch.nn.init.constant_(m.weight, 1)
        #         torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        # identity = x
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        # x = self.relu(self.bn4(self.conv4(x))) + identity
        # x = torch.nn.functional.dropout(x, p=0.25, training=True)
        x = self.avgpool(x).flatten(1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class ResNet1D(nn.Module):
    def __init__(
        self, block, layers, channels=[4, 16, 32, 64], num_classes=10, dropout=0.0
    ):
        super().__init__()
        self.in_channels = channels[1]
        self.conv = nn.Conv1d(
            channels[0], channels[1], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm1d(channels[1])
        self.layer1 = self.make_layer(
            block, channels[1], layers[0], stride=1, dropout=dropout
        )
        self.layer2 = self.make_layer(
            block, channels[2], layers[1], stride=2, dropout=dropout
        )
        self.layer3 = self.make_layer(
            block, channels[3], layers[2], stride=2, dropout=dropout
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[3], num_classes)

    def make_layer(self, block, out_channels, n_layer, stride, dropout):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, dropout))
        self.in_channels = out_channels
        for _ in range(1, n_layer):
            layers.append(block(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18_1D(num_classes, dropout):
    return ResNet1D(ResidualBlock, [2, 2, 2, 2], [4, 16, 32, 64], num_classes, dropout)


from behavior.helpers import (
    EinOpsRearrange,
    IMUPreprocessor,
    LearnableLogitScaling,
    Normalize,
    PatchEmbedGeneric,
    SelectElement,
)
from behavior.transformer import MultiheadAttention, SimpleTransformer


class BirdModelTransformer(nn.Module):
    def __init__(self, out_channels, embed_dim=512, drop=0.7) -> None:
        super().__init__()
        out_embed_dim = out_channels
        embed_dim = embed_dim  # 512
        num_blocks = 1  # 6
        num_heads = 8  # 8
        drop_path = (
            0.0  # drop_path_type == "progressive" and num_blocks=1 is always zero
        )
        drop = drop
        pre_transformer_ln = False
        add_bias_kv = True
        kernel_size = 1  # 8
        img_size = [4, 20]  # [6, 2000]
        in_feature = img_size[0] * kernel_size

        imu_stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=in_feature,
                    out_features=embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=embed_dim),
        )

        self.imu_preprocessor = IMUPreprocessor(
            img_size=img_size,
            num_cls_tokens=1,
            kernel_size=kernel_size,
            embed_dim=embed_dim,
            imu_stem=imu_stem,
        )

        # trunk
        self.simple_transformer = SimpleTransformer(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            drop_path_rate=drop_path,
            ffn_dropout_rate=drop,
            attn_target=partial(
                MultiheadAttention,
                embed_dim=embed_dim,
                num_heads=num_heads,
                bias=True,
                add_bias_kv=add_bias_kv,
                dropout=drop,
            ),
            pre_transformer_layer=nn.Sequential(
                (
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity()
                ),
                EinOpsRearrange("b l d -> l b d"),
            ),
            post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
        )

        # head
        self.head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(embed_dim, out_embed_dim, bias=False),
        )

        # postprocess
        self.postproccess = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )

    def forward(self, x):
        # x = B x 4 x 20
        x = self.imu_preprocessor(x)["trunk"]["tokens"]
        x = self.simple_transformer(x)
        x = self.head(x)
        x = self.postproccess(x)
        return x


from timm.models.vision_transformer import Block, PatchEmbed


# encoder part of MAE
def sincos_pos_embed(embed_dim, max_len, cls_token=False):
    pe = torch.zeros(max_len, embed_dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float()
        * (-torch.log(torch.tensor(10000.0)) / embed_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if cls_token:
        pe = np.concatenate([np.zeros([1, embed_dim]), pe], axis=0)
    return pe


"""
m = BirdModelTransformer()
o = m(torch.rand(2, 4, 20))
optim = torch.optim.SGD(m.parameters(), lr=0.001, momentum=0.9)
# criterion = torch.nn.L1Loss()
# loss = criterion(o, torch.rand(o.shape))
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(o, torch.randint(0, 1, (o.shape[0],)))
optim.zero_grad()
loss.backward()
print(o.shape)
"""

"""
from behavior.helpers import (
    EinOpsRearrange,
    IMUPreprocessor,
    LearnableLogitScaling,
    Normalize,
    PatchEmbedGeneric,
    SelectElement,
)
from behavior.transformer import MultiheadAttention, SimpleTransformer

batch_size = 2
out_embed_dim = 4
embed_dim = 512
num_blocks = 1  # 6
num_heads = 1  # 8
drop_path = 0.7
pre_transformer_ln = False
add_bias_kv = True
kernel_size = 1  # 8
img_size = [4, 20]  # [6, 2000]
in_feature = img_size[0] * kernel_size

# preprocess: token
imu_stem = PatchEmbedGeneric(
    [
        nn.Linear(
            in_features=in_feature,
            out_features=embed_dim,
            bias=False,
        ),
    ],
    norm_layer=nn.LayerNorm(normalized_shape=embed_dim),
)

imu_preprocessor = IMUPreprocessor(
    img_size=img_size,
    num_cls_tokens=1,
    kernel_size=kernel_size,
    embed_dim=embed_dim,
    imu_stem=imu_stem,
)

# trunk
simple_transformer = SimpleTransformer(
    embed_dim=embed_dim,
    num_blocks=num_blocks,
    ffn_dropout_rate=0.0,
    drop_path_rate=drop_path,
    attn_target=partial(
        MultiheadAttention,
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        add_bias_kv=add_bias_kv,
    ),
    pre_transformer_layer=nn.Sequential(
        nn.LayerNorm(embed_dim, eps=1e-6) if pre_transformer_ln else nn.Identity(),
        EinOpsRearrange("b l d -> l b d"),
    ),
    post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
)

# head
head = nn.Sequential(
    nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6),
    SelectElement(index=0),
    nn.Dropout(p=0.5),
    nn.Linear(embed_dim, out_embed_dim, bias=False),
)

# postprocess
postproccess = nn.Sequential(
    Normalize(dim=-1),
    LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
)


# before tokenize_input_and_cls_pos  2 x 250 x 48 after 2 x 251 x 512 (1 for class token)
# ImageBind: input 2 x 6 x 2000 -> token 2 x 251 x 512 -> trunk 2 x 251 x 512 -> head 2 x 4
# Bird data: input 2 x 4 x 20   -> token 2 x 21  x 512 -> trunk 2 x 21  x 512 -> head 2 x 4
# sum([p.numel() for p in list(simple_transformer.parameters()) + \
# list(imu_preprocessor.parameters()) +  list(head.parameters()) if p.requires_grad])
# for ImageBind data 3,311,104, for bird data: 3170816

# batch_size, number of tokens, channel = 2 x 250 x 512
B, T, C = batch_size, int(img_size[1] / kernel_size), embed_dim
o = simple_transformer(torch.rand(B, T, C))
optim = torch.optim.SGD(simple_transformer.parameters(), lr=0.001, momentum=0.9)


o = simple_transformer(imu_preprocessor(torch.rand(B, *img_size))["trunk"]["tokens"])
optim = torch.optim.SGD(
    list(simple_transformer.parameters()) + list(imu_preprocessor.parameters()),
    lr=0.001,
    momentum=0.9,
)

# B, *img_size = 2 x 6 x 2000
o = head(
    simple_transformer(imu_preprocessor(torch.rand(B, *img_size))["trunk"]["tokens"])
)
optim = torch.optim.SGD(
    list(simple_transformer.parameters())
    + list(imu_preprocessor.parameters())
    + list(head.parameters()),
    lr=0.001,
    momentum=0.9,
)

o = postproccess(
    head(
        simple_transformer(
            imu_preprocessor(torch.rand(B, *img_size))["trunk"]["tokens"]
        )
    )
)
optim = torch.optim.SGD(
    list(simple_transformer.parameters())
    + list(imu_preprocessor.parameters())
    + list(head.parameters()),
    lr=0.001,
    momentum=0.9,
)

criterion = torch.nn.L1Loss()
loss = criterion(o, torch.rand(o.shape))
loss.backward()
print(o.shape)
"""


class BirdModelTransformer_(nn.Module):
    # bug: Trying to backward through the graph a second time
    # specify .backward(retain_graph=True), doesn't solve the problem
    # I don't see which part has this problem
    def __init__(self, in_channels=4, out_channels=10) -> None:
        super().__init__()

        self.upsample = torch.nn.ConvTranspose2d(in_channels, 3, 1, stride=(10, 1))
        self.model = VisionTransformer(
            img_size=14,
            patch_size=1,
            embed_dim=384,
            depth=1,
            num_heads=1,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        )
        self.model.head = torch.nn.Linear(
            in_features=384, out_features=out_channels, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.upsample(x, output_size=torch.Size([196, 1])).reshape(-1, 3, 14, 14)
        x = self.model(x)
        return x


# criterion = torch.nn.L1Loss()
# m = BirdModelTransformer_(4, 4)
# o = m(torch.rand(2, 4, 20))
# optim = torch.optim.SGD(m.parameters(), lr=0.001, momentum=0.9)
# loss = criterion(o, torch.rand(o.shape))
# # backpropagation RuntimeError: Trying to backward through the graph a second time
# # loss.backward()

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


def _caculate_metrics(data, ldts, model, criterion, device):
    labels = ldts  # ugly
    if ldts.dim() == 2:
        labels = ldts[:, 0]
    labels = labels.to(device)
    data = data.to(device)  # N x C x L
    outputs = model(data)  # N x C
    loss = criterion(outputs, labels)  # 1
    corrects = (torch.argmax(outputs.data, 1) == labels).sum().item()
    return loss, corrects


# def train_one_epoch(
#     loader, model, criterion, device, epoch, no_epochs, writer, optimizer
# ):
#     stage = "train"
#     data_len, batch_size, no_batches, start_time = _count_data(loader)

#     model.train()
#     running_loss = 0
#     running_corrects = 0
#     for i, (data, labels) in enumerate(loader):
#         optimizer.zero_grad()

#         loss, corrects = _caculate_metrics(data, labels, model, criterion, device)

#         loss.backward()
#         optimizer.step()

#         running_corrects, running_loss = _calculate_batch_stats(
#             running_loss, running_corrects, loss, corrects
#         )

#         # daata = data.permute(0, 2, 1).reshape(-1, data.shape[1])
#         # print(
#         #     f"batch: {i}, data: {data.shape}, min: {daata.min(0)}, max: {daata.max(0)}"
#         # )
#         # start_time = _print_per_batch(
#         #     i, batch_size, no_batches, start_time, loss, corrects, stage
#         # )

#     total_loss, total_accuracy = _calculate_total_stats(
#         running_loss, running_corrects, data_len, i
#     )
#     _print_final(
#         epoch, no_epochs, data_len, running_corrects, total_loss, total_accuracy, stage
#     )
#     write_info_in_tensorboard(writer, epoch, total_loss, total_accuracy, stage)


# @torch.no_grad()
# def evaluate(loader, model, criterion, device, epoch, no_epochs, writer):
#     stage = "valid"
#     data_len, batch_size, no_batches, start_time = _count_data(loader)

#     model.eval()
#     running_loss = 0
#     running_corrects = 0
#     for i, (data, labels) in enumerate(loader):
#         loss, corrects = _caculate_metrics(data, labels, model, criterion, device)

#         running_corrects, running_loss = _calculate_batch_stats(
#             running_loss, running_corrects, loss, corrects
#         )

#         # daata = data.permute(0, 2, 1).reshape(-1, data.shape[1])
#         # print(
#         #     f"batch: {i}, data: {data.shape}, min: {daata.min(0)}, max: {daata.max(0)}"
#         # )
#         # start_time = _print_per_batch(
#         #     i, batch_size, no_batches, start_time, loss, corrects, stage
#         # )

#     total_loss, total_accuracy = _calculate_total_stats(
#         running_loss, running_corrects, data_len, i
#     )
#     _print_final(
#         epoch, no_epochs, data_len, running_corrects, total_loss, total_accuracy, stage
#     )
#     write_info_in_tensorboard(writer, epoch, total_loss, total_accuracy, stage)
#     return total_accuracy


def train_one_epoch(
    loader, model, criterion, device, epoch, no_epochs, writer, optimizer
):
    stage = "train"
    data_len, batch_size, no_batches, start_time = _count_data(loader)

    model.train()
    running_loss = 0
    running_corrects = 0
    for i, (data, ldts) in enumerate(loader):
        optimizer.zero_grad()

        loss, corrects = _caculate_metrics(data, ldts, model, criterion, device)

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
    for i, (data, ldts) in enumerate(loader):
        loss, corrects = _caculate_metrics(data, ldts, model, criterion, device)

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
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)["model"]
    )
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


# TODO: This part should be removed later.
# ======
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from behavior import data as bd
from behavior import model as bm
from behavior import utils as bu
from behavior.utils import n_classes, target_labels, target_labels_names


def temp_model_test(data_path, model_path):
    width = 30

    seed = 1234
    bu.set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    measurements, ldts = bd.load_csv(data_path)
    measurements, ldts = bd.get_specific_labesl(measurements, ldts, target_labels)

    dataset = bd.BirdDataset(measurements, ldts)
    data_loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )

    in_channel = next(iter(data_loader))[0].shape[1]
    model = bm.BirdModel(in_channel, width, n_classes).to(device)
    bm.load_model(model_path, model, device)
    model.eval()

    data, ldts = next(iter(data_loader))
    with torch.no_grad():
        labels = ldts[:, 0]
        labels = labels.to(device)  # N
        data = data.to(device)  # N x C x L
        outputs = model(data)  # N x C
        prob = torch.nn.functional.softmax(outputs, dim=-1).detach()  # N x C
        pred = torch.argmax(outputs.data, 1)  # N

    pred = pred.item()  # 1d. otherwise pred.cpu().numpy()
    return pred
