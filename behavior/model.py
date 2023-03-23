# N x C x L = N x 4 x 20

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


# since there is one model, I put all engine stuff here, not to spreed the stuff.


def write_info_in_tensorboard(writer, epoch, loss, accuracy, stage):
    loss_scalar_dict = dict()
    acc_scalar_dict = dict()
    loss_scalar_dict[stage] = loss
    acc_scalar_dict[stage] = accuracy
    writer.add_scalars("loss", loss_scalar_dict, epoch)
    writer.add_scalars("accuracy", acc_scalar_dict, epoch)


def train_one_epoch(loader, model, criterion, device, epoch, writer, optimizer):
    model.train()
    running_loss = 0
    running_accuracy = 0
    for i, (data, labels) in enumerate(loader):
        optimizer.zero_grad()

        data = (
            data.squeeze(1).type(torch.float32).to(device)
        )  # dataloader change 1xCxL to Nx1xCxL
        outputs = model(data)  # N x C

        labels = labels.to(device)
        loss = criterion(outputs, labels)  # 1
        loss.backward()
        optimizer.step()

        accuracy = (torch.argmax(outputs.data, 1) == labels).sum().item()
        running_accuracy += accuracy
        running_loss += loss.item()

        # batch_size = len(labels)
        # print(
        #     f"train: epoch: {epoch}, total loss: {loss.item()}, accuracy: {accuracy * 100/batch_size}, no. correct: {accuracy}, bs:{batch_size}"
        # )

    total_loss = running_loss / (i + 1)
    total_accuracy = running_accuracy / len(loader.dataset) * 100
    print(
        f"train: epoch: {epoch}, total loss: {total_loss:.4f}, accuracy: {total_accuracy:.2f}, no. correct: {running_accuracy}, length data:{len(loader.dataset)}"
    )
    write_info_in_tensorboard(writer, epoch, total_loss, total_accuracy, stage="train")


@torch.no_grad()
def evaluate(loader, model, criterion, device, epoch, writer):
    model.eval()
    running_loss = 0
    running_accuracy = 0
    for i, (data, labels) in enumerate(loader):
        data = (
            data.squeeze(1).type(torch.float32).to(device)
        )  # dataloader change 1xCxL to Nx1xCxL
        outputs = model(data)  # N x C

        labels = labels.to(device)

        loss = criterion(outputs, labels)  # 1

        accuracy = (torch.argmax(outputs.data, 1) == labels).sum().item()
        running_accuracy += accuracy
        running_loss += loss.item()

        # batch_size = len(labels)
        # print(
        #     f"eval: epoch: {epoch}, total loss: {loss.item()}, accuracy: {accuracy * 100/batch_size}, no. correct: {accuracy}, bs:{batch_size}"
        # )

    total_loss = running_loss / (i + 1)
    total_accuracy = running_accuracy / len(loader.dataset) * 100
    print(
        f"train: epoch: {epoch}, total loss: {total_loss:.4f}, accuracy: {total_accuracy:.2f}, no. correct: {running_accuracy}, length data:{len(loader.dataset)}"
    )
    write_info_in_tensorboard(writer, epoch, total_loss, total_accuracy, stage="valid")


def load_model(checkpoint_path, model, device) -> None:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model"])
    return model


def save_model(checkpoint_path, exp, epoch, model, optimizer, scheduler) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "date": datetime.now().isoformat(),
    }
    torch.save(checkpoint, checkpoint_path / f"{exp}_{epoch}.pth")
