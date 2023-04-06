from datetime import datetime
from pathlib import Path

import torch
import torchvision
import tqdm
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from behavior import data as bd
from behavior import model as bm

"""
# quick model test
labels, label_ids, device_ids, time_stamps, all_measurements = bd.read_data(al.json_path)
x = torch.from_numpy(all_measurements).type(torch.float32).permute(0, 2, 1)
x = torch.zeros((1402, 4, 20), dtype=torch.float32)
model = bm.BirdModel(4, 30, 10)
model(x)
"""

save_path = Path("/home/fatemeh/test")
exp = 1  # sys.argv[1]
no_epochs = 1000  # int(sys.argv[2])

train_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data/train_set.json")
valid_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data/validation_set.json")
test_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data/test_set.json")


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ]
)

train_dataset = bd.BirdDataset(train_path, transform)
train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=False, num_workers=1, drop_last=False
)
eval_dataset = bd.BirdDataset(valid_path, transform)
eval_loader = DataLoader(
    eval_dataset, batch_size=8, shuffle=False, num_workers=1, drop_last=False
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = bm.BirdModel(4, 30, 10).to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.1)

len_train, len_eval = len(train_dataset), len(eval_dataset)
print(f"device: {device}, train: {len_train:,}, valid: {len_eval:,} images")
best_accuracy = 0
with tensorboard.SummaryWriter(save_path / f"tensorboard/{exp}") as writer:
    for epoch in tqdm.tqdm(range(1, no_epochs + 1)):
        start_time = datetime.now()
        print(f"start time: {start_time}")
        bm.train_one_epoch(
            train_loader, model, criterion, device, epoch, no_epochs, writer, optimizer
        )
        # scheduler.step()
        accuracy = bm.evaluate(
            eval_loader, model, criterion, device, epoch, no_epochs, writer
        )
        end_time = datetime.now()
        print(f"end time: {end_time}, elapse time: {end_time-start_time}")

        # save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 1-based save for epoch
            bm.save_model(save_path, exp, epoch, model, optimizer, scheduler, best=True)
            print(f"best model accuracy: {best_accuracy:.2f} at epoch: {epoch}")

# 1-based save for epoch
bm.save_model(save_path, exp, epoch, model, optimizer, scheduler)
