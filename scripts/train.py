from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm
from torch.utils import tensorboard
from torch.utils.data import DataLoader, random_split

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1

# import wandb
# wandb.init(project="uncategorized")

# There are more into reproducibility:
# https://pytorch.org/docs/stable/notes/randomness.html
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # for multiple gpu
generator = torch.Generator().manual_seed(seed)  # for random_split
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

"""
# quick model test
labels, label_ids, device_ids, time_stamps, all_measurements = bd.read_data(bd.json_path)
x = torch.from_numpy(all_measurements).type(torch.float32).permute(0, 2, 1)
x = torch.zeros((1402, 4, 20), dtype=torch.float32)
model = bm.BirdModel(4, 30, 10)
model(x)
"""


save_path = Path("/home/fatemeh/Downloads/bird/result/")
exp = 105  # sys.argv[1]
no_epochs = 2000  # int(sys.argv[2])
save_every = 2000
train_per = 0.9
data_per = 1.0
# target_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
target_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # no Other
# target_labels = [0, 2, 3, 4, 5, 6] # no: Exflap:1, Other:7, Manauvre:8, Pecking:9
# target_labels = [0, 3, 4, 5, 6]  # no: Exflap:1, Soar:2, Other:7, Manauvre:8, Pecking:9
# target_labels = [0, 2, 4, 5]
# target_labels = [8, 9]
# target_labels = [0, 1, 2, 3, 4, 5, 6, 9]  # no Other:7; combine soar:2 and manuver:8
n_classes = len(target_labels)
# hyperparam
warmup_epochs = 1000
step_size = 2000
max_lr = 3e-4  # 1e-3
min_lr = max_lr / 10
weight_decay = 1e-2  # default 1e-2
# model
width = 30

# """
# data_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data")
# combined_file = data_path / "combined.json"
# all_measurements, label_ids = bd.combine_all_data(combined_file)
all_measurements, label_ids = bd.load_csv(
    "/home/fatemeh/Downloads/bird/data/combined_s_w_m_j.csv"
)
# label_ids = bd.combine_specific_labesl(label_ids, [2, 8])
all_measurements, label_ids = bd.get_specific_labesl(
    all_measurements, label_ids, target_labels
)
# make data shorter
# label_ids = np.repeat(label_ids, 2, axis=0)
# all_measurements = all_measurements.reshape(-1, 10, 4)

# all = 4365
n_trainings = 100  # (10% data)# int(all_measurements.shape[0] * train_per * data_per)
n_valid = 100  # all_measurements.shape[0] - n_trainings
train_measurments = all_measurements[:n_trainings]
valid_measurements = all_measurements[n_trainings : n_trainings + n_valid]
train_labels, valid_labels = (
    label_ids[:n_trainings],
    label_ids[n_trainings : n_trainings + n_valid],
)
print(
    len(train_labels),
    len(valid_labels),
    train_measurments.shape,
    valid_measurements.shape,
)
train_dataset = bd.BirdDataset(train_measurments, train_labels)
eval_dataset = bd.BirdDataset(valid_measurements, valid_labels)

# ind_data = int(data_per * len(all_measurements))
# all_measurements, label_ids = all_measurements[:ind_data], label_ids[:ind_data]
# dataset = bd.BirdDataset(all_measurements, label_ids)
# train_size = int(train_per * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, eval_dataset = random_split(dataset, [train_size, val_size], generator)

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

"""
csv_files = Path("/home/fatemeh/Downloads/bird/test_data/split_200").glob("part*")
csv_files = sorted(csv_files, key=lambda x: int(x.stem.split("_")[1]))
csv_files = [str(csv_file) for csv_file in csv_files]

# csv_files = Path("/home/fatemeh/Downloads/bird/test_data/split_600").glob("part*")
# dataset = bd.BirdDataset2(
#     csv_files, "/home/fatemeh/Downloads/bird/test_data/group_counts.json", group_size=20
# )
dataset = bd.BirdDataset3(csv_files)
# Calculate the sizes for training and validation datasets
train_size = int(train_per * data_per * len(dataset))
val_size = len(dataset) - train_size

# Use random_split to divide the dataset
train_dataset, eval_dataset = random_split(dataset, [train_size, val_size])

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
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
torchvision.transforms.ToTensor() changes the CxL to 1xCxL and 
dataloader change 1xCxL to Nx1xCxL
I don't use ToTensor anymore. I put everything now in dataset instead of model.
"""

print(f"data shape: {train_dataset[0][0].shape}")  # 3x20
in_channel = train_dataset[0][0].shape[0]  # 3 or 4
# model = bm.BirdModel(in_channel, width, n_classes).to(device)
# model = bm.ResNet18_1D(n_classes, dropout=0.3).to(device)
# model = bm.BirdModelTransformer(n_classes, embed_dim=16, drop=0.7).to(device)
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

# model = bm.BirdModelTransformer_(in_channel, n_classes).to(device)
# bm.load_model(save_path / f"{exp}_4000.pth", model, device) # start from a checkpoint

# weights = bd.get_labels_weights(label_ids)
# criterion = torch.nn.CrossEntropyLoss(torch.tensor(weights).to(device))
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(
#     filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9
# )
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=no_epochs, eta_min=min_lr
# )
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, warmup_epochs, eta_min=min_lr
# )
# warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
#     optimizer, start_factor=0.1, end_factor=1, total_iters=warmup_epochs
# )
# main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=no_epochs - warmup_epochs, eta_min=min_lr
# )
# scheduler = torch.optim.lr_scheduler.SequentialLR(
#     optimizer,
#     schedulers=[warmup_lr_scheduler, main_lr_scheduler],
#     milestones=[warmup_epochs],
# )

len_train, len_eval = len(train_dataset), len(eval_dataset)
print(
    f"device: {device}, train: {len_train:,}, valid: {len_eval:,} \
    images, train_loader: {len(train_loader)}, eval_loader: {len(eval_loader)}"
)
best_accuracy = 0
with tensorboard.SummaryWriter(save_path / f"tensorboard/{exp}") as writer:
    for epoch in tqdm.tqdm(range(1, no_epochs + 1)):
        # tqdm.tqdm(range(4001, no_epochs + 1)): # start from a checkpoint
        start_time = datetime.now()
        print(f"start time: {start_time}")
        bm.train_one_epoch(
            train_loader, model, criterion, device, epoch, no_epochs, writer, optimizer
        )
        accuracy = bm.evaluate(
            eval_loader, model, criterion, device, epoch, no_epochs, writer
        )
        end_time = datetime.now()
        print(f"end time: {end_time}, elapse time: {end_time-start_time}")

        scheduler.step()
        lr_optim = round(optimizer.param_groups[-1]["lr"], 6)
        lr_sched = scheduler.get_last_lr()[0]
        writer.add_scalar("lr/optim", lr_optim, epoch)
        writer.add_scalar("lr/sched", lr_sched, epoch)
        print(
            f"optim: {optimizer.param_groups[-1]['lr']:.6f}, sched: {scheduler.get_last_lr()[0]:.6f}"
        )

        if epoch % save_every == 0:
            bm.save_model(save_path, exp, epoch, model, optimizer, scheduler)
        # save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 1-based save for epoch
            bm.save_model(save_path, exp, epoch, model, optimizer, scheduler, best=True)
            print(f"best model accuracy: {best_accuracy:.2f} at epoch: {epoch}")

# 1-based save for epoch
bm.save_model(save_path, exp, epoch, model, optimizer, scheduler)


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
