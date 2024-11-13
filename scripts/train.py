from datetime import datetime
from pathlib import Path

import torch
import tqdm
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.utils import n_classes, target_labels

# import wandb
# wandb.init(project="uncategorized")

seed = 32984
save_path = Path("/home/fatemeh/Downloads/bird/result/")
exp = 114  # sys.argv[1]
no_epochs = 4000  # int(sys.argv[2])
save_every = 2000
train_per = 0.9
data_per = 1.0
# hyperparam
warmup_epochs = 1000
step_size = 2000
max_lr = 3e-4  # 1e-3
min_lr = max_lr / 10
weight_decay = 1e-2  # default 1e-2
# model
width = 30


bu.set_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset, eval_dataset = bd.prepare_train_valid_dataset(
    train_per, data_per, target_labels
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


"""
torchvision.transforms.ToTensor() changes the CxL to 1xCxL and 
dataloader change 1xCxL to Nx1xCxL
I don't use ToTensor anymore. I put everything now in dataset instead of model.
"""

in_channel = train_dataset[0][0].shape[0]  # 3 or 4
model = bm.BirdModel(in_channel, width, n_classes).to(device)
# model = bm.ResNet18_1D(n_classes, dropout=0.3).to(device)
# model = bm.BirdModelTransformer(n_classes, embed_dim=16, drop=0.7).to(device)
# model = bm1.TransformerEncoderMAE(
#     img_size=20,
#     in_chans=4,
#     out_chans=9,
#     embed_dim=16,
#     depth=1,
#     num_heads=8,
#     mlp_ratio=4,
#     drop=0.0,
#     norm_layer=partial(nn.LayerNorm, eps=1e-6),
# ).to(device)

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

bm.load_model(save_path / f"{exp}_best.pth", model, device)
model.eval()
fail_path = save_path / f"failed/{exp}"
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
