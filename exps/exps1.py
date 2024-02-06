from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from behavior import data as bd
from behavior import model as bm
from behavior import prepare as bp
from behavior import utils as bu

# import wandb
# wandb.init(project="uncategorized")

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)


ind2name = {
    0: "Flap",
    1: "ExFlap",
    2: "Soar",
    3: "Boat",
    4: "Float",
    5: "SitStand",
    6: "TerLoco",
    7: "Other",
    8: "Manouvre",
    9: "Pecking",
}

save_path = Path("/home/fatemeh/Downloads/bird/result/")
train_per = 0.9
data_per = 1
exp = 77  # sys.argv[1]
save_name = f"{exp}"
width = 30
# target_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
target_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # no Other
# target_labels = [0, 2, 3, 4, 5, 6] # no: Exflap:1, Other:7, Manauvre:8, Pecking:9
# target_labels = [0, 3, 4, 5, 6]  # no: Exflap:1, Soar:2, Other:7, Manauvre:8, Pecking:9
# target_labels = [0, 2, 4, 5]
# target_labels = [8, 9]
# target_labels = [0, 1, 2, 3, 4, 5, 6, 9]  # no Other:7; combine soar:2 and manuver:8
target_labels_names = [ind2name[t] for t in target_labels]
n_classes = len(target_labels)
fail_path = save_path / f"failed/{save_name}"
fail_path.mkdir(parents=True, exist_ok=True)

data_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data")
combined_file = data_path / "combined.json"

all_measurements, label_ids = bp.load_csv("/home/fatemeh/Downloads/bird/data/set1.csv")
# all_measurements, label_ids = bd.combine_all_data(combined_file)
# label_ids = bd.combine_specific_labesl(label_ids, [2, 8])
all_measurements, label_ids = bd.get_specific_labesl(
    all_measurements, label_ids, target_labels
)
# make data shorter
label_ids = np.repeat(label_ids, 2, axis=0)
all_measurements = all_measurements.reshape(-1, 10, 4)

n_trainings = int(all_measurements.shape[0] * train_per * data_per)
n_valid = all_measurements.shape[0] - n_trainings
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

# train_dataset = bd.BirdDataset(all_measurements, label_ids)
# eval_dataset = deepcopy(train_dataset)
train_dataset = bd.BirdDataset(train_measurments, train_labels)
eval_dataset = bd.BirdDataset(valid_measurements, valid_labels)

# train_dataset = bd.BirdDataset_old(train_path)
train_loader = DataLoader(
    train_dataset,
    batch_size=len(train_dataset),
    shuffle=True,
    num_workers=1,
    drop_last=True,
)
# eval_dataset = bd.BirdDataset_old(valid_path)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=len(eval_dataset),
    shuffle=False,
    num_workers=1,
    drop_last=True,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()


print(f"data shape: {eval_dataset[0][0].shape}")  # 3x20
in_channel = eval_dataset[0][0].shape[0]  # 3 or 4
model = bm.BirdModel(in_channel, width, n_classes).to(device)
model.eval()
bm.load_model(save_path / f"{exp}_best.pth", model, device)

print(device)

data, ldts = next(iter(train_loader))
bu.helper_results(
    data,
    ldts,
    model,
    criterion,
    device,
    fail_path,
    target_labels,
    target_labels_names,
    n_classes,
    stage="train",
    SAVE_FAILED=True,
)

data, ldts = next(iter(eval_loader))
bu.helper_results(
    data,
    ldts,
    model,
    criterion,
    device,
    fail_path,
    target_labels,
    target_labels_names,
    n_classes,
    stage="valid",
    SAVE_FAILED=False,
)


print(sum([p.numel() for p in model.parameters()]))

# bad classes: Other, Exflap (less data), Pecking (noisy), Manuver/Mix

# for i in range(len(target_labels)):
#     inds = torch.where(labels == 3)[0]
#     sel_labels = labels[inds]
#     sel_prob = prob[inds]
#     average_precision_score(sel_labels.cpu().numpy(), sel_prob.cpu().numpy())
#     average_precision_score(    labels.cpu().numpy(),     prob.cpu().numpy())


# y_true = np.array([0, 0, 1, 1, 2, 2])
# y_scores = np.array([
#     [0.7, 0.2, 0.1],
#     [0.4, 0.3, 0.3],
#     [0.1, 0.8, 0.1],
#     [0.2, 0.3, 0.5],
#     [0.4, 0.4, 0.2],
#     [0.1, 0.2, 0.7],
# ])
# average_precision_score(y_true, y_scores)
