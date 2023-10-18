import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, average_precision_score


from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
import tqdm
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from typing import Iterable, Tuple

from behavior import data as bd
from behavior import model as bm

# import wandb
# wandb.init(project="uncategorized")

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()

    num_classes = len(class_names)
    plt.xticks(np.arange(num_classes), class_names, rotation=45)
    plt.yticks(np.arange(num_classes), class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(confusion_matrix[i, j]), horizontalalignment="center", color="black" if i == j else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def precision_recall(
    labels: Iterable[float], predictions: Iterable[float]
) -> Tuple[float, float, float]:
    """Compute precision, recall and f-score from given labels and predictions"""
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (recall * precision) / (recall + precision) * 100
    return precision, recall, f_score




ind2name = {0: 'Flap', 1: 'ExFlap', 2: 'Soar', 3: 'Boat', 4: 'Float', 5: 'SitStand', 6: 'TerLoco', 7: 'Other', 8: 'Manouvre', 9: 'Pecking'}

save_path = Path("/home/fatemeh/Downloads/bird/result/")
train_per = 0.9
data_per = .2
exp = 40  # sys.argv[1]
target_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# target_labels = [0, 2, 3, 4, 5, 6] # no: Exflap:1, Other:7, Manauvre:8, Pecking:9
# target_labels = [0, 3, 4, 5, 6] # no: Exflap:1, Soar:2, Other:7, Manauvre:8, Pecking:9 
# target_labels = [0, 2, 4, 5]
target_labels_names = [ind2name[t] for t in target_labels]
n_classes = len(target_labels)

# train_set, tmp.json
train_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data/train_set.json")
valid_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data/validation_set.json")
test_path = Path("/home/fatemeh/Downloads/bird/bird/set1/data/test_set.json")

all_measurements, label_ids = bd.combine_all_data()
all_measurements, label_ids = bd.get_specific_labesl(
    all_measurements, label_ids, target_labels
)
n_trainings = int(all_measurements.shape[0] * train_per * data_per)
n_valid = int(all_measurements.shape[0] * data_per) - n_trainings
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


print(f"data shape: {train_dataset[0][0].shape}")  # 3x20
in_channel = train_dataset[0][0].shape[0]  # 3 or 4
model = bm.BirdModel(in_channel, 30, n_classes).to(device)
model.eval()
bm.load_model(save_path / f"{exp}_best.pth", model, device)


def helper_results(data, labels):
    labels = labels.to(device)
    data = data.to(device)  # N x C x L
    outputs = model(data)  # N x C
    pred = torch.argmax(outputs.data, 1)
    # loss and accuracy
    loss = criterion(outputs, labels)  # 1
    corrects = (pred == labels).sum().item()
    accuracy = corrects/len(labels) * 100
    # confusion matrix
    confmat = confusion_matrix(labels.cpu().numpy(), pred.cpu().numpy())
    plot_confusion_matrix(confmat, target_labels_names);plt.show(block=False)
    plot_confusion_matrix(confmat, target_labels);plt.show(block=False)
    # average precision
    prob = torch.nn.functional.softmax(outputs, dim=-1).detach()
    ap = average_precision_score(labels.cpu().numpy(), prob.cpu().numpy())
    print(ap, loss.item(), accuracy)
    print(confmat)

print(device)

data, label = next(iter(train_loader))
helper_results(data, label)

data, label = next(iter(eval_loader))
helper_results(data, label)


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