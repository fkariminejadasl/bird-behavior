import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from momentfm import MOMENTPipeline
from momentfm.data.classification_dataset import ClassificationDataset
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import contingency_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu

"""
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "embedding"},
    # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
)

# # model(torch.rand(10,4,16)).forecast: [10, 4, 192], no embeddings
# model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large", model_kwargs={"task_name": "forecasting", 'seq_len':16, 'forecast_horizon': 192, 'head_dropout': 0.1, 'weight_decay': 0, 'freeze_encoder': True, 'freeze_embedder': True, 'freeze_head': False}, local_files_only=True)
# # embeddings: [bach_size, seq_len//8, 1024*num_classes] e.g [10, 2, 4096] for model(torch.rand(10,4,16))
# model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large", model_kwargs={"task_name": "classification", "n_channels": 4, "num_class": 9}, local_files_only=True)
# # embeddings: [10 x 1024]
# model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large", model_kwargs={"task_name": "embedding"}, local_files_only=True)
# # reconstruction: [10, 4, 16] for model(torch.rand(10,4,16)), no embeddings
# model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large", model_kwargs={"task_name": "reconstruction"}, local_files_only=True)

model.init()
model.eval()
device = "cuda"
model = model.to(device)

all_measurements, label_ids = bd.load_csv("/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv")
all_measurements, label_ids = bd.get_specific_labesl(
    all_measurements, label_ids, bu.target_labels
)
dataset = bd.BirdDataset(all_measurements, label_ids, channel_first=True)
dataloader = DataLoader(
    dataset, batch_size=len(dataset), shuffle=False, num_workers=1, drop_last=False
)

data, ldts = next(iter(dataloader)) # [4338, 4, 20], [4338, 3]
labels = ldts[:, 0]
data = data[:, :, :16]
# batch_size = 100
# labels = ldts[:batch_size, 0]
# data = torch.nn.functional.pad(data[:batch_size], (0, 512 - 20), mode='constant', value=0)
with torch.no_grad():
    output = model(data.to(device))
embeddings = output.embeddings.cpu().numpy() # [4338, 1024]

reducer = TSNE(n_components=2, random_state=42)
reduced = reducer.fit_transform(embeddings)

true_labels = [bu.ind2name[i] for i in bu.target_labels]
unique_classes = np.unique(labels)
bounds = np.concatenate((unique_classes-0.5, [unique_classes[-1]+0.5]))
norm   = mpl.colors.BoundaryNorm(bounds, ncolors=len(unique_classes))

plt.figure()
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab20", s=5, norm=norm)
cbar = plt.colorbar(scatter, label="Label")
cbar.set_ticks(unique_classes)
cbar.set_ticklabels(true_labels)
print("Done")
"""


# # MiniBatchKMeans
# kmeans = MiniBatchKMeans(9)
# kmeans.partial_fit(embeddings)
# preds = kmeans.predict(embeddings)
# # KMeans
# kmeans = KMeans(9)
# embeddings = output.embeddings.detach().cpu().numpy()
# kmeans.fit(embeddings)
# preds = kmeans.predict(embeddings)
# # DBSCAN
# dbscan = DBSCAN()
# preds = dbscan.fit_predict(embeddings)

# # Semi supervised KMeans from GCD
# cut_class = 3  # 5
# uf = l_feats[l_targets >= cut_class]
# ut = l_targets[l_targets >= cut_class]
# lf = l_feats[l_targets < cut_class]
# lt = l_targets[l_targets < cut_class]
# # fmt: off
# kmeans = K_Means(k=cfg.out_channel, tolerance=1e-4, max_iterations=100, n_init=3, random_state=10, pairwise_batch_size=8192)
# # fmt: off
# kmeans.fit_mix(uf, lf, lt)
# preds = kmeans.labels_.cpu().numpy()
# assert np.all(preds[:lt.shape[0]] == lt.cpu().numpy()) == True
# # 1092 1092 2184 # balanced data
# print(ut.shape[0], lt.shape[0], l_feats.shape[0])

# ordered_feat = np.concatenate((lf.cpu(),uf.cpu()), axis=0)
# ordered_labels = np.concatenate((lt.cpu(), ut.cpu()))
# cm = contingency_matrix(ordered_labels, preds)

# cm = contingency_matrix(labels, preds)
# bu.plot_confusion_matrix(cm)

"""
x = torch.randn(1, 4, 16)  # [batch_size, n_channels, seq_len]
output = model(x_enc=x)
logits = output.logits
predicted_labels = logits.argmax(dim=1)  # [batch_size, ]


def get_embedding(model, dataloader):
    embeddings, labels = [], []
    with torch.no_grad():
        for batch_x, batch_masks, batch_labels in tqdm(
            dataloader, total=len(dataloader)
        ):
            batch_x = batch_x.to("cuda").float()
            batch_masks = batch_masks.to("cuda")

            output = model(
                x_enc=batch_x, input_mask=batch_masks
            )  # [batch_size x d_model (=1024)]
            embedding = output.embeddings
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(batch_labels)

    embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
    return embeddings, labels

train_dataset = ClassificationDataset(data_split="train")
test_dataset = ClassificationDataset(data_split="test")
train_dataloader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, drop_last=False
)
test_dataloader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, drop_last=False
)

model.to("cuda").float()
train_embeddings, train_labels = get_embedding(model, train_dataloader)
test_embeddings, test_labels = get_embedding(model, test_dataloader)


# Define a data loader
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for data, labels in train_dataloader:
    # forward [batch_size, n_channels, forecast_horizon]
    output = model(x_enc=data)

    # backward
    loss = criterion(output.logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"loss: {loss.item():.3f}")
"""

import numpy as np
import torch
from momentfm.models.statistical_classifiers import fit_svm
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_embeddings(model, device, reduction, dataloader: DataLoader):
    """
    labels: [num_samples]
    embeddings: [num_samples x d_model]
    """
    embeddings, labels = [], []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_x, batch_labels in tqdm(dataloader, total=len(dataloader)):
            # [batch_size x 12 x 512]
            batch_x = batch_x.to(device).float()
            # [batch_size x num_patches x d_model (=1024)]
            output = model(x_enc=batch_x, reduction=reduction)
            # mean over patches dimension, [batch_size x d_model]
            embedding = output.embeddings.mean(dim=1)
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(batch_labels)

    embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
    return embeddings, labels


def train_epoch(
    model, device, train_dataloader, criterion, optimizer, scheduler, reduction="mean"
):
    """
    Train only classification head
    """
    model.to(device)
    model.train()
    losses = []

    # for batch_x, batch_labels in train_dataloader:
    for _ in train_dataloader:
        optimizer.zero_grad()
        # batch_x = batch_x.to(device).float()
        # batch_labels = batch_labels.to(device)
        batch_x = torch.rand(
            (16, 1, 512), device=device, dtype=torch.float32
        )  # [batch_size, n_channels, seq_len]
        batch_labels = torch.randint(0, 5, (16,), device=device, dtype=torch.long)

        # note that since MOMENT encoder is based on T5, it might experiences numerical unstable issue with float16
        with torch.autocast(
            device_type="cuda",
            dtype=(
                torch.bfloat16
                if torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 8
                else torch.float32
            ),
        ):
            output = model(x_enc=batch_x, reduction=reduction)
            loss = criterion(output.logits, batch_labels)
        loss.backward()

        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    avg_loss = np.mean(losses)
    return avg_loss


def evaluate_epoch(dataloader, model, criterion, device, phase="val", reduction="mean"):
    model.eval()
    model.to(device)
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for batch_x, batch_labels in dataloader:
            batch_x = batch_x.to(device).float()
            batch_labels = batch_labels.to(device)

            output = model(x_enc=batch_x, reduction=reduction)
            loss = criterion(output.logits, batch_labels)
            total_loss += loss.item()
            total_correct += (output.logits.argmax(dim=1) == batch_labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy


# train_dataset = ClassificationDataset(data_split='train')
# test_dataset = ClassificationDataset(data_split='test')
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
train_loader = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        "task_name": "classification",
        "n_channels": 1,  # 12, # number of input channels
        "num_class": 5,
        "freeze_encoder": True,  # Freeze the patch embedding layer
        "freeze_embedder": True,  # Freeze the transformer encoder
        "freeze_head": False,  # The linear forecasting head must be trained
        ## NOTE: Disable gradient checkpointing to supress the warning when linear probing the model as MOMENT encoder is frozen
        "enable_gradient_checkpointing": False,
        # Choose how embedding is obtained from the model: One of ['mean', 'concat']
        # Multi-channel embeddings are obtained by either averaging or concatenating patch embeddings
        # along the channel dimension. 'concat' results in embeddings of size (n_channels * d_model),
        # while 'mean' results in embeddings of size (d_model)
        "reduction": "mean",
    },
    # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
)
model.init()


epoch = 5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=epoch * len(train_loader)
)
device = "cuda"

for i in tqdm(range(epoch)):
    train_loss = train_epoch(
        model, device, train_loader, criterion, optimizer, scheduler
    )
    print(f"Epoch {i}, train loss: {train_loss}")
    # train_loss = train_epoch(model, device, train_loader, criterion, optimizer, scheduler)
    # val_loss, val_accuracy = evaluate_epoch(test_loader, model, criterion, device, phase='test')
    # print(f'Epoch {i}, train loss: {train_loss}, val loss: {val_loss}, val accuracy: {val_accuracy}')

# test_loss, test_accuracy = evaluate_epoch(test_loader, model, criterion, device, phase='test')
# print(f'Test loss: {test_loss}, test accuracy: {test_accuracy}')
