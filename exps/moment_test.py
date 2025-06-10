import numpy as np
import torch
import torch.nn as nn
from momentfm import MOMENTPipeline
from momentfm.data.classification_dataset import ClassificationDataset
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.metrics.cluster import contingency_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "embedding"},
    local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
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

device = "cuda"
all_measurements, label_ids = bd.load_csv(
    "/home/fatemeh/Downloads/bird/data/final/balance.csv"  # corrected_combined_unique_sorted012.csv"
)
all_measurements, label_ids = bd.get_specific_labesl(
    all_measurements, label_ids, bu.target_labels
)
dataset = bd.BirdDataset(all_measurements, label_ids, channel_first=True)
dataloader = DataLoader(
    dataset, batch_size=len(dataset), shuffle=False, num_workers=1, drop_last=False
)

# embeddings, labels = [], []
# with torch.no_grad():
#     for data, ldts in tqdm(dataloader, total=len(dataloader)): # [4694, 4, 20], [4694, 3]
#         data = data[:, :, :16]
#         data = data.to(device)
#         labels = ldts[:, 0].cpu().numpy()
#         output = model(x_enc=data) # [batch_size x d_model (=1024)]
#         embedding = output.embeddings
#         embeddings.append(embedding.detach().cpu().numpy())
#         labels.append(labels)
# embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)


data, ldts = next(iter(dataloader))
labels = ldts[:, 0].cpu().numpy()
output = model(data)
embeddings = output.embeddings.detach().cpu().numpy()

# MiniBatchKMeans
kmeans = MiniBatchKMeans(9)
kmeans.partial_fit(embeddings)
preds = kmeans.predict(embeddings)
# KMeans
kmeans = KMeans(9)
embeddings = output.embeddings.detach().cpu().numpy()
kmeans.fit(embeddings)
preds = kmeans.predict(embeddings)
# DBSCAN
dbscan = DBSCAN()
preds = dbscan.fit_predict(embeddings)
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


cm = contingency_matrix(labels, preds)
bu.plot_confusion_matrix(cm)

false_neg = cm.sum(axis=1) - cm.max(axis=1)
# fmt: off
print(sum(false_neg), cm.sum() - sum(false_neg), cm.sum(), (cm.sum() - sum(false_neg)) / cm.sum())
# fmt: on

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
