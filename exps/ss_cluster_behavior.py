import gc
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap  # umap-learn
from omegaconf import ListConfig, OmegaConf
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.ss_kmeans import K_Means
from behavior.utils import n_classes, new_label_inds, target_labels


def read_csv_files(data_path):
    gimus = []
    # dis = []
    # timestamps = []
    for csv_file in data_path.glob("*.csv"):
        df = pd.read_csv(csv_file, header=None)
        gimus.append(df[[4, 5, 6, 7]].values)
        # dis.append(df[[0, 2]].values)
        # timestamps.extend(df[1].tolist())

    gimus = np.concatenate(gimus, axis=0)
    # dis = np.concatenate(dis, axis=0)
    # timestamps = np.array(timestamps)
    return gimus


@dataclass
class PathConfig:
    save_path: Path
    data_file: Path
    model_checkpoint: Path


cfg_file = Path(__file__).parents[1] / "configs/cluster.yaml"
cfg = OmegaConf.load(cfg_file)
cfg_paths = OmegaConf.structured(
    PathConfig(
        save_path=cfg.save_path,
        data_file=cfg.data_file,
        model_checkpoint=cfg.model_checkpoint,
    )
)
cfg = OmegaConf.merge(cfg, cfg_paths)

# Read hyperparameters from the config file
n_clusters_list = (
    cfg.n_clusters if isinstance(cfg.n_clusters, ListConfig) else [cfg.n_clusters]
)
batch_sizes_list = (
    cfg.batch_size if isinstance(cfg.batch_size, ListConfig) else [cfg.batch_size]
)

# General loads
bu.set_seed(cfg.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfg.save_path.mkdir(parents=True, exist_ok=True)


def setup_training_dataloader(cfg, batch_size):
    # Load data
    # all_measurements, label_ids = bd.load_csv(cfg.data_file)
    # all_measurements, label_ids = bd.get_specific_labesl(
    #     all_measurements, label_ids, bu.target_labels
    # )
    # dataset = bd.BirdDataset(all_measurements, label_ids, channel_first=cfg.channel_first)
    gimus = read_csv_files(cfg.data_file)
    print(gimus.shape)
    gimus = gimus.reshape(-1, cfg.g_len, cfg.in_channel)
    gimus = np.ascontiguousarray(gimus)
    print(gimus.shape)
    dataset = bd.BirdDataset(gimus, channel_first=cfg.channel_first)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # len(dataset), # 4096
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    del gimus
    gc.collect()
    return loader


def setup_testing_dataloader(cfg):
    # Load data
    all_measurements, label_ids = bd.load_csv(cfg.test_data_file)
    all_measurements, label_ids = bd.get_specific_labesl(
        all_measurements, label_ids, cfg.labels_to_use
    )
    dataset = bd.BirdDataset(
        all_measurements, label_ids, channel_first=cfg.channel_first
    )
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),  # 4096
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    print(all_measurements.shape)
    del all_measurements, label_ids
    gc.collect()
    return loader


def get_activation(activation):
    """Create a hook function that updates the given activation list."""

    def hook(model, input, output):
        activation.append(output.detach())

    return hook


def save_unlabeled_embeddings(loader, model, layer_to_hook, device):
    """
    collect all embeddings.
    Returns:
      embeddings: torch.Tensor (size [total, embed_dim])
      labels: torch.LongTensor (size [total])
    """
    start_time = time.time()

    activation = []

    hook_handle = layer_to_hook.register_forward_hook(get_activation(activation))

    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            print(i)
            # if i == 30:
            #     break
            data = data.to(device)
            _ = model(data)
            if activation:
                feats = activation.pop()  # shape (B, embed_dim, 1)
                feats = feats.flatten(1)  # shape (B, embed_dim)
                # feats = feats[:, 0, :] # shape (B, L, embed_dim) -> (B, embed_dim) # for norm
                # torch.save(feats.detach().cpu().numpy(), cfg.save_path / f"{i}.npy")
                feats = feats.cpu().numpy()
                np.savez(cfg.save_path / f"{i}", **{"feats": feats})
            del data, feats
            torch.cuda.empty_cache()

    hook_handle.remove()

    end_time = time.time()
    print(f"Train embeddings are loaded in {end_time - start_time:.2f} seconds.")


def save_labeled_embeddings(loader, model, layer_to_hook, device):
    """
    save all embeddings and their labels.
    Returns:
      embeddings: torch.Tensor (size [total_labeled, embed_dim])
    """
    start_time = time.time()

    activation = []

    hook_handle = layer_to_hook.register_forward_hook(get_activation(activation))

    with torch.no_grad():
        for data, ldts in tqdm(loader):
            data = data.to(device)
            output = model(data)
            if activation:
                feats = activation.pop()  # shape (B, embed_dim, 1)
                feats = feats.flatten(1)  # shape (B, embed_dim)
                # feats = feats[:, 0, :] # shape (B, L, embed_dim) -> (B, embed_dim) # for norm
                labels = ldts[:, 0].cpu().numpy()
                feats = feats.detach().cpu().numpy()
                ouput = output.detach().cpu().numpy()
                # torch.save({"feats": feats, "labels": labels}, cfg.save_path / "test.npy")
                np.savez(
                    cfg.save_path / "test",
                    **{"feats": feats, "labels": labels, "output": ouput},
                )
            del data, feats, labels
            torch.cuda.empty_cache()

    hook_handle.remove()

    end_time = time.time()
    print(f"Test embeddings are loaded in {end_time - start_time:.2f} seconds.")


def load_labeled_embeddings(file_name):
    # Load test embeddings
    results = np.load(cfg.save_path / file_name)
    feats = results["feats"]  # N x D
    labels = results["labels"]  # N
    output = results["output"]
    print(feats.shape)
    return feats, labels, output


def load_unlabeled_embeddings():
    n = len(list(cfg.save_path.glob("*.npz"))) - 1
    # Load train embeddings
    u_feats = []
    for i in range(n):
        # feats = torch.load(cfg.save_path/f"{i}.npy")
        feats = np.load(cfg.save_path / f"{i}.npz")["feats"]
        u_feats.append(feats)
    u_feats = np.concatenate(u_feats, axis=0)  # N x D
    print(u_feats.shape)
    return u_feats


def collect_unlabeled_embeddings(loader, model, layer_to_hook, device):
    """
    collect all embeddings in memory.
    Returns:
      embeddings: torch.Tensor (size [total, embed_dim])
      labels: torch.LongTensor (size [total])
    """
    start_time = time.time()

    activation = []

    hook_handle = layer_to_hook.register_forward_hook(get_activation(activation))

    embeddings_list = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            print(i)
            # if i == 30:
            #     break
            data = data.to(device)
            _ = model(data)
            if activation:
                feats = activation.pop()  # shape (B, 1, embed_dim)
                feats = feats[:, 0, :]  # shape (B, embed_dim)
                embeddings_list.append(feats)
            del data, feats
            torch.cuda.empty_cache()

    hook_handle.remove()

    embeddings = torch.cat(embeddings_list, dim=0)  # shape (N, D)

    end_time = time.time()
    print(f"Train embeddings are loaded in {end_time - start_time:.2f} seconds.")
    print(f"Embedding {embeddings.shape}")

    return embeddings


def collect_labeled_embeddings(loader, model, layer_to_hook, device):
    """
    collect all embeddings and their labels in memory.
    Returns:
      embeddings: torch.Tensor (size [total_labeled, embed_dim])
    """
    start_time = time.time()

    activation = []

    hook_handle = layer_to_hook.register_forward_hook(get_activation(activation))

    embeddings_list = []
    label_list = []

    with torch.no_grad():
        for data, ldts in tqdm(loader):
            data = data.to(device)
            _ = model(data)
            if activation:
                feats = activation.pop()  # shape (B, 1, embed_dim)
                feats = feats[:, 0, :]  # shape (B, embed_dim)
                embeddings_list.append(feats)
                label_list.append(ldts[:, 0])

    hook_handle.remove()

    embeddings = torch.cat(embeddings_list, dim=0)  # shape (N, D)
    labels = torch.cat(label_list, dim=0)  # shape (N,)

    end_time = time.time()
    print(f"Test embeddings are loaded in {end_time - start_time:.2f} seconds.")
    print(f"Embedding {embeddings.shape}, labels {labels.shape}")

    return embeddings, labels


"""
# Save Embeddings
# ==============
model = bm1.TransformerEncoderMAE(
    img_size=cfg.g_len,
    in_chans=cfg.in_channel,
    out_chans=cfg.out_channel,
    embed_dim=cfg.embed_dim,
    depth=cfg.depth,
    num_heads=cfg.num_heads,
    mlp_ratio=cfg.mlp_ratio,
    drop=cfg.drop,
    layer_norm_eps=cfg.layer_norm_eps,
)
model.eval()
pmodel = torch.load(cfg.model_checkpoint, weights_only=True, map_location="cpu")["model"]
state_dict = model.state_dict()
for name, p in pmodel.items():
    if (
        "decoder" not in name and "mask" not in name
    ):  # and name!="norm.weight" and name!="norm.bias":
        state_dict[name].data.copy_(p.data)
        # state_dict[name].copy_(p)
# model.load_state_dict(state_dict)

model = model.to(device)

layer_to_hook = dict(model.named_modules())[cfg.layer_name]

del pmodel, name, p, state_dict
torch.cuda.empty_cache()
print("model is loaded")
"""
"""
test_loader = setup_testing_dataloader(cfg)
save_labeled_embeddings(test_loader, model, layer_to_hook, device)
print("test data is finished")

train_loader = setup_training_dataloader(cfg, 8192)
save_unlabeled_embeddings(train_loader, model, layer_to_hook, device)
print("train data is finished")
"""

# """
# # small test for small bird
# model = bm.BirdModel(4, 30, cfg.out_channel)

model = bm1.TransformerEncoderMAE(
    img_size=cfg.g_len,
    in_chans=cfg.in_channel,
    out_chans=cfg.out_channel,
    embed_dim=cfg.embed_dim,
    depth=cfg.depth,
    num_heads=cfg.num_heads,
    mlp_ratio=cfg.mlp_ratio,
    drop=cfg.drop,
    layer_norm_eps=cfg.layer_norm_eps,
)
bm.load_model(cfg.model_checkpoint, model, device)
model.to(device)
model.eval()
torch.cuda.empty_cache()
print("model is loaded")
layer_to_hook = dict(model.named_modules())[cfg.layer_name]  # fc

"""
activation = []
hook_handle = layer_to_hook.register_forward_hook(get_activation(activation))
test_loader = setup_testing_dataloader(cfg)
with torch.no_grad():
    for data, ldts in tqdm(test_loader):
        data = data.to(device)
        output = model(data)  # shape (B, embed_dim)
        feats = activation.pop()  # shape (B, embed_dim, 1)
        feats = feats.flatten(1)  # shape (B, embed_dim)
        # feats = feats[:, 0, :] # shape (B, L, embed_dim) -> (B, embed_dim) # for norm
        labels = ldts[:, 0].cpu().numpy()
        feats = feats.detach().cpu().numpy()
        # torch.save({'feats':feats,'labels':labels}, cfg.save_path/"test_c3_avg.npy")
        # np.savez(cfg.save_path/"test_c3_avg_np", **{'feats':feats,'labels':labels})
hook_handle.remove()
"""

# [0, 1, 2, 3, 4, 5] -> [0, 2, 4, 5, 6, 9]
# TODO: ind2name contains all labels (0-9). But label 7 is removed. In the balance data everything is mapped to 0-5.
# mapping = {0: 0, 1: 2, 2: 4, 3: 5, 4: 6, 5: 9}
keep_labels = [0, 2, 4, 5, 6, 9]  # [0, 2, 4, 6, 9] # [0, 2, 4, 5, 6, 9]
discover_labels = [3]  # [1, 3, 5, 7] # [1, 3, 7]
rem_labels = [1, 3, 7, 8]
mapping = {i: l for i, l in enumerate(keep_labels)}

test_loader = setup_testing_dataloader(cfg)
save_labeled_embeddings(test_loader, model, layer_to_hook, device)
print("test data is finished")

# Load test embeddings
feats, labels, output = load_labeled_embeddings("test.npz")
l_feats = torch.tensor(feats, device="cuda")
l_targets = torch.tensor(labels, device="cuda")

# Load train embeddings
u_feats_np = load_unlabeled_embeddings()  # (n=0)
u_feats = torch.tensor(u_feats_np, device="cuda")

# classification
preds = torch.argmax(torch.tensor(output), dim=1).cpu().numpy()
probs = torch.softmax(torch.tensor(output), dim=1)
cm = contingency_matrix(labels, preds)
# 4516 4694 # small
# 4532 4694 # ft: large 9 classes f_mem1
# 2080 2184 # ft: large 9 classes small on balance data
# 2084 2184 # ft: large 9 classes f_mem1 on balance data
# 2034 2184 # ft: large 6 classes f_mem_bal1
# 2020 2184 # ft: large 6 classes f_mem_bal2
# 2065 2184 # ft: large 6 classes f_mem1_bal3
# 1999 2184 # ft: large 6 classes f_mem_bal4
# 4068 4327 # ft: large 6 classes f_mem1_bal3
# For model trained on 6 parameters the sum is not correct. Since labels are 9 and preds are 6. contingency_matrix is correct.
# sum(preds == labels) # is incorrect if preds are 6 and labels are 9.
if cm.shape[0] != cm.shape[1]:
    print(sum(cm[[0, 1, 2, 4, 5]].diagonal()), l_feats.shape[0])
    # print(sum(cm[[0, 2, 4, 5, 6, 8]].diagonal()), l_feats.shape[0])
else:
    print(sum(cm.diagonal()), l_feats.shape[0])

# bu.plot_confusion_matrix(cm, [bu.ind2name[i] for i in keep_labels])
false_neg = cm.sum(axis=1) - cm.max(axis=1)
# fmt: off
print(sum(false_neg), cm.sum() - sum(false_neg), cm.sum(), (cm.sum() - sum(false_neg)) / cm.sum())
bu.plot_confusion_matrix(cm, true_labels=[bu.ind2name[i] for i in cfg.labels_to_use])
# bu.plot_confusion_matrix(cm, true_labels=[bu.ind2name[i] for i in [0, 1, 2, 3, 4, 5, 6, 8, 9]], pred_labels=[bu.ind2name[i] for i in keep_labels])
# fmt: on

# Some plotting
# fmt: off
from datetime import datetime, timezone

df = pd.read_csv(cfg.test_data_file, header=None)
d, l = next(iter(test_loader)) # d.shape 4694 x 20 x 4, l.shape 4694 x 3
if cfg.channel_first:
    d = d.permute(0, 2, 1)
d = d.cpu().numpy()
l = l.cpu().numpy()

def plot_one(i):
    j = np.where((np.abs(df.iloc[:,4] - d[i,0,0]) < 1e-6) & (np.abs(df.iloc[:,5] - d[i,0,1]) < 1e-6) & (np.abs(df.iloc[:,6] - d[i,0,2]) < 1e-6))[0][0]
    bu.plot_one(np.array(df.iloc[j:j+20,4:])) # bu.plot_one(d[i])
    plt.title(f"{torch.max(probs[i]).item():.2f}, {df.iloc[j,7]:.2f}")

label, pred = 3, 5
label_name, pred_name = bu.ind2name[mapping[label]], bu.ind2name[mapping[pred]]
inds = np.where((labels==label) & (preds==pred))[0]
save_path = cfg.model_checkpoint.parent/cfg.model_checkpoint.stem/f"{label_name}_{pred_name}"
save_path.mkdir(parents=True, exist_ok=True)
for i in inds:
    plot_one(i)
    dt = datetime.fromtimestamp(l[i,2], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    name = f"{i},{l[i,1]},{dt},{torch.max(probs[i]).item():.2f}"
    plt.savefig(save_path / f"{name}.png", bbox_inches="tight")
    plt.close()
# fmt: on

# # Get second max probability value
# second_max_probs = torch.sort(probs, dim=1, descending=True)[0][:,1]

# GCD
"""
# First classes for supervised learning
cut_class = 3  # 5
uf = l_feats[l_targets >= cut_class]
ut = l_targets[l_targets >= cut_class]
lf = l_feats[l_targets < cut_class]
lt = l_targets[l_targets < cut_class]
"""
"""
# 6 classes for supervised learning. 3 (1, 3, 7) classes for clustering.
uf = l_feats[np.isin(labels, [1,3,7])]
ut = l_targets[np.isin(labels, [1,3,7])]
lf = l_feats[~np.isin(labels, [1,3,7])]
lt = l_targets[~np.isin(labels, [1,3,7])]
u_old2new = dict(zip([1,3,7],[6,7,8]))
l_old2new = dict(zip([0, 2, 4, 5, 6, 8],[0, 1, 2, 3, 4, 5]))
ut = torch.tensor([u_old2new[i.item()] for i in ut], device=device)
lt = torch.tensor([l_old2new[i.item()] for i in lt], device=device)
"""

"""
# 6 classes for supervised learning. 3 (1, 3, 7) classes for clustering. 2000 samples for supervised and the rest for clustering.
uf1 = l_feats[np.isin(labels, discover_labels)]
ut1 = l_targets[np.isin(labels, discover_labels)]
uf2 = l_feats[~np.isin(labels, discover_labels)][2000:]
ut2 = l_targets[~np.isin(labels, discover_labels)][2000:]
ut = torch.cat((ut1, ut2))
uf = torch.cat((uf1, uf2))
lf = l_feats[~np.isin(labels, discover_labels)][:2000]
lt = l_targets[~np.isin(labels, discover_labels)][:2000]
u_old2new = dict(zip([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 6, 1, 7, 2, 3, 4, 8, 5]))
l_old2new = dict(zip([0, 2, 4, 5, 6, 8], [0, 1, 2, 3, 4, 5]))
ut = torch.tensor([u_old2new[i.item()] for i in ut], device=device)
lt = torch.tensor([l_old2new[i.item()] for i in lt], device=device)
"""


def old2new(all_labels, pred_labels, discover_labels):
    """
    GCD (Generalized Category Discovery) maps labels to new labels.
    """
    # all_labels = list(np.unique(labels))
    # pred_labels = list(np.unique(preds))
    new_labels = np.arange(len(pred_labels), len(all_labels))
    d2n = {d: n for d, n in zip(discover_labels, new_labels)}
    old = [i for i in all_labels if i not in discover_labels]
    l_old2new = dict(zip(old, pred_labels))
    new = []
    for i in all_labels:
        if i in discover_labels:
            new.append(d2n[i])
        else:
            new.append(l_old2new[i])
    u_old2new = dict(zip(all_labels, new))
    return l_old2new, u_old2new


def test_old2new():
    u_old2new = dict(zip([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 6, 1, 7, 2, 3, 4, 8, 5]))
    l_old2new = dict(zip([0, 2, 4, 5, 6, 8], [0, 1, 2, 3, 4, 5]))
    discover_labels = [1, 3, 7]
    all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    pred_labels = [0, 1, 2, 3, 4, 5]
    pred_l_old2new, pred_u_old2new = old2new(all_labels, pred_labels, discover_labels)
    assert (
        pred_u_old2new == u_old2new
    ), f"u_old2new mismatch: {pred_u_old2new} != {u_old2new}"
    assert (
        pred_l_old2new == l_old2new
    ), f"l_old2new mismatch: {pred_l_old2new} != {l_old2new}"

    u_old2new = dict(zip([0, 1, 2, 3, 4, 5], [0, 1, 2, 5, 3, 4]))
    l_old2new = dict(zip([0, 1, 2, 4, 5], [0, 1, 2, 3, 4]))
    discover_labels = [3]
    all_labels = [0, 1, 2, 3, 4, 5]
    pred_labels = [0, 1, 2, 3, 4]
    pred_l_old2new, pred_u_old2new = old2new(all_labels, pred_labels, discover_labels)
    assert (
        pred_u_old2new == u_old2new
    ), f"u_old2new mismatch: {pred_u_old2new} != {u_old2new}"
    assert (
        pred_l_old2new == l_old2new
    ), f"l_old2new mismatch: {pred_l_old2new} != {l_old2new}"


# """
# 6 classes for supervised learning. 3 (1, 3, 7) classes for clustering.
uf1 = l_feats[np.isin(labels, discover_labels)]
ut1 = l_targets[np.isin(labels, discover_labels)]
uf2 = l_feats[~np.isin(labels, discover_labels)][2000:]
ut2 = l_targets[~np.isin(labels, discover_labels)][2000:]
ut = torch.cat((ut1, ut2))
uf = torch.cat((uf1, uf2, u_feats))
lf = l_feats[~np.isin(labels, discover_labels)][:2000]
lt = l_targets[~np.isin(labels, discover_labels)][:2000]
all_labels = list(np.unique(labels))
pred_labels = list(np.unique(preds))
l_old2new, u_old2new = old2new(all_labels, pred_labels, discover_labels)
ut = torch.tensor([u_old2new[i.item()] for i in ut], device=device)
lt = torch.tensor([l_old2new[i.item()] for i in lt], device=device)
# """


# fmt: off
kmeans = K_Means(k=cfg.n_clusters, tolerance=1e-4, max_iterations=100, n_init=3, random_state=10, pairwise_batch_size=8192)
# fmt: off
kmeans.fit_mix(uf, lf, lt)
preds = kmeans.labels_.cpu().numpy()
assert np.all(preds[:lt.shape[0]] == lt.cpu().numpy()) == True
# 1092 1092 2184 # balanced data
# 2313 2014 4327 # unbalance_1378
print(uf.shape[0], lf.shape[0], l_feats.shape[0])

# ordered_feat = np.concatenate((lf.cpu(),uf.cpu()), axis=0)
# ordered_labels = np.concatenate((lt.cpu(), ut.cpu()))
# cm = contingency_matrix(ordered_labels, preds)

ordered_feat = np.concatenate((lf.cpu(),uf1.cpu(), uf2.cpu()), axis=0)
ordered_labels = np.concatenate((lt.cpu(), ut.cpu()))
cm = contingency_matrix(ordered_labels, preds[:labels.shape[0]])

false_neg = cm.sum(axis=1) - cm.max(axis=1)
# fmt: off
print(sum(false_neg), cm.sum() - sum(false_neg), cm.sum(), (cm.sum() - sum(false_neg)) / cm.sum())
bu.plot_confusion_matrix(cm, [bu.ind2name[i] for i in [0, 2, 4, 5, 6, 9, 1, 3, 8]])
# bu.plot_confusion_matrix(cm, [bu.ind2name[i] for i in [0, 2, 4, 6, 9, 5]])
# ['Flap', 'Soar', 'Float', 'SitStand', 'TerLoco', 'Pecking', 'ExFlap', 'Boat', 'Manouvre']
# fmt: on

# ft: large 6 classes f_mem1_bal3
# mo: 119 2065 2184 0.95
# ss: 110 2074 2184 0.95 # fc
# ss: 127 2057 2184 0.94 # norm
# us: 186 1998 2184 0.91 # fc
# us: 299 1885 2184 0.86 # norm
# us: 130 2054 2184 0.94 # norm f_mem_bal1
#
# ft: large 6 classes f_mem_bal1 unbalance_1378
# mo:  259 4068 4327 0.94
# ss:  220 4107 4327 0.95 # fc
# ss:  495 3832 4327 0.89 # norm
# us:  311 4016 4327 0.93 # fc
# us:  324 4003 4327 0.93 # norm
#
# ft: large 6 classes f_mem_bal1 corrected_combined_unique_sorted012 6 given 3 predicted
# mo:  435 4259 4694 0.91
# ss:  152 4542 4694 0.97 # fc
# ss:  440 4254 4694 0.91 # fc with 2000 samples
# us:  1202 3492 4694 0.74 # fc # 1151 3543 4694 0.75 chaged after loading
#
# ft: large 6 classes f_mem1_bal4 (only fc trained.)
# mo: 185 1999 2184 0.92
# ss: 230 1954 2184 0.89 # norm
# us: 495 1689 2184 0.77 # norm
#
# ft: large 9 classes f_mem1
# mo: 100 2084 2184 0.95
# ss: 382 1802 2184 0.83 # norm
# us: 484 1700 2184 0.78 # norm
#
# small: 9 classes
# mo: 104 2080 2184 0.95 # avgpool
# ss: 403 1781 2184 0.82 # avgpool
# ss: 432 1752 2184 0.80 # fc
# us: 510 1674 2184 0.77 # avgpool
# us: 503 1681 2184 0.77 # fc

# Clustering
kmeans = MiniBatchKMeans(cfg.n_clusters, random_state=cfg.seed)
kmeans.fit(feats)
preds = kmeans.labels_
cm = contingency_matrix(labels, preds)
false_neg = cm.sum(axis=1) - cm.max(axis=1)
# fmt: off
print(sum(false_neg), cm.sum() - sum(false_neg), cm.sum(), (cm.sum() - sum(false_neg)) / cm.sum())
# fmt: on
# bu.plot_confusion_matrix(cm, [bu.ind2name[i] for i in range(10) if i != 7])

# fmt: off
reducer = TSNE(n_components=2, random_state=cfg.seed)
reduced = reducer.fit_transform(feats)
plt.figure();scatter=plt.scatter(reduced[:,0], reduced[:,1],c=labels,cmap="tab20",s=5);plt.colorbar(scatter, label="Cluster Label")
plt.figure();scatter=plt.scatter(reduced[:,0], reduced[:,1],c=preds, cmap="tab20",s=5);plt.colorbar(scatter, label="Cluster Label")
plt.show(block=True)
# fm: off
print("done")


"""
GCD (Generalized Category Discovery)
I have: 
Model:
- supervised small model 9 classes
- Pretraining and finetuning on 6 classes
- foundation model time-series: chronos, moment, moirai
Data:
- unlabeled
- unbalanced
- balanced
Clustering:
- KMean
- semisup KMean
FT on 6 classes with balanced data with KMean and semsup KMean is working.
"""
