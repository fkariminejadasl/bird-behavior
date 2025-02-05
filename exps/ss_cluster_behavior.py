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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
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
        all_measurements, label_ids, bu.target_labels
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
                feats = activation.pop()  # shape (B, 1, embed_dim)
                feats = feats[:, 0, :]  # shape (B, embed_dim)
                torch.save(feats.detach().cpu().numpy(), cfg.save_path / f"{i}.npy")
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
            _ = model(data)
            if activation:
                feats = activation.pop()  # shape (B, 1, embed_dim)
                feats = feats[:, 0, :]  # shape (B, embed_dim)
                labels = ldts[:, 0].cpu().numpy()
                feats = feats.detach().cpu().numpy()
                torch.save(
                    {"feats": feats, "labels": labels}, cfg.save_path / "test.npy"
                )
            del data, feats, labels
            torch.cuda.empty_cache()

    hook_handle.remove()

    end_time = time.time()
    print(f"Test embeddings are loaded in {end_time - start_time:.2f} seconds.")


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


# ======================================
# Semi Supervised KMeans from GCD
# https://github.com/sgvaze/generalized-category-discovery/blob/main/methods/clustering/faster_mix_k_means_pytorch.py

import copy
import random

import numpy as np
import torch
from sklearn.utils import check_random_state
from sklearn.utils._joblib import Parallel, delayed, effective_n_jobs


def pairwise_distance(data1, data2, batch_size=None):
    r"""
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    """
    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    if batch_size == None:
        dis = (A - B) ** 2
        # return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)
        #  torch.cuda.empty_cache()
    else:
        i = 0
        dis = torch.zeros(data1.shape[0], data2.shape[0])
        while i < data1.shape[0]:
            if i + batch_size < data1.shape[0]:
                dis_batch = (A[i : i + batch_size] - B) ** 2
                dis_batch = dis_batch.sum(dim=-1)
                dis[i : i + batch_size] = dis_batch
                i = i + batch_size
                #  torch.cuda.empty_cache()
            elif i + batch_size >= data1.shape[0]:
                dis_final = (A[i:] - B) ** 2
                dis_final = dis_final.sum(dim=-1)
                dis[i:] = dis_final
                #  torch.cuda.empty_cache()
                break
    #  torch.cuda.empty_cache()
    return dis


class K_Means:

    def __init__(
        self,
        k=3,
        tolerance=1e-4,
        max_iterations=100,
        init="k-means++",
        n_init=10,
        random_state=None,
        n_jobs=None,
        pairwise_batch_size=None,
        mode=None,
    ):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pairwise_batch_size = pairwise_batch_size
        self.mode = mode

    def split_for_val(self, l_feats, l_targets, val_prop=0.2):

        np.random.seed(0)

        # Reserve some labelled examples for validation
        num_val_instances = int(val_prop * len(l_targets))
        val_idxs = np.random.choice(
            range(len(l_targets)), size=(num_val_instances), replace=False
        )
        val_idxs.sort()
        remaining_idxs = list(set(range(len(l_targets))) - set(val_idxs.tolist()))
        remaining_idxs.sort()
        remaining_idxs = np.array(remaining_idxs)

        val_l_targets = l_targets[val_idxs]
        val_l_feats = l_feats[val_idxs]

        remaining_l_targets = l_targets[remaining_idxs]
        remaining_l_feats = l_feats[remaining_idxs]

        return remaining_l_feats, remaining_l_targets, val_l_feats, val_l_targets

    def kpp(self, X, pre_centers=None, k=10, random_state=None):
        random_state = check_random_state(random_state)

        if pre_centers is not None:

            C = pre_centers

        else:

            C = X[random_state.randint(0, len(X))]

        C = C.view(-1, X.shape[1])

        while C.shape[0] < k:

            dist = pairwise_distance(X, C, self.pairwise_batch_size)
            dist = dist.view(-1, C.shape[0])
            d2, _ = torch.min(dist, dim=1)
            prob = d2 / d2.sum()
            cum_prob = torch.cumsum(prob, dim=0)
            r = random_state.rand()

            if len((cum_prob >= r).nonzero()) == 0:
                debug = 0
            else:
                ind = (cum_prob >= r).nonzero()[0][0]
            C = torch.cat((C, X[ind].view(1, -1)), dim=0)

        return C

    def fit_mix_once(self, u_feats, l_feats, l_targets, random_state):

        def supp_idxs(c):
            return l_targets.eq(c).nonzero().squeeze(1)

        l_classes = torch.unique(l_targets)
        support_idxs = list(map(supp_idxs, l_classes))
        l_centers = torch.stack(
            [l_feats[idx_list].mean(0) for idx_list in support_idxs]
        )
        cat_feats = torch.cat((l_feats, u_feats))

        centers = torch.zeros([self.k, cat_feats.shape[1]]).type_as(cat_feats)
        centers[: len(l_classes)] = l_centers

        labels = -torch.ones(len(cat_feats)).type_as(cat_feats).long()

        l_classes = l_classes.cpu().long().numpy()
        l_targets = l_targets.cpu().long().numpy()
        l_num = len(l_targets)
        cid2ncid = {
            cid: ncid for ncid, cid in enumerate(l_classes)
        }  # Create the mapping table for New cid (ncid)
        for i in range(l_num):
            labels[i] = cid2ncid[l_targets[i]]

        # initialize the centers, the first 'k' elements in the dataset will be our initial centers
        centers = self.kpp(u_feats, l_centers, k=self.k, random_state=random_state)

        # Begin iterations
        best_labels, best_inertia, best_centers = None, None, None
        for it in range(self.max_iterations):
            centers_old = centers.clone()

            dist = pairwise_distance(u_feats, centers, self.pairwise_batch_size)
            u_mindist, u_labels = torch.min(dist, dim=1)
            u_inertia = u_mindist.sum()
            l_mindist = torch.sum((l_feats - centers[labels[:l_num]]) ** 2, dim=1)
            l_inertia = l_mindist.sum()
            inertia = u_inertia + l_inertia
            labels[l_num:] = u_labels

            for idx in range(self.k):

                selected = torch.nonzero(labels == idx).squeeze()
                selected = torch.index_select(cat_feats, 0, selected)
                centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            center_shift = torch.sum(
                torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1))
            )

            if center_shift**2 < self.tolerance:
                # break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break

        return best_labels, best_inertia, best_centers, i + 1

    def fit_mix(self, u_feats, l_feats, l_targets):

        random_state = check_random_state(self.random_state)
        best_inertia = None
        fit_func = self.fit_mix_once

        if effective_n_jobs(self.n_jobs) == 1:
            for it in tqdm(range(self.n_init), maxinterval=self.n_init):

                labels, inertia, centers, n_iters = fit_func(
                    u_feats, l_feats, l_targets, random_state
                )

                if best_inertia is None or inertia < best_inertia:
                    self.labels_ = labels.clone()
                    self.cluster_centers_ = centers.clone()
                    best_inertia = inertia
                    self.inertia_ = inertia
                    self.n_iter_ = n_iters

        else:

            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(fit_func)(u_feats, l_feats, l_targets, seed) for seed in seeds
            )
            # Get results with the lowest inertia

            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]


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

# Function to extract embeddings per batch using hooks
named_mods = dict(model.named_modules())
layer_to_hook = named_mods[cfg.layer_name]

del pmodel, name, p, state_dict
torch.cuda.empty_cache()
print("model is loaded")

with torch.no_grad():
    o = model(torch.rand(1,20,4, device=device))

test_loader = setup_testing_dataloader(cfg)
save_labeled_embeddings(test_loader, model, layer_to_hook, device)
print("test data is loaded in cpu")

train_loader = setup_training_dataloader(cfg, 8192)
print("data is loaded in cpu")
save_unlabeled_embeddings(train_loader, model, layer_to_hook, device)
"""

# """
# small test for small bird
model = bm.BirdModel(4, 30, 9).to(device)
bm.load_model("/home/fatemeh/Downloads/bird/result/45_best.pth", model, device)
model.eval()
activation = []
hook_handle = model.fc.register_forward_hook(get_activation(activation))
test_loader = setup_testing_dataloader(cfg)
with torch.no_grad():
    for data, ldts in tqdm(test_loader):
        data = data.permute((0, 2, 1))
        data = data.to(device)
        _ = model(data)  # shape (B, embed_dim)
        feats = activation.pop()  # shape (B, embed_dim, 1)
        feats = feats.flatten(1)  # shape (B, embed_dim)
        labels = ldts[:, 0].cpu().numpy()
        feats = feats.detach().cpu().numpy()
        # torch.save({'feats':feats,'labels':labels}, cfg.save_path/"test_c3_avg.npy")
        # np.savez(cfg.save_path/"test_c3_avg_np", **{'feats':feats,'labels':labels})
hook_handle.remove()
# """

# # Load test embeddings
# l_feats = []
# l_targets = []
# fls = torch.load(cfg.save_path / f"test.npy")
# l_feats = fls["feats"]  # N x D
# l_targets = fls["labels"]  # N
# l_feats = torch.tensor(l_feats, device="cuda")
# l_targets = torch.tensor(l_targets, device="cuda")
# print(l_feats.shape)
# print("done")


# # Load train embeddings
# u_feats = []
# for i in range(263):
#     feats = torch.load(cfg.save_path/f"{i}.npy")
#     u_feats.append(feats)
# u_feats = np.concatenate(u_feats, axis=0)  # N x D
# u_feats = torch.tensor(u_feats, device="cuda")
# print(u_feats.shape)
# print("done")


# kmeans = K_Means(k=10, tolerance=1e-4, max_iterations=10, n_init=3, random_state=1, pairwise_batch_size=8192)
# kmeans.fit_mix(u_feats, l_feats, l_targets)
# preds = kmeans.labels_.cpu().numpy()
# centers = kmeans.cluster_centers_.cpu()
# preds = kmeans.labels_.cpu()
# all(preds[:l_feats.shape[0]]==l_targets.cpu())
# print("nmi", adjusted_mutual_info_score(preds, fls['labels']))

l_feats = torch.tensor(feats, device="cuda")
l_targets = torch.tensor(labels, device="cuda")
# m = torch.nn.LayerNorm(l_feats.shape[1]).to(device)
# l_feats = m(l_feats)
# l_feats = torch.nn.functional.normalize(l_feats, p=2, dim=1)

uf = l_feats[l_targets >= 5]
ut = l_targets[l_targets >= 5]
lf = l_feats[l_targets < 5]
lt = l_targets[l_targets < 5]
# fmt: off
kmeans = K_Means(k=9, tolerance=1e-4, max_iterations=100, n_init=3, random_state=10, pairwise_batch_size=8192)
# fmt: off
kmeans.fit_mix(uf, lf, lt)
preds = kmeans.labels_.cpu().numpy()
sum(preds[lt.shape[0]:] == ut.cpu().numpy())
assert np.all(preds[:lt.shape[0]] == lt.cpu().numpy()) == True
# 2472: 618 (10), 127 (9,r=1), 444 (9,r=10)
print(sum(preds[lt.shape[0]:] == ut.cpu().numpy()), ut.shape[0])

reducer = PCA(n_components=2, random_state=42)
reduced = reducer.fit_transform(feats)
kmeans = MiniBatchKMeans(9)
kmeans.fit(reduced)
preds = kmeans.labels_
sum(preds == labels) # 826 avgpool, 1855 fc
# preds = torch.argmax(l_feats, dim=1).cpu().numpy() # fc: 4516
# fmt: off
plt.figure();scatter=plt.scatter(reduced[:,0], reduced[:,1],c=labels,cmap="tab20",s=5);plt.colorbar(scatter, label="Cluster Label")
plt.figure();scatter=plt.scatter(reduced[:,0], reduced[:,1],c=preds,cmap="tab20",s=5);plt.colorbar(scatter, label="Cluster Label")
# fm: off
print("done")
