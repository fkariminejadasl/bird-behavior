import gc
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

# import umap  # umap-learn
from omegaconf import ListConfig, OmegaConf
from scipy.optimize import linear_sum_assignment
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
    silhouette_samples,
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

# cfg.save_path.mkdir(parents=True, exist_ok=True)


def setup_training_dataloader(cfg, batch_size, channel_first):
    # Load data
    # all_measurements, label_ids = bd.load_csv(cfg.data_file)
    # all_measurements, label_ids = bd.get_specific_labesl(
    #     all_measurements, label_ids, bu.target_labels
    # )
    # dataset = bd.BirdDataset(all_measurements, label_ids, channel_first=channel_first)
    gimus = read_csv_files(cfg.data_file)
    print(gimus.shape)
    gimus = gimus.reshape(-1, cfg.g_len, cfg.in_channel)
    gimus = np.ascontiguousarray(gimus)
    print(gimus.shape)
    dataset = bd.BirdDataset(gimus, channel_first=channel_first)
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


def setup_testing_dataloader(test_data_file, labels_to_use, channel_first):
    # Load data
    # all_measurements, label_ids = bd.load_csv(test_data_file)
    # all_measurements, label_ids = bd.get_specific_labesl(
    #     all_measurements, label_ids, labels_to_use
    # )
    df = pd.read_csv(test_data_file, header=None)
    df = df[df[3].isin(labels_to_use)]
    mapping = {l: i for i, l in enumerate(labels_to_use)}
    df[3] = df[3].map(mapping)
    all_measurements = df[[4, 5, 6, 7]].values.reshape(-1, 20, 4)
    label_ids = df[[3, 0, 0]].iloc[::20].values
    dataset = bd.BirdDataset(all_measurements, label_ids, channel_first=channel_first)
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


def save_unlabeled_embeddings(save_path, loader, model, layer_to_hook, device):
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
                feats = activation.pop()
                if cfg.layer_name == "norm":
                    feats = feats[:, 0, :]  # B x L x embed_dim -> B x embed_dim
                else:
                    feats = feats.flatten(1)  # B x embed_dim x 1 -> B x embed_dim
                # torch.save(feats.detach().cpu().numpy(), save_path / f"{i}_{cfg.layer_name}.npy")
                feats = feats.cpu().numpy()
                np.savez(save_path / f"{i}_{cfg.layer_name}", **{"feats": feats})
            del data, feats
            torch.cuda.empty_cache()

    hook_handle.remove()

    end_time = time.time()
    print(f"Train embeddings are loaded in {end_time - start_time:.2f} seconds.")


def save_labeled_embeddings(save_file, loader, model, layer_to_hook, device):
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
            if len(output) == 2:
                output = output[1]
            if activation:
                feats = activation.pop()
                if cfg.layer_name == "norm":
                    feats = feats[:, 0, :]  # B x L x embed_dim -> B x embed_dim
                else:
                    feats = feats.flatten(1)  # B x embed_dim x 1 -> B x embed_dim
                labels = ldts[:, 0].cpu().numpy()
                feats = feats.detach().cpu().numpy()
                ouput = output.detach().cpu().numpy()
                # torch.save({"feats": feats, "labels": labels}, save_file) # "test*.npy"
                np.savez(
                    save_file,  # test*.npz
                    **{"feats": feats, "labels": labels, "output": ouput},
                )
            del data, feats, labels
            torch.cuda.empty_cache()

    hook_handle.remove()

    end_time = time.time()
    print(f"Test embeddings are loaded in {end_time - start_time:.2f} seconds.")


def load_labeled_embeddings(save_file):
    # Load test embeddings
    results = np.load(save_file)
    feats = results["feats"]  # N x D
    labels = results["labels"]  # N
    output = results["output"]
    print(feats.shape)
    return feats, labels, output


def load_unlabeled_embeddings(save_path):
    n = len(list(save_path.glob("*.npz"))) - 1
    # Load train embeddings
    u_feats = []
    for i in range(n):
        # feats = torch.load(save_path/f"{i}.npy")
        feats = np.load(save_path / f"{i}_{cfg.layer_name}.npz")["feats"]
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


class Mapper:
    def __init__(self, old2new: dict):
        # old is a list like [0,2,4,5,6,9], new is [0, ..., 5]
        self.old2new = old2new
        self.new2old = {n: o for o, n in old2new.items()}

    def encode(self, orig: torch.Tensor):
        """Map original labels → 0…K-1 space"""
        return torch.tensor([self.old2new[int(i)] for i in orig], device=orig.device)

    def decode(self, chang):
        """Map 0…K-1 predictions back → original labels"""
        return torch.tensor([self.new2old[int(i)] for i in chang], device=chang.device)


def gcd_old2new(lt_labels, discover_labels):
    l_old2new = {l: i for i, l in enumerate(lt_labels)}
    u_old2new = l_old2new.copy()
    max_lt_label = len(lt_labels)
    for i, l in enumerate(discover_labels):
        u_old2new[l] = i + max_lt_label
    return l_old2new, u_old2new


def test_gcd_old2new():
    discover_labels = [1, 3, 8]
    lt_labels = [0, 2, 4, 5, 6, 9]
    ut_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]
    l_old2new, u_old2new = gcd_old2new(lt_labels, discover_labels)
    # fmt:off
    assert sorted(discover_labels + lt_labels)==ut_labels
    assert l_old2new == {0: 0, 2: 1, 4: 2, 5: 3, 6: 4, 9: 5}
    assert u_old2new == {0: 0, 2: 1, 4: 2, 5: 3, 6: 4, 9: 5, 1: 6, 3: 7, 8: 8}
    # fmt:on

    discover_labels = [5]
    lt_labels = [0, 2, 4, 6, 9]
    ut_labels = [0, 2, 4, 5, 6, 9]
    # fmt:off
    assert sorted(discover_labels + lt_labels)==ut_labels
    assert l_old2new == {0: 0, 2: 1, 4: 2, 6: 3, 9: 4}
    assert u_old2new == {0: 0, 2: 1, 4: 2, 6: 3, 9: 4, 5: 5}
    # fmt:on


# fmt: off
def save_cm_embeddings(save_path, name, cm, true_labels, pred_labels, reduced, labels, preds, rd_method="tsne", PLOT_LABELS=False, centers=None):
    # base = plt.get_cmap("tab20").colors # "tab10", "tab20b", "tab20c"
    # cmap9 = ListedColormap(base[:len(true_labels)])
    
    bu.plot_confusion_matrix(cm, true_labels=true_labels, pred_labels=pred_labels)
    plt.savefig(save_path / f"{name}_cm.png", bbox_inches="tight")

    unique_classes = np.unique(preds)
    bounds = np.concatenate((unique_classes-0.5, [unique_classes[-1]+0.5]))
    norm   = mpl.colors.BoundaryNorm(bounds, ncolors=len(unique_classes))
    
    plt.figure()
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=preds, cmap="tab20", s=5, norm=norm)#, vmin=0, vmax=len(pred_labels)-1)
    if centers is not None:
        for i in range(len(centers)):
            plt.scatter(centers[i][0], centers[i][1], s = 130, marker = "*", color='r')
    cbar = plt.colorbar(scatter, label="Label")
    cbar.set_ticks(unique_classes)
    plt.title("Predictions")
    plt.savefig(save_path / f"{name}_{rd_method}.png", bbox_inches="tight")

    if PLOT_LABELS:
        unique_classes = np.unique(labels)
        bounds = np.concatenate((unique_classes-0.5, [unique_classes[-1]+0.5]))
        norm   = mpl.colors.BoundaryNorm(bounds, ncolors=len(unique_classes))

        plt.figure()
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab20", s=5, norm=norm)#, vmin=0, vmax=len(true_labels)-1)
        cbar = plt.colorbar(scatter, label="Label")
        # cbar = plt.colorbar(scatter, ticks=unique_classes, label="Label")
        # place ticks at integer positions 0…n-1: place ticks before relabel them
        cbar.set_ticks(unique_classes)
        cbar.set_ticklabels(true_labels)
        plt.title("Labeled Data")
        plt.savefig(save_path / f"{name}_{rd_method}_label.png", bbox_inches="tight")
    # plt.show(block=True)
# fmt: on


def save_hungarian(cm, method_name, save_path):
    """
    Save the Hungarian assignment for the contingency matrix.
    """
    # n = max(cm.shape)
    # padded = np.zeros((n, n), dtype=cm.dtype)
    # padded[:cm.shape[0], :cm.shape[1]] = cm.max() - cm
    rows, cols = linear_sum_assignment(cm.max() - cm)
    with open(save_path, "a") as f:
        f.write(f"{method_name}:\n")
        f.write(", ".join(map(str, rows)) + "\n")
        f.write(", ".join(map(str, cols)) + "\n\n")


def save_scores(cm, discover_labels, all_labels, name=None, save_path=None):
    """
    Example: test_calculate_score
    """
    # if-else is not needed but if part make the code easier to understand.
    org2new = {l: i for i, l in enumerate(all_labels)}
    if len(discover_labels) == 1:
        idx = org2new[discover_labels[0]]
        sum = cm[idx].sum()
        acc = cm[idx, idx]
        acc /= sum if sum > 0 else 0  # avoid division by zero
    else:
        # Get indices in the cm corresponding to discover_labels
        idxs = [org2new[i] for i in discover_labels]

        # Slice the confusion matrix to only those rows/cols
        slice_cm = cm[np.ix_(idxs, idxs)]

        # Perform assignment on the smaller matrix
        rs, cs = linear_sum_assignment(slice_cm.max() - slice_cm)

        acc = 0
        sum = 0
        for r, c in zip(rs, cs):
            orig_r = idxs[r]
            orig_c = idxs[c]
            sum += cm[orig_r].sum()
            acc += cm[orig_r, orig_c]

        acc /= sum if sum > 0 else 0  # avoid division by zero
        """
        # older version: take all the rows
        org2new = {l: i for i, l in enumerate(all_labels)}
        rows, cols = linear_sum_assignment(cm.max() - cm)
        acc = 0
        sum = 0
        for discover_label in discover_labels:
            idx = np.where(rows == org2new[discover_label])[0][0]
            sum += cm[rows[idx]].sum()
            acc += cm[rows[idx], cols[idx]]
        acc /= sum
        """

    if save_path is not None:
        score_file = save_path.parent / "scores.txt"
        with open(score_file, "a") as f:
            f.write(f"{name}:\n")
            f.write(f"{acc:.3f}\n")
    return acc


def test_calculate_score():
    # fmt: off
    cm = np.array([
       [   0,  159,    0,    0,    0,    0,    0,  484,    0],
       [   0,   38,    0,    0,    0,    0,    0,    0,    0],
       [ 410,    0,    4,   31,   34,    0,    1,   17,   40],
       [   0,    0,    0,  176,    0,    0,    0,    0,    0],
       [   0,    0,    0,    0,  729,    0,    0,    0,    0],
       [   4,    0,    0,    0,    0, 1492,    0,    0,    6],
       [   0,    0,    0,    0,    0,    1,  326,    0,   10],
       [   0,    0,    0,    0,    0,    0,    0,  151,    0],
       [   0,    0,   76,    0,    0,    0,    2,    0,  147]])
    expected = 0.3474576271186441
    discover_labels = [0, 2]
    all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]
    score = save_scores(cm, discover_labels, all_labels)
    assert round(score, 2) == round(expected, 2)
    cm = np.array([
       [ 643,    0,    0,    0,    0],
       [  10,  226,   64,  237,    0],
       [   0,    5,  724,    0,    0],
       [   0,   15,    0, 1486,    1],
       [   0,    9,    0,    1,  327]])
    
    expected = 0.4208566108007449
    discover_labels = [2]
    all_labels = [0, 2, 4, 5, 6]
    score = save_scores(cm, discover_labels, all_labels)
    assert round(score, 2) == round(expected, 2)
    # fmt: on


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


def main(cfg):
    # Settings
    # ==============
    channel_first = cfg.model.channel_first

    save_path = (
        cfg.model_checkpoint.parent / cfg.model_checkpoint.stem.split("_best")[0]
    )
    save_path.mkdir(parents=True, exist_ok=True)

    lt_labels = cfg.lt_labels
    discover_labels = cfg.discover_labels
    labels_to_use = ut_labels = cfg.labels_to_use
    labels_trained = cfg.labels_trained
    if discover_labels is None:
        discover_labels = sorted(set(cfg.all_labels) - set(cfg.lt_labels))
    if labels_to_use is None:
        labels_to_use = ut_labels = cfg.all_labels.copy()
    if labels_trained is None:
        labels_trained = lt_labels.copy()  # lt_labels.copy() ut_labels.copy()

    # assert sorted(discover_labels + lt_labels) == labels_to_use

    l_old2new, u_old2new = gcd_old2new(lt_labels, discover_labels)
    l_mapper = Mapper(l_old2new)
    u_mapper = Mapper(u_old2new)
    mapper = Mapper({l: i for i, l in enumerate(ut_labels)})
    mapper_trained = Mapper({l: i for i, l in enumerate(labels_trained)})

    if cfg.model.name == "smallemb":
        model = bm.BirdModelWithEmb(cfg.in_channel, 30, cfg.out_channel)

    if cfg.model.name == "small":
        model = bm.BirdModel(cfg.in_channel, 30, cfg.out_channel)

    if cfg.model.name == "mae":
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

    save_file = save_path / f"test_{cfg.layer_name}.npz"
    if not save_file.exists():
        test_loader = setup_testing_dataloader(
            cfg.test_data_file, labels_to_use, channel_first
        )
        save_labeled_embeddings(save_file, test_loader, model, layer_to_hook, device)
        print("test data is finished")

    # Load test embeddings
    feats, labels, output = load_labeled_embeddings(save_file)
    labels = mapper.decode(torch.tensor(labels)).numpy()

    """
    train_loader = setup_training_dataloader(cfg, 8192, channel_first)
    save_unlabeled_embeddings(save_path, train_loader, model, layer_to_hook, device)
    print("train data is finished")
    """
    # # Load train embeddings
    # u_feats_np = load_unlabeled_embeddings(save_path) # (n=0)
    # u_feats = torch.tensor(u_feats_np, device="cuda")

    # GCD
    # ==============
    l_feats = torch.tensor(feats, device="cuda")
    l_targets = torch.tensor(labels, device="cuda")

    # tensor_utl = torch.tensor(ut_labels, device=device)
    # mask = torch.isin(l_targets, tensor_utl)
    # l_feats = l_feats[mask]
    # l_targets = l_targets[mask]

    tensor_dl = torch.tensor(discover_labels, device=device)
    mask = torch.isin(l_targets, tensor_dl)
    uf1 = l_feats[mask]
    ut1 = l_targets[mask]
    uf2 = l_feats[~mask][2000:]
    ut2 = l_targets[~mask][2000:]
    ut = torch.cat((ut1, ut2))
    uf = torch.cat((uf1, uf2))  # uf = torch.cat((uf1, uf2, u_feats))
    lf = l_feats[~mask][:2000]
    lt = l_targets[~mask][:2000]

    ut = u_mapper.encode(ut)
    lt = l_mapper.encode(lt)

    # fmt: off
    kmeans = K_Means(k=cfg.n_clusters, tolerance=1e-4, max_iterations=100, n_init=3, random_state=10, pairwise_batch_size=8192)
    # fmt: off
    kmeans.fit_mix(uf, lf, lt)
    # preds = kmeans.labels_.cpu().numpy()
    preds = kmeans.labels_
    centers = kmeans.cluster_centers_

    # assert np.all(preds[:lt.shape[0]] == lt.cpu().numpy()) == True
    assert preds[:lt.shape[0]].equal(lt)
    print(uf.shape[0], lf.shape[0], l_feats.shape[0])

    # ordered_feat = torch.concatenate((lf, uf1, uf2), axis=0)
    ordered_feat = torch.concatenate((lf, uf), axis=0)
    ordered_labels = torch.concatenate((lt, ut))
    ordered_labels = u_mapper.decode(ordered_labels.cpu()).numpy()
    k_preds = u_mapper.decode(preds.cpu()).numpy()

    # fmt: off
    reducer = TSNE(n_components=2, random_state=cfg.seed)
    # reduced = reducer.fit_transform(ordered_feat.cpu().numpy())
    feats_and_centers = torch.cat((ordered_feat, centers), axis=0)
    reduced_feats_and_centers = reducer.fit_transform(feats_and_centers.cpu().numpy())
    reduced = reduced_feats_and_centers[:-centers.shape[0], :]
    reduced_centers = reduced_feats_and_centers[-centers.shape[0]:, :]

    cm = contingency_matrix(ordered_labels, k_preds[:ordered_labels.shape[0]])
    # fmt: on

    save_path_results = save_path / (
        f"{cfg.layer_name}_tr" + "".join([str(i) for i in labels_trained])
    )
    save_path_results.mkdir(parents=True, exist_ok=True)
    hungarian_file = save_path_results / "hungarian.txt"
    true_labels = [bu.ind2name[i] for i in labels_to_use]
    pred_labels = range(len(u_old2new))
    method_name = "gcd"
    name = (
        f"{method_name}_gvn"
        + "".join(map(str, lt_labels))
        + "_dsl"
        + "".join(map(str, discover_labels))
    )

    # base = plt.get_cmap("tab20").colors # "tab10", "tab20b", "tab20c"
    # cmap9 = ListedColormap(base[:len(true_labels)])
    # scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=k_preds, cmap=cmap9, s=5, vmin=0, vmax=len(pred_labels)-1)
    # cbar = plt.colorbar(scatter, label="Label")
    # cbar.set_ticklabels(true_labels)
    save_cm_embeddings(
        save_path_results,
        name,
        cm,
        true_labels,
        pred_labels,
        reduced,
        ordered_labels,
        k_preds,
        centers=reduced_centers,
    )
    save_hungarian(cm, method_name, hungarian_file)

    exp = cfg.model_checkpoint.stem.split("_best")[0]
    name = (
        f"{exp}_tr"
        + "".join(map(str, labels_trained))
        + f"_gvn"
        + "".join(map(str, lt_labels))
        + "_dsl"
        + "".join(map(str, discover_labels))
    )
    save_scores(cm, discover_labels, cfg.all_labels, name, save_path)

    # classification
    # ==============

    preds = torch.argmax(torch.tensor(output), dim=1)
    probs = torch.softmax(torch.tensor(output), dim=1)

    preds = mapper_trained.decode(preds).cpu().numpy()
    cm = contingency_matrix(labels, preds)

    reducer = TSNE(n_components=2, random_state=cfg.seed)
    reduced = reducer.fit_transform(feats)

    # # new score to check which points are mixed (separability)
    # import umap
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    # reduced = reducer.fit_transform(feats)
    # per_point_scores = silhouette_samples(feats, labels, metric="euclidean")
    # mean_score = silhouette_score(feats, labels, metric="euclidean")
    # inds = np.where(labels==6)[0] # np.unique(labels)
    # sinds = np.where(per_point_scores[inds]<0)[0]
    # per_point_scores[inds[sinds]]
    # new_score = (len(inds) - len(sinds))/len(inds)

    # plt.figure()
    # plt.plot(reduced[:,0], reduced[:,1], '.b')
    # plt.plot(reduced[inds,0], reduced[inds,1], '*g')
    # plt.plot(reduced[inds[sinds],0], reduced[inds[sinds],1], '*r')

    true_labels = [bu.ind2name[i] for i in ut_labels]
    pred_labels = [bu.ind2name[i] for i in labels_trained]
    name = "cls"
    save_cm_embeddings(
        save_path_results,
        name,
        cm,
        true_labels,
        pred_labels,
        reduced,
        labels,
        preds,
        PLOT_LABELS=True,
    )
    save_hungarian(cm, name, hungarian_file)
    """
    # Some plotting
    # fmt: off
    from datetime import datetime, timezone

    df = pd.read_csv(cfg.test_data_file, header=None)
    d, l = next(iter(test_loader)) # d.shape 4694 x 20 x 4, l.shape 4694 x 3
    if channel_first:
        d = d.permute(0, 2, 1)
    d = d.cpu().numpy()
    l = l.cpu().numpy()

    def plot_one(i):
        j = np.where((np.abs(df.iloc[:,4] - d[i,0,0]) < 1e-6) & (np.abs(df.iloc[:,5] - d[i,0,1]) < 1e-6) & (np.abs(df.iloc[:,6] - d[i,0,2]) < 1e-6))[0][0]
        bu.plot_one(np.array(df.iloc[j:j+20,4:])) # bu.plot_one(d[i])
        plt.title(f"{torch.max(probs[i]).item():.2f}, {df.iloc[j,7]:.2f}")

    name2ind = {name: ind for ind, name in bu.ind2name.items()}
    label_name, pred_name = "Boat", "SitStand"
    label, pred = name2ind["Boat"], name2ind["SitStand"]
    inds = np.where((labels==label) & (preds==pred))[0]
    save_path = cfg.model_checkpoint.parent/cfg.model_checkpoint.stem.split('_')[0]/f"{label_name}_{pred_name}"
    save_path.mkdir(parents=True, exist_ok=True)
    for i in inds:
        plot_one(i)
        dt = datetime.fromtimestamp(l[i,2], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        name = f"{i},{l[i,1]},{dt},{torch.max(probs[i]).item():.2f}"
        plt.savefig(save_path / f"{name}.png", bbox_inches="tight")
        plt.close()
    # fmt: on
    """

    # # Get second max probability value
    # second_max_probs = torch.sort(probs, dim=1, descending=True)[0][:,1]

    # Clustering
    # ==============

    kmeans = MiniBatchKMeans(cfg.n_clusters, random_state=cfg.seed)
    kmeans.fit(feats)
    k_preds = kmeans.labels_
    cm = contingency_matrix(labels, k_preds)
    false_neg = cm.sum(axis=1) - cm.max(axis=1)
    # fmt: off
    print(sum(false_neg), cm.sum() - sum(false_neg), cm.sum(), (cm.sum() - sum(false_neg)) / cm.sum())
    # fm: on

    true_labels = range(len(mapper.new2old))
    pred_labels = range(len(mapper.new2old))
    method_name = "kmn"
    name = f"{method_name}_gvn_ds{cfg.n_clusters}" 
    save_cm_embeddings(save_path_results, name, cm, true_labels, pred_labels, reduced, labels, k_preds)
    save_hungarian(cm, method_name, hungarian_file)
    print("Done")


def get_config():
    return cfg


if __name__ == "__main__":
    exclude = [0, 2]  # [1, 3, 8]
    cfg.all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9] #[0, 2, 4, 5, 6]  # [0, 1, 2, 3, 4, 5, 6, 8, 9]
    exp = "145"
    cfg.lt_labels = sorted(set(cfg.all_labels) - set(exclude))
    cfg.model_checkpoint = Path(f"/home/fatemeh/Downloads/bird/result/{exp}_best.pth")
    cfg.model.name = "small"  # "smallemb"
    cfg.model.channel_first = True
    cfg.labels_trained = cfg.lt_labels.copy()  # cfg.all_labels, cfg.lt_labels
    cfg.out_channel = len(cfg.labels_trained)
    cfg.n_clusters = len(cfg.all_labels)
    cfg.layer_name = "avgpool"  # avgpool, fc
    print(f"Experiment {exp}: Excluding label {exclude}")
    main(cfg)

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
