import gc
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# import umap  # umap-learn
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from sklearn.metrics.cluster import contingency_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from behavior import data as bd
from behavior import metrics as bmet
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.ss_kmeans import K_Means


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
    save_path: Path | None
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

# General loads
bu.set_seed(cfg.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# cfg.save_path.mkdir(parents=True, exist_ok=True)


def setup_training_dataloader(csv_file, cfg, batch_size, channel_first):
    df = pd.read_csv(csv_file, header=None)
    gimus = df[[4, 5, 6, 7]].values
    print(gimus.shape)
    gimus = gimus.reshape(-1, cfg.g_len, cfg.in_channel)
    gimus = np.ascontiguousarray(gimus, dtype=np.float32)
    print(gimus.shape)
    dataset = bd.BirdDataset(gimus, channel_first=channel_first)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # len(dataset), # 4096
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    del gimus
    gc.collect()
    return loader


def setup_testing_dataloader(test_data_file, data_labels, channel_first):
    # Load data
    # all_measurements, label_ids = bd.load_csv(test_data_file)
    # all_measurements, label_ids = bd.get_specific_labesl(
    #     all_measurements, label_ids, data_labels
    # )
    df = pd.read_csv(test_data_file, header=None)
    df = df[df[3].isin(data_labels)]
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
    """
    Load test embeddings

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - feats: Feature embeddings (N x D)
        - labels: Labels (N)
        - output: Additional model outputs
    """
    #
    results = np.load(save_file)
    feats = results["feats"]  # N x D
    labels = results["labels"]  # N
    output = results["output"]
    print(feats.shape)
    return feats, labels, output


def load_unlabeled_embeddings(save_path):
    """
    Return: np.ndarray N x C
    """
    # n = len(list(save_path.glob(f"[0-9]*_{cfg.layer_name}.npz")))
    n = len(
        [
            p
            for p in save_path.glob(f"*_{cfg.layer_name}.npz")
            if p.stem.split("_")[0].isdigit()
        ]
    )

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
def plot_cm_embeddings(cm, true_labels, pred_labels, reduced, preds, centers=None, u_data=None, u_labels=None):
    
    bu.plot_confusion_matrix(cm, true_labels=true_labels, pred_labels=pred_labels)

    unique_classes = np.unique(preds)
    cmap = plt.get_cmap("tab20")
    cmap_dict = {i: mcolors.to_hex(cmap(i)) for i in range(20)}
    custom_cmap = mcolors.ListedColormap([cmap_dict[p] for p in unique_classes])

    # Colormap issue: Matplotlib treats integer labels as continuous values from [vmin, vmax] to [0, 1]. 
    # Missing label numbers (e.g., no class 7) cause colors labels 8 and 9 merge into color 8 unless you use a discrete BoundaryNorm.
    # Build boundaries that separate integer classes cleanly, even with gaps
    # e.g., for [0,1,2,3,4,5,6,8,9] => [-0.5, 0.5, 1.5, ..., 7.5, 8.5, 9.5]
    bounds = np.concatenate((unique_classes - 0.5, [unique_classes[-1] + 0.5]))
    norm = mcolors.BoundaryNorm(bounds, ncolors=len(unique_classes))
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=preds, cmap=custom_cmap, norm=norm, s=5)
    if u_data is not None:
        _ = ax.scatter(u_data[:, 0], u_data[:, 1], c=u_labels, cmap=custom_cmap, norm=norm, s=15, edgecolors='black', linewidth=0.5)
    if centers is not None:
        for i in range(len(centers)):
            plt.scatter(centers[i][0], centers[i][1], s = 130, marker = "*", color='r')
    # Tkinter issue: The issue happens because the Tkinter plotting window’s toolbar crashes 
    # when a new colorbar is added. Using fig.colorbar() instead of plt.colorbar() fixes it.
    cbar = fig.colorbar(scatter, ax=ax, ticks=unique_classes, label="Label")
    plt.title("Predictions")
    # plt.show(block=True)
# fmt: on


def tsne_label_unlabel_embs(feats, u_feats):
    all_feats = np.concatenate((feats, u_feats), axis=0)
    reducer = TSNE(n_components=2, random_state=cfg.seed)
    all_reduced = reducer.fit_transform(all_feats)
    reduced = all_reduced[: feats.shape[0]]
    u_reduced = all_reduced[feats.shape[0] :]
    return reduced, u_reduced


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


def calculate_accuracy(cm, discover_labels, data_labels):
    """
    org2new: mapping original label to new labels
    """
    # if-else is not needed but if part make the code easier to understand.
    org2new = {l: i for i, l in enumerate(data_labels)}
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
    return acc


def save_scores(scores, name=None, save_path=None):
    if save_path is not None:
        score_file = save_path.parent / "scores.txt"
        score_text = ", ".join(
            [
                f"{k}:{v}" if isinstance(v, int) else f"{k}:{v:.3f}"
                for k, v in scores.items()
            ]
        )
        record = f"{name}:\n{score_text}\n"
        with open(score_file, "a") as f:
            f.write(record)


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
    score = calculate_accuracy(cm, discover_labels, all_labels)
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
    score = calculate_accuracy(cm, discover_labels, all_labels)
    assert round(score, 2) == round(expected, 2)
    # fmt: on


def calculate_separability_scores(feats, labels, discover_labels):
    """Calculate separability scores for discovered class.
    For both scores higher is better.

    Args:
        feats: Feature matrix of shape (n_samples, n_features)
        labels: Ground truth labels
        discover_labels: List of labels to discover

    Returns:
        density_separability_score: Proportion of points with positive silhouette score
        cluster_silhouette_score: Mean silhouette score for discovered class
        cluster_size: Number of points in discovered class
    """
    # mean_score = silhouette_score(feats, labels, metric="euclidean")
    per_point_scores = silhouette_samples(feats, labels, metric="euclidean")
    inds = np.where(labels == discover_labels[0])[0]
    sinds = np.where(per_point_scores[inds] < 0)[0]

    density_separability_score = (len(inds) - len(sinds)) / len(inds)
    cluster_silhouette_score = np.mean(per_point_scores[inds])
    cluster_size = len(inds)

    # print(per_point_scores[inds[sinds]])
    # plt.figure()
    # plt.plot(reduced[:,0], reduced[:,1], '.b')
    # plt.plot(reduced[inds,0], reduced[inds,1], '*g')
    # plt.plot(reduced[inds[sinds],0], reduced[inds[sinds],1], '*r')

    return density_separability_score, cluster_silhouette_score, cluster_size


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


# GCD
# ==============
def gcd(
    feats: np.ndarray,
    u_feats: np.ndarray,
    labels: np.ndarray,
    discover_labels: list,
    u_mapper,
    l_mapper,
):
    l_feats = torch.tensor(feats, device="cuda")
    l_targets = torch.tensor(labels, device="cuda")

    tensor_dl = torch.tensor(discover_labels, device=device)
    mask = torch.isin(l_targets, tensor_dl)

    known_f = l_feats[~mask]
    known_t = l_targets[~mask]
    splits = bu.stratified_split(known_t, split_ratios=[0.5, 0.5], seed=cfg.seed)
    idx1, idx2 = splits[0], splits[1]
    uf2 = known_f[idx1]
    ut2 = known_t[idx1]
    lf = known_f[idx2]
    lt = known_t[idx2]

    uf1 = l_feats[mask]
    ut1 = l_targets[mask]
    ut = torch.cat((ut1, ut2))

    u_feats = torch.tensor(u_feats, device="cuda")
    uf = torch.cat((uf1, uf2, u_feats))
    ut = u_mapper.encode(ut)
    lt = l_mapper.encode(lt)

    # fmt: off
    kmeans = K_Means(k=cfg.n_clusters, tolerance=1e-4, max_iterations=100, n_init=3, random_state=10, pairwise_batch_size=8192)
    # fmt: on
    kmeans.fit_mix(uf, lf, lt)
    preds = kmeans.labels_  # torch .cpu().numpy()
    centers = kmeans.cluster_centers_
    k_preds = u_mapper.decode(preds.cpu()).numpy()  # (N, )
    print(lf.shape, uf1.shape, uf2.shape, u_feats.shape)
    assert preds[: lt.shape[0]].equal(lt)

    ordered_feat = torch.concatenate((lf, uf), axis=0)
    ordered_labels = torch.concatenate((lt, ut))
    ordered_labels = u_mapper.decode(ordered_labels.cpu()).numpy()
    k_preds = u_mapper.decode(preds.cpu()).numpy()

    cm = contingency_matrix(ordered_labels, k_preds[: ordered_labels.shape[0]])
    u_cm = contingency_matrix(
        ordered_labels[lt.shape[0] :], k_preds[lt.shape[0] : ordered_labels.shape[0]]
    )
    l_cm = contingency_matrix(ordered_labels[: lt.shape[0]], k_preds[: lt.shape[0]])

    print("model\n", cm)

    reducer = TSNE(n_components=2, random_state=cfg.seed)
    # reduced = reducer.fit_transform(ordered_feat.cpu().numpy())
    feats_and_centers = torch.cat((ordered_feat, centers), axis=0)
    reduced_feats_and_centers = reducer.fit_transform(feats_and_centers.cpu().numpy())
    reduced = reduced_feats_and_centers[: -centers.shape[0], :]
    reduced_centers = reduced_feats_and_centers[-centers.shape[0] :, :]

    # Calcuate Scores
    # Compare discover label with labeled data:
    # so the embeddings and predictions are concatenated (dl_{})
    # data: labeled + unlabeled(discover+old+unlabel)
    mask = k_preds == discover_labels[0]

    max_ovl_hdr = bmet.max_ovl_kde_hdr(
        reduced[mask], reduced[: lt.shape[0]], k_preds[: lt.shape[0]]
    )

    true_labels = [bu.ind2name[i] for i in np.unique(ordered_labels)]
    pred_labels = [i for i in np.unique(k_preds)]

    DEBUG = False
    if DEBUG:
        # All = Labeled + Unlabeled
        pred_labels = [i for i in np.unique(k_preds)]
        plot_cm_embeddings(
            cm,
            true_labels,
            pred_labels,
            reduced,
            k_preds,
            centers=reduced_centers,
        )
        # Unlabeled
        plot_cm_embeddings(
            u_cm,
            true_labels,
            pred_labels,
            reduced[lt.shape[0] :],
            k_preds[lt.shape[0] :],
            centers=reduced_centers,
            u_data=reduced[k_preds.shape[0] - u_feats.shape[0] :],
            u_labels=k_preds[k_preds.shape[0] - u_feats.shape[0] :],
        )

        # Labeled
        true_labels = [bu.ind2name[i] for i in np.unique(ordered_labels[: lt.shape[0]])]
        pred_labels = [i for i in np.unique(k_preds[: lt.shape[0]])]
        plot_cm_embeddings(
            l_cm,
            true_labels,
            pred_labels,
            reduced[: lt.shape[0]],
            k_preds[: lt.shape[0]],
            centers=None,
        )
    return max_ovl_hdr


def main(cfg):
    # Settings
    # ==============
    channel_first = cfg.model.channel_first

    if cfg.save_path is None:
        save_path = cfg.model_checkpoint.parent
    else:
        save_path = cfg.save_path
    save_path = save_path / cfg.model_checkpoint.stem.split("_best")[0]
    save_path.mkdir(parents=True, exist_ok=True)

    discover_labels = cfg.discover_labels
    assert discover_labels is not None
    data_labels = cfg.data_labels
    trained_labels = cfg.trained_labels
    lt_labels = cfg.lt_labels
    if data_labels is None:
        data_labels = cfg.all_labels.copy()
    if trained_labels is None:
        trained_labels = lt_labels.copy()

    l_old2new, u_old2new = gcd_old2new(lt_labels, discover_labels)
    l_mapper = Mapper(l_old2new)
    u_mapper = Mapper(u_old2new)
    data_mapper = Mapper({l: i for i, l in enumerate(data_labels)})
    trained_mapper = Mapper({l: i for i, l in enumerate(trained_labels)})

    name = f"{cfg.model.name}_{cfg.layer_name}"
    if cfg.use_unlabel:
        name += "_unlabel"
    results_path = save_path / (
        name
        + "_tr"
        + "".join([str(i) for i in trained_labels])
        + "_dsl"
        + "".join(map(str, discover_labels))
    )
    results_path.mkdir(parents=True, exist_ok=True)

    if cfg.model.name == "smallemb":
        model = bm.BirdModelWithEmb(cfg.in_channel, 30, cfg.out_channel)
        bm.load_model(cfg.model_checkpoint, model, device)

    if cfg.model.name == "small":
        model = bm.BirdModel(cfg.in_channel, 30, cfg.out_channel)
        bm.load_model(cfg.model_checkpoint, model, device)

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
        pmodel = torch.load(
            "/home/fatemeh/Downloads/bird/snellius/p20_2_best.pth", weights_only=True
        )["model"]
        state_dict = model.state_dict()
        for name, p in pmodel.items():
            if (
                "decoder" not in name and "mask" not in name
            ):  # and name!="norm.weight" and name!="norm.bias":
                state_dict[name].data.copy_(p.data)
        # bm.load_model(cfg.model_checkpoint, model, device)
    model.to(device)
    model.eval()
    torch.cuda.empty_cache()
    print("model is loaded")
    layer_to_hook = dict(model.named_modules())[cfg.layer_name]  # fc

    # Prepare test embeddings
    save_file = results_path / f"test_{cfg.layer_name}.npz"
    if not save_file.exists():
        test_loader = setup_testing_dataloader(
            cfg.test_data_file, data_labels, channel_first
        )
        save_labeled_embeddings(save_file, test_loader, model, layer_to_hook, device)
        print("test data is finished")
    # Load test embeddings
    feats, labels, output = load_labeled_embeddings(save_file)
    # Cleanup: Remove file
    save_file.unlink()

    # Prepare sequence
    C = feats.shape[1]
    u_feats = np.empty((0, C), dtype=np.float32)
    csv_file = Path("/home/fatemeh/Downloads/bird/data/ssl/ssl20/298_0.csv")
    train_loader = setup_training_dataloader(csv_file, cfg, 8192, channel_first)
    save_unlabeled_embeddings(results_path, train_loader, model, layer_to_hook, device)
    # Load embeddings
    u_feats = load_unlabeled_embeddings(results_path)  # N x C
    # Cleanup: Remove file
    _ = [
        p.unlink()
        for p in results_path.glob(f"*_{cfg.layer_name}.npz")
        if p.stem.split("_")[0].isdigit()
    ]

    df = pd.read_csv(csv_file, header=None)
    n_data = u_feats.shape[0]
    step = 100
    for i in range(0, n_data, step):
        s_date = df.iloc[i * 20, 1]
        e_date = df.iloc[min(i + step, n_data) * 20, 1]
        print(f"start date: {s_date}, end date: {e_date}")
        u_feats_crop = u_feats[i : min(i + step, n_data)]
        score = gcd(
            feats,
            u_feats_crop,
            labels,
            discover_labels,
            u_mapper,
            l_mapper,
        )
        if score < 0.3:
            print(f"class discovered for {i}:{i + step}")


def get_config():
    return cfg


if __name__ == "__main__":
    # fmt: off
    exp = 125
    # e.g. (exclude_labels_from_data, discover_labels) = ([], [2]), ([2], [10])
    exclude_labels_from_data = []
    cfg.discover_labels = [10]
    cfg.all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9] # [0, 2, 4, 5, 6]  # [0, 1, 2, 3, 4, 5, 6, 8, 9]
    
    cfg.data_labels = sorted(set(cfg.all_labels)-set(exclude_labels_from_data))
    cfg.trained_labels = sorted(set(cfg.all_labels) - set(cfg.discover_labels) - set(exclude_labels_from_data))

    cfg.lt_labels = cfg.trained_labels.copy()
    cfg.n_clusters = 10
    cfg.out_channel = len(cfg.trained_labels)
    cfg.model_checkpoint = Path(f"/home/fatemeh/Downloads/bird/result/1discover_2/{exp}_best.pth")
    cfg.model.name = "small"  # "small", "smallemb", "mae"
    cfg.model.channel_first = True # False
    cfg.layer_name = "fc"  # avgpool, fc, norm
    cfg.save_path = Path(f"/home/fatemeh/Downloads/bird/result")
    cfg.use_unlabel = False
    cfg.data_file = Path("/home/fatemeh/Downloads/bird/data/ssl_mini")
    print(f"Experiment {exp}: Discover label {cfg.discover_labels}")
    main(cfg)
    # fmt: on
