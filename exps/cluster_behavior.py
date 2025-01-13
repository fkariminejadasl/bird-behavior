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
    dis = []
    timestamps = []
    for csv_file in data_path.glob("*.csv"):
        df = pd.read_csv(csv_file, header=None)
        gimus.append(df[[4, 5, 6, 7]].values)
        dis.append(df[[0, 2]].values)
        timestamps.extend(df[1].tolist())

    gimus = np.concatenate(gimus, axis=0)
    dis = np.concatenate(dis, axis=0)
    timestamps = np.array(timestamps)
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
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    return loader


def get_activation(activation):
    """Create a hook function that updates the given activation list."""

    def hook(model, input, output):
        activation.append(output.detach().cpu().numpy())

    return hook


def train_model(cfg, loader, model, kmeans, layer_to_hook):
    activation = []

    # Register the forward hook
    hook_handle = layer_to_hook.register_forward_hook(get_activation(activation))

    sample_rate = 0.01  # 10% of data
    sampled_embeddings = []
    sampled_labels = []
    # Extract embeddings and perform clustering in batches
    start_time = time.time()
    try:
        with torch.no_grad():
            for data in tqdm(loader):
                # Forward pass
                output = model(data.to(device))

                # Process current batch embeddings
                if activation:
                    X = activation.pop()[:, 0, :]
                    kmeans.partial_fit(X)  # Incremental clustering with partial_fit

        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
        return kmeans
    except MemoryError as e:
        print(
            "MemoryError encountered! The dataset might be too large to fit in available memory."
        )
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        # Remove the hook after extraction
        hook_handle.remove()


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
    return loader


def test_model(cfg, loader, model, kmeans, layer_to_hook):
    activation = []

    # Register the forward hook
    hook_handle = layer_to_hook.register_forward_hook(get_activation(activation))

    try:
        # Extract embeddings and perform clustering in batches
        start_time = time.time()

        with torch.no_grad():
            for data, ldts in tqdm(loader):
                # Forward pass
                output = model(data.to(device))
                X = activation[0][:, 0, :]
                c_labels = kmeans.predict(X)

        end_time = time.time()
        print(f"Testing completed in {end_time - start_time:.2f} seconds.")
        print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
        return c_labels, ldts, X
    except Exception as e:
        print(f"Testing error: {e}")
    finally:
        # Remove the hook after extraction
        hook_handle.remove()


def get_cluster_metrics(kmeans, n_clusters, X, predicted_labels, true_labels):
    # Internal metrics (no ground truth required)
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, predicted_labels)
    calinski_harabasz = calinski_harabasz_score(X, predicted_labels)
    davies_bouldin = davies_bouldin_score(X, predicted_labels)

    # External metrics (ground truth required)
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    completeness = completeness_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    fmi = fowlkes_mallows_score(true_labels, predicted_labels)

    result = {
        "n_clusters": n_clusters,
        "inertia": inertia,
        "silhouette": silhouette,
        "calinski_harabasz": calinski_harabasz,
        "davies_bouldin": davies_bouldin,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "ari": ari,
        "ami": ami,
        "fmi": fmi,
    }
    return result


def visualize_test_clusters(
    test_embeddings,
    test_labels,
    cluster_centers,
    method="pca",
    save_file=None,
):
    """
    Visualize test embeddings + cluster centers in 2D using PCA, t-SNE, or UMAP.

    Args:
        test_embeddings (np.ndarray): Shape (N, embed_dim)
        test_labels (np.ndarray): Shape (N,) of predicted cluster labels for test
        cluster_centers (np.ndarray): Shape (n_clusters, embed_dim)
        method (str): 'pca', 'tsne', or 'umap'
    """

    # 1) Combine embeddings + centers for a single transform
    combined_data = np.concatenate([test_embeddings, cluster_centers], axis=0)
    n_test = len(test_embeddings)

    # 2) Choose the reducer
    if method.lower() == "pca":
        start_time = time.time()
        reducer = PCA(n_components=2, random_state=42)
    elif method.lower() == "tsne":
        start_time = time.time()
        reducer = TSNE(n_components=2, perplexity=cfg.perplexity, random_state=42)
    elif method.lower() == "umap":
        start_time = time.time()
        reducer = umap.UMAP(n_neighbors=cfg.n_neighbors, min_dist=cfg.min_dist)
    else:
        raise ValueError(f"Unknown method: {method} (choose 'pca', 'tsne', or 'umap')")

    # 3) Fit + transform combined data
    reduced = reducer.fit_transform(combined_data)  # shape (N + n_clusters, 2)
    end_time = time.time()
    print(f"{method.upper()} completed in {end_time - start_time:.2f} seconds.")

    # 4) Separate out the test embeddings vs. cluster centers
    test_reduced = reduced[:n_test]
    center_reduced = reduced[n_test:]

    # 5) Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        test_reduced[:, 0],
        test_reduced[:, 1],
        c=test_labels,
        cmap="tab20",  # "viridis", "Spectral".
        s=5,
        alpha=0.8,
        label="Test points",
    )

    center_labels = np.arange(cluster_centers.shape[0])
    plt.scatter(
        center_reduced[:, 0],
        center_reduced[:, 1],
        c=center_labels,
        cmap="tab20",
        marker="X",
        s=100,
        label="Cluster center",
    )
    plt.scatter(
        center_reduced[:, 0],
        center_reduced[:, 1],
        c="black",
        marker="X",
        s=50,
        label="Cluster center",
    )
    plt.title(f"Test Clusters + Centers ({method.upper()})")
    plt.colorbar(scatter, label="Cluster Label")
    plt.legend()

    if save_file:
        save_file = save_file.parent / f"{method}_{save_file.name}"
        plt.savefig(save_file, bbox_inches="tight", dpi=300)


# Load model
# model = bm.BirdModel(4, 30, 9).to(device)
# model = bm1.TransformerEncoderMAE(
#     img_size=cfg.g_len,
#     in_chans=cfg.in_channel,
#     out_chans=cfg.out_channel,
#     embed_dim=cfg.embed_dim,
#     depth=cfg.depth,
#     num_heads=cfg.num_heads,
#     mlp_ratio=cfg.mlp_ratio,
#     drop=cfg.drop,
#     layer_norm_eps=cfg.layer_norm_eps,
# ).to(device)
# model = bm1.MaskedAutoencoderViT(
#     img_size=cfg.g_len,
#     in_chans=cfg.in_channel,
#     patch_size=cfg.patch_size,
#     embed_dim=cfg.embed_dim,
#     depth=cfg.depth,
#     num_heads=cfg.num_heads,
#     decoder_embed_dim=cfg.decoder_embed_dim,
#     decoder_depth=cfg.decoder_depth,
#     decoder_num_heads=cfg.decoder_num_heads,
#     mlp_ratio=cfg.mlp_ratio,
#     layer_norm_eps=cfg.layer_norm_eps,
# ).to(device)
# bm.load_model(cfg.model_checkpoint, model, device)
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
).to(device)
pmodel = torch.load(cfg.model_checkpoint, weights_only=True)["model"]
state_dict = model.state_dict()
for name, p in pmodel.items():
    if (
        "decoder" not in name and "mask" not in name
    ):  # and name!="norm.weight" and name!="norm.bias":
        state_dict[name].data.copy_(p.data)
model.eval()


# Function to extract embeddings per batch using hooks
named_mods = dict(model.named_modules())
layer_to_hook = named_mods[cfg.layer_name]

# Prepare training data and train the model
# =============
all_metrics = []
for n_clusters in n_clusters_list:
    for batch_size in batch_sizes_list:
        print(f"Testing n_clusters={n_clusters}, batch_size={batch_size}")

        # Initialize MiniBatchKMeans # TODO IncrementalDBSCAN
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=batch_size, random_state=cfg.seed
        )

        # Train the model
        train_loader = setup_training_dataloader(cfg, batch_size)
        kmeans = train_model(cfg, train_loader, model, kmeans, layer_to_hook)

        # Prepare test data and test the model
        # ==============
        test_loader = setup_testing_dataloader(cfg)
        c_labels, ldts, embeddings = test_model(
            cfg, test_loader, model, kmeans, layer_to_hook
        )

        print(c_labels.shape)

        # Compare cluster labels with actual labels
        labels = ldts[:, 0]

        mtrcs = get_cluster_metrics(kmeans, n_clusters, embeddings, c_labels, labels)
        all_metrics.append(mtrcs)

        # TODO use: from sklearn.metrics.cluster import contingency_matrix
        n_labels = len(np.unique(labels))
        counts = np.zeros((n_labels, n_clusters), dtype=np.int64)
        for label in range(n_classes):
            sel = c_labels[labels == label]
            for c_label in range(n_clusters):
                counts[label, c_label] = sum(sel == c_label)
        bu.plot_confusion_matrix(counts)

        cfg.save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            cfg.save_path / f"row_labels_col_clusters_c{n_clusters}_b{batch_size}.png",
            bbox_inches="tight",
        )

        rfile = open(
            cfg.save_path / f"cluster_with_all_data_c{n_clusters}_b{batch_size}.csv",
            "w",
        )
        for label in range(n_classes):
            items = [str(i) for i in c_labels[labels == label]]
            items = ", ".join(items)
            rfile.write(f"\n{label}\n")
            rfile.write(items)
        rfile.close()

        # 1) PCA
        save_file = cfg.save_path / f"c{n_clusters}_b{batch_size}.png"
        # visualize_test_clusters(
        #     test_embeddings=embeddings,
        #     test_labels=c_labels,
        #     cluster_centers=kmeans.cluster_centers_,
        #     method="pca",
        #     save_file=save_file,
        # )

        # # 2) t-SNE
        # visualize_test_clusters(
        #     test_embeddings=embeddings,
        #     test_labels=c_labels,
        #     cluster_centers=kmeans.cluster_centers_,
        #     method="tsne",
        #     save_file=save_file,
        # )

        # 3) UMAP
        visualize_test_clusters(
            test_embeddings=embeddings,
            test_labels=c_labels,
            cluster_centers=kmeans.cluster_centers_,
            method="umap",
            save_file=save_file,
        )
        plt.close("all")
        print("\n")

all_metrics = pd.DataFrame(all_metrics)
all_metrics.to_csv(cfg.save_path / "metrics.csv", index=False, float_format="%.4f")

print("done")
"""
# ======================
named_mods = dict(model.named_modules())
layer_to_hook = named_mods[cfg.layer_name]

# Get the embeddings
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

activation = dict()
hook_handle = layer_to_hook.register_forward_hook(get_activation(cfg.layer_name))

# Perform a forward pass with input to trigger the hook
# data = torch.rand(1, 20, 4).to(device)
# _ = model(data.to(device))
# with torch.no_grad():
#     for data, ldts in loader:
#         output = model(data.to(device))
with torch.no_grad():
    for data in loader:
        output = model(data.to(device))

# After the forward pass, remove the hook
hook_handle.remove()

X = activation[cfg.layer_name][:,0,:].flatten(1)  # N x C
X = X.cpu().numpy()
# X = torch.rand(2148933, 5376).numpy() # 21 * 256 # gcn144
centers = 9
print(
    "===>",
    output.shape,
    len(activation[cfg.layer_name]),
    X.shape,
)

# # Generate synthetic 10-dimensional data
# n_samples = 1000
# n_features = 10
# centers = 5  # Number of clusters
# X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=42)

# Standardize features
X = StandardScaler().fit_transform(X)

# Compute cosine similarity matrix
cosine_sim_matrix = cosine_similarity(X)

# Convert cosine similarity to cosine distance
cosine_dist_matrix = 1 - cosine_sim_matrix
# cosim is always between -1, 1 so the weight should be between 0, 2. But due to
# numerical issue, we get negative value for distance. e.g. cosim=1.0000000000000004
cosine_dist_matrix = np.clip(cosine_dist_matrix, 0, None)

# Apply DBSCAN
# db = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples)
# labels = db.fit_predict(X)
db = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples, metric="precomputed")
labels = db.fit_predict(cosine_dist_matrix)

# Number of clusters in labels, ignoring noise if present
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Estimated number of clusters: {n_clusters}")

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot result
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X_pca[class_member_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"Estimated number of clusters: {n_clusters}")
plt.show()
print("done")
"""
