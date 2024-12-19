import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
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


# # General loads
bu.set_seed(cfg.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    batch_size=cfg.batch_size,  # len(dataset), # 4096
    shuffle=False,
    num_workers=cfg.num_workers,
    drop_last=False,
)

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


def get_activation(name):
    def hook(model, input, output):
        activation.append(output.detach().cpu().numpy())

    return hook


activation = []

# Register the forward hook
hook_handle = layer_to_hook.register_forward_hook(get_activation(cfg.layer_name))

# Initialize MiniBatchKMeans
kmeans = MiniBatchKMeans(
    n_clusters=cfg.n_clusters, batch_size=cfg.batch_size, random_state=cfg.seed
)

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
    print(f"Clustering completed in {end_time - start_time:.2f} seconds.")
    print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
except MemoryError as e:
    print(
        "MemoryError encountered! The dataset might be too large to fit in available memory."
    )
except Exception as e:
    print(f"An error occurred: {e}")

# Remove the hook after extraction
hook_handle.remove()
print("done")

# Test data
# ==========

# Load data
all_measurements, label_ids = bd.load_csv(cfg.test_data_file)
all_measurements, label_ids = bd.get_specific_labesl(
    all_measurements, label_ids, bu.target_labels
)
dataset = bd.BirdDataset(all_measurements, label_ids, channel_first=cfg.channel_first)
loader = DataLoader(
    dataset,
    batch_size=len(dataset),  # 4096
    shuffle=False,
    num_workers=1,
    drop_last=False,
)

activation = []

# Register the forward hook
hook_handle = layer_to_hook.register_forward_hook(get_activation(cfg.layer_name))

# Extract embeddings and perform clustering in batches
start_time = time.time()

with torch.no_grad():
    for data, ldts in tqdm(loader):
        # Forward pass
        output = model(data.to(device))
        X = activation[0][:, 0, :]
        c_labels = kmeans.predict(X)

end_time = time.time()
print(f"Clustering completed in {end_time - start_time:.2f} seconds.")
print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")

# Remove the hook after extraction
hook_handle.remove()
print(all_measurements.shape, c_labels.shape)

# Compare cluster labels with actual labels
labels = ldts[:, 0]

n_labels = len(np.unique(labels))
counts = np.zeros((n_labels, cfg.n_clusters), dtype=np.int64)
for label in range(n_classes):
    sel = c_labels[labels == label]
    for c_label in range(cfg.n_clusters):
        counts[label, c_label] = sum(sel == c_label)
bu.plot_confusion_matrix(counts)

cfg.save_path.mkdir(parents=True, exist_ok=True)
plt.savefig(
    cfg.save_path / f"row_labels_col_clusters_c{cfg.n_clusters}_b{cfg.batch_size}.png",
    bbox_inches="tight",
)

rfile = open(
    cfg.save_path / f"cluster_with_all_data_c{cfg.n_clusters}_b{cfg.batch_size}.csv",
    "w",
)
for label in range(n_classes):
    items = [str(i) for i in c_labels[labels == label]]
    items = ", ".join(items)
    rfile.write(f"\n{label}\n")
    rfile.write(items)
rfile.close()
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
