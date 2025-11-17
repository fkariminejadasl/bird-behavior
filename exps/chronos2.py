import gc
import time
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # requires: pip install 'pandas[pyarrow]'
import torch
from chronos import Chronos2Pipeline
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from behavior.utils import ind2name


def get_activation(activation):
    def hook(model, input, output):
        activation.append(output.detach())

    return hook


def example_get_activations():
    # Load model
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

    model = pipeline.model

    activation = []
    handle = model.encoder.final_layer_norm.register_forward_hook(
        get_activation(activation)
    )

    # Data
    x = torch.rand(4, 20, device="cuda")  # 20 time steps, 4 channels

    timestamps = pd.date_range("2000-01-01", periods=20, freq="h")

    context_df = pd.DataFrame(
        {
            "id": ["series_0"] * 20,
            "timestamp": timestamps,
            "ch0": x[0, :].cpu().numpy(),
            "ch1": x[1, :].cpu().numpy(),
            "ch2": x[2, :].cpu().numpy(),
            "ch3": x[3, :].cpu().numpy(),
        }
    )

    # For multivariate targets, you pass multiple columns as target
    with torch.no_grad():
        _ = pipeline.predict_df(
            context_df,
            prediction_length=4,
            id_column="id",
            timestamp_column="timestamp",
            target=["ch0", "ch1", "ch2", "ch3"],
            quantile_levels=[0.5],
        )

    with torch.no_grad():
        _ = model(x)
    # activation[1][:,0,0]
    activation[0][:, 0, 0]  # 4 x 4 x 768 (input_channels x n_tokkens x hidden_size)
    handle.remove()


# example_get_activations()

# fmt: off
def plot_embeddings(reduced, preds, true_labels):
    
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

    # Tkinter issue: The issue happens because the Tkinter plotting window’s toolbar crashes 
    # when a new colorbar is added. Using fig.colorbar() instead of plt.colorbar() fixes it.
    cbar = fig.colorbar(scatter, ax=ax, ticks=unique_classes, label="Label")
    cbar.set_ticklabels(true_labels)
    plt.title("Predictions")
    # plt.show(block=True)
# fmt: on


def main(cfg):
    if cfg.results_path.exists() is False:
        cfg.results_path.mkdir(parents=True, exist_ok=True)
    data_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2", device_map="cuda", torch_dtype=torch.float16
    )
    model = pipeline.model
    activation = []
    handle = model.encoder.final_layer_norm.register_forward_hook(
        get_activation(activation)
    )

    # Data
    df = pd.read_csv(cfg.data_file, header=None)
    df = df[df[3].isin(data_labels)]
    gimus = df[[4, 5, 6, 7]].values.reshape(-1, 20, 4)
    gimus[:, :, :3] = np.clip(gimus[:, :, :3], -2.0, 2.0)

    N, seq_len, n_channels = gimus.shape  # N=4338, seq_len=20, n_channels=4

    # Create ids and timestamps for all rows
    ids = np.repeat(np.arange(N), seq_len)  # [N * 20]
    timestamps = pd.date_range("2000-01-01", periods=seq_len, freq="h")
    timestamps_all = np.tile(timestamps.values, N)  # [N * 20]

    # Flatten gimus to [N*20, 4]
    flat = gimus.reshape(-1, n_channels)

    context_df = pd.DataFrame(
        {
            "id": ids,
            "timestamp": timestamps_all,
            "ch0": flat[:, 0],
            "ch1": flat[:, 1],
            "ch2": flat[:, 2],
            "ch3": flat[:, 3],
        }
    )  # [N*20, 6]

    with torch.no_grad():
        _ = pipeline.predict_df(
            context_df,
            prediction_length=4,
            id_column="id",
            timestamp_column="timestamp",
            target=["ch0", "ch1", "ch2", "ch3"],
            quantile_levels=[0.5],
        )

    # activation: list of tensors [N*4, 4, 768]. N mostly 256 and then last batch smaller 200 here.
    acts = []
    for act in activation:
        acts.append(act)
    acts = torch.concat(acts)
    handle.remove()

    num_targets = 4
    N_eff, T_tokens, d = acts.shape  # [N*4, 4, 768]
    assert N_eff == N * num_targets

    # fmt: off
    acts = acts.view(N, num_targets, T_tokens, d)  # [N, 4, T_tokens, 768]
    # vecs = acts.mean(dim=(1, 2)).cpu().to(torch.float32).numpy() # [N, 768]
    vecs = acts.mean(dim=2).reshape(N, -1).cpu().to(torch.float32).numpy() # [N, 4*768=3072]
    # fmt: on

    pca = PCA(n_components=50, random_state=cfg.seed)
    vecs_pca = pca.fit_transform(vecs)
    tsne = TSNE(n_components=2, random_state=cfg.seed)
    r_vecs = tsne.fit_transform(vecs_pca)
    labels = df[3].iloc[::20].values
    true_labels = [ind2name[l] for l in data_labels]

    plot_embeddings(r_vecs, labels, true_labels)
    print("done")

    # """Simple classifier on learned representations"""
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import accuracy_score
    # clf = LogisticRegression(max_iter=10)
    # clf.fit(vecs, labels)
    # preds = clf.predict(vecs)
    # print("Train accuracy:", accuracy_score(labels, preds))


cfg = dict(
    seed=12,
    results_path=Path(f"/home/fatemeh/Downloads/bird/results/chronos2"),
    data_file="/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv",
)
cfg = OmegaConf.create(cfg)
main(cfg)
