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
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from behavior import data as bd
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu


def setup_testing_dataloader(test_data_file, data_labels, channel_first):
    df = pd.read_csv(test_data_file, header=None)
    df = df[df[3].isin(data_labels)]
    # GPS 2D speed smaller than 30 m/s
    df = df[df[7] < 30.0].copy()
    # Clip IMU x, y, z values between -2, 2
    df[[4, 5, 6]] = df[[4, 5, 6]].clip(-2.0, 2.0)
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


def compute_labeled_embeddings(loader, model, layer_to_hook, device):
    """
    Compute test embeddings

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - feats: Feature embeddings (N x D)
        - labels: Labels (N)
        - output: Additional model outputs
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
                if cfg.layer_name != "fc":
                    feats = feats[:, 0, :]  # B x L x embed_dim -> B x embed_dim
                else:
                    feats = feats.flatten(1)  # B x embed_dim x 1 -> B x embed_dim
                labels = ldts[:, 0].cpu().numpy()
                feats = feats.detach().cpu().numpy()
                ouput = output.detach().cpu().numpy()

    hook_handle.remove()

    end_time = time.time()
    print(f"Embeddings are loaded in {end_time - start_time:.2f} seconds.")
    print(f"Embedding {feats.shape}, labels {labels.shape}, output {ouput.shape}")
    return feats, labels, ouput


def plot_labeled_embeddings(reduced, labels, true_labels):
    # Colormap issue: Matplotlib treats integer labels as continuous values from [vmin, vmax] to [0, 1].
    # Missing label numbers (e.g., no class 7) cause colors labels 8 and 9 merge into color 8 unless you use a discrete BoundaryNorm.
    # Build boundaries that separate integer classes cleanly, even with gaps
    # e.g., for [0,1,2,3,4,5,6,8,9] => [-0.5, 0.5, 1.5, ..., 7.5, 8.5, 9.5]

    unique_classes = np.unique(labels)
    bounds = np.concatenate((unique_classes - 0.5, [unique_classes[-1] + 0.5]))
    norm = mcolors.BoundaryNorm(bounds, ncolors=len(unique_classes))
    cmap = plt.get_cmap("tab20")
    cmap_dict = {i: mcolors.to_hex(cmap(i)) for i in range(20)}
    custom_cmap = mcolors.ListedColormap([cmap_dict[p] for p in unique_classes])

    fig, ax = plt.subplots()
    title = "Embeddings"
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap=custom_cmap,
        s=5,
        norm=norm,
        zorder=0,
    )
    # Tkinter issue: The issue happens because the Tkinter plotting window’s toolbar crashes
    # when a new colorbar is added. Using fig.colorbar() instead of plt.colorbar() fixes it.
    cbar = fig.colorbar(scatter, ax=ax, ticks=unique_classes, label="Label")
    cbar.set_ticklabels(true_labels)
    plt.title(title)


def tsne_embs(feats):
    # import umap
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    # reduced = reducer.fit_transform(feats)
    reducer = TSNE(n_components=2, random_state=cfg.seed)
    reduced = reducer.fit_transform(feats)
    return reduced


# classification
# ==============
def classification_and_clustering(
    feats,
    labels,
    data_labels,
    results_file,
):
    # Dimensionality reduction for visualization
    reduced = tsne_embs(feats)
    true_labels = [bu.ind2name[i] for i in data_labels]
    plot_labeled_embeddings(reduced, labels, true_labels)
    plt.savefig(results_file, bbox_inches="tight")
    plt.close("all")


def main(cfg):
    # Load model
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
        bm.load_model(cfg.model_checkpoint, model, device)

    if cfg.model.name == "vit":
        model = bm1.build_mae_vit_encoder_from_checkpoint(
            cfg.model_checkpoint, device, cfg
        )

    model.to(device)
    model.eval()
    torch.cuda.empty_cache()
    print("model is loaded")

    # Prepare embeddings
    test_loader = setup_testing_dataloader(
        cfg.test_data_file, cfg.data_labels, cfg.model.channel_first
    )
    layer_to_hook = dict(model.named_modules())[cfg.layer_name]  # fc
    feats, labels, output = compute_labeled_embeddings(
        test_loader, model, layer_to_hook, device
    )
    print("Embeddings are computed")

    if cfg.save_path.exists() is False:
        cfg.save_path.mkdir(parents=True, exist_ok=True)
    name = f"{cfg.model_checkpoint.stem}_{cfg.layer_name}.png"
    save_file = cfg.save_path / name
    classification_and_clustering(
        feats,
        labels,
        cfg.data_labels,
        save_file,
    )


def get_config():
    exp = "self_distill_test"
    cfg = dict(
        data_labels=[0, 1, 2, 3, 4, 5, 6, 8, 9],
        model=dict(
            name="mae",  # "small", "smallemb", "mae", "vit"
            channel_first=False,  # False
        ),
        layer_name="fc",  # (avgpool, fc) small, (fc, norm) vit
        save_path=Path("/home/fatemeh/Downloads/bird/result/embeddings_plot"),
        test_data_file=Path("/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv"),
        # model_checkpoint=Path("/home/fatemeh/Downloads/bird/result/125_best.pth"),
        # model_checkpoint = Path(f"/home/fatemeh/Downloads/bird/result/1discover_2/{exp}_best.pth"),
        # model_checkpoint=Path("/home/fatemeh/Downloads/bird/snellius/p20_4_best.pth"),
        model_checkpoint=Path(
            "/home/fatemeh/Downloads/bird/snellius/self_distill_1_best.pth"
        ),
        exp=exp,
        channel_first=False,
        # model parameters
        # # small model
        # in_channel=4,
        # mid_channel=30,
        # out_channel=9,
        # vit model
        g_len=20,  # 60, 20
        in_channel=4,
        out_channel=256,  # 6, 9 #len(cfg.trained_labels)
        embed_dim=256,  # 256, 16
        depth=6,  # 6, 1
        num_heads=8,
        decoder_embed_dim=256,  # 256, 16
        decoder_depth=6,  # 6, 1
        decoder_num_heads=8,
        mlp_ratio=4,
        drop=0.0,
        layer_norm_eps=1e-6,
        # General
        seed=1234,
        num_workers=1,  # 17 (a_100), 15 (h_100)
        no_epochs=1,
        save_every=200,
        # Data
        train_per=0.9,
        data_per=1.0,
        batch_size=1024,  # 4000
    )
    cfg = OmegaConf.create(cfg)

    return cfg


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = get_config()
    bu.set_seed(cfg.seed)
    acc = main(cfg)
    # fmt: on
