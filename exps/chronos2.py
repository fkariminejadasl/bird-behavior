import pandas as pd  # requires: pip install 'pandas[pyarrow]'
from chronos import Chronos2Pipeline
import torch
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path
import gc
from omegaconf import OmegaConf
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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
    handle = model.encoder.final_layer_norm.register_forward_hook(get_activation(activation))


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

    with torch.no_grad():_ = model(x)
    activation[0][:,0,0] # activation[1][:,0,0]
    handle.remove()

def prepare_data(data_file, data_labels):
    df = pd.read_csv(data_file, header=None)
    df = df[df[3].isin(data_labels)]
    all_measurements = df[[4, 5, 6, 7]].values.reshape(-1, 20, 4)
    label_ids = df[[3, 0, 0]].iloc[::20].values
    print(all_measurements.shape)
    del all_measurements, label_ids
    gc.collect()
    # feats = feats.mean(dim=(0, 1))  # 768
    # # feats = feats.mean(dim=1).reshape(-1)


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
                feats = activation.pop() # 
                feats = feats.mean(dim=(0, 1))  # 768
                # feats = feats.mean(dim=1).reshape(-1)
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
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    model = pipeline.model
    activation = []
    handle = model.encoder.final_layer_norm.register_forward_hook(get_activation(activation))
    

    # Data
    df = pd.read_csv(cfg.data_file, header=None)
    df = df[df[3].isin(data_labels)]
    gimus = df[[4, 5, 6, 7]].values.reshape(-1, 20, 4)
    gimus[:,:,:3] = np.clip(gimus[:,:,:3], -2.0, 2.0)

    vecs = []
    for x in tqdm(gimus, total=gimus.shape[0]):
        timestamps = pd.date_range("2000-01-01", periods=20, freq="h")

        context_df = pd.DataFrame(
            {
                "id": ["series_0"] * 20,
                "timestamp": timestamps,
                "ch0": x[:, 0],
                "ch1": x[:, 1],
                "ch2": x[:, 2],
                "ch3": x[:, 3],
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

        act = activation.pop() # (4 x 4 x 768)
        vec = act.mean(dim=(0, 1))  # 768 # # feats = act.mean(dim=1).reshape(-1)
        vecs.append(vec.cpu().numpy())
                
    handle.remove()

    reducer = TSNE(n_components=2, random_state=cfg.seed)
    r_vecs = reducer.fit_transform(vecs)
    labels = df[3].iloc[::20].values

    plot_embeddings(r_vecs, labels, ind2name.values())

cfg = dict(seed=12, results_path=Path(f"/home/fatemeh/Downloads/bird/results/chronos2"), data_file="/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv")
cfg = OmegaConf.create(cfg)
main(cfg)