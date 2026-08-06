import numpy as np
import torch
from torch.utils.data import DataLoader

from behavior import data as bd
from behavior import model as bm
from behavior import utils as bu

seed = 32984
bu.set_seed(seed)


MODELS = {
    "BirdModel": bm.BirdModel,
    "BirdModelWideRF": bm.BirdModelWideRF,
    "BirdModelSmallDilated": bm.BirdModelSmallDilated,
}


class Mapper:
    def __init__(self, old2new: dict):
        # old is a list like [0,2,4,5,6,9], new is [0, ..., 5]
        self.old2new = old2new
        self.new2old = {n: o for o, n in old2new.items()}

    def encode(self, orig):
        """Map original labels → 0…K-1 space"""
        return np.array([self.old2new[int(i)] for i in orig])

    def decode(self, chang):
        """Map 0…K-1 predictions back → original labels"""
        return np.array([self.new2old[int(i)] for i in chang])


def build_model(model_name, model_parameters, device):
    return MODELS[model_name](**model_parameters).to(device)


def infer_update_classes(
    df,
    glen,
    labels_to_use,
    checkpoint_file,
    n_classes,
    model_name="BirdModel",
    model_parameters=None,
):
    """
    Inference and update class/confidence columns in app-format data.
    -> df is mutated
    """
    if df.shape[1] < 12:
        raise ValueError(
            "Expected app-format data with 12 columns: "
            "device,date_time,index,gt_label,imu_x,imu_y,imu_z,gps_speed,"
            "class,confidence,lat,lon,altitude"
            " altitude is optional"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    if model_parameters is None:
        model_parameters = {
            "in_channels": 4,
            "mid_channels": 30,
            "out_channels": n_classes,
        }
    model = build_model(model_name, model_parameters, device)
    bm.load_model(checkpoint_file, model, device)
    model.eval()

    # Data. A 7-channel model wants the magnitude channels appended.
    igs = df[[4, 5, 6, 7]].values.reshape(-1, glen, 4)
    add_magnitudes = model_parameters.get("in_channels", 4) == 7
    dataset = bd.BirdDataset(igs, add_magnitudes=add_magnitudes)
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    data = next(iter(loader))

    # Predictions
    probs, preds = bu.inference(data, model, device)
    mapper = Mapper({l: i for i, l in enumerate(labels_to_use)})
    preds = mapper.decode(preds)

    # Update app-format class and confidence columns.
    df[8] = preds[:, np.newaxis].repeat(glen, axis=1).reshape(-1)
    max_probs = np.max(probs, axis=1)
    df[9] = max_probs[:, np.newaxis].repeat(glen, axis=1).reshape(-1)

    return df


# from pathlib import Path
# import pandas as pd

# checkpoint_file = Path(f"/home/fatemeh/Downloads/bird/results/125_best.pth")
# glen=20
# labels_to_use=[0, 1, 2, 3, 4, 5, 6, 8, 9]
# in_channe=4
# width=30
# n_classes=len(labels_to_use)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = bm.BirdModel(4, 30, n_classes).to(device)
# model.load_state_dict(torch.load(checkpoint_file, map_location=device, weights_only=True)["model"])

# df = pd.read_csv("/home/fatemeh/Downloads/bird/data/ssl/gimu_behavior/gull/6004.csv", header=None)
# df = df[(df[0] == 6004) & (df[1] == "2013-07-14 14:26:06")].reset_index(drop=True) # 2013-07-23 07:42:47
# df = df[df[7] < 30.0].copy()
# df[[4, 5, 6]] = df[[4, 5, 6]].clip(-2.0, 2.0)

# igs = df[[4, 5, 6, 7]].values.reshape(-1, glen, 4)
# dataset = bd.BirdDataset(igs)
# data = dataset[0].unsqueeze(0).to(device)  # N x C x L
# model.eval()
# with torch.no_grad():
#     data = data.to(device)  # N x C x L
#     outputs = model(data)  # N x C
#     prob = torch.nn.functional.softmax(outputs, dim=-1)  # N x C
#     pred = torch.argmax(outputs.data, 1)  # N
# prob = prob.cpu().numpy()
# pred = pred.cpu().numpy()
# print(prob, pred)


# loader = DataLoader(
#         dataset,
#         batch_size=len(dataset),
#         shuffle=False,
#         num_workers=1,
#         drop_last=False,
#     )
# data = next(iter(loader))

# probs, preds = bu.inference(data, model, device)
# mapper = Mapper({l: i for i, l in enumerate(labels_to_use)})
# preds = mapper.decode(preds)

# print(preds)
