import numpy as np
import torch
from torch.utils.data import DataLoader

from behavior import data as bd
from behavior import model as bm
from behavior import utils as bu

seed = 32984
bu.set_seed(seed)


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


def infer_update_classes(df, glen, labels_to_use, checkpoint_file, n_classes):
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
    model = bm.BirdModel(4, 30, n_classes).to(device)
    bm.load_model(checkpoint_file, model, device)
    model.eval()

    # Data
    igs = df[[4, 5, 6, 7]].values.reshape(-1, glen, 4)
    dataset = bd.BirdDataset(igs)
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
