from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from behavior import data as bd
from behavior import model as bm
from behavior import utils as bu
from behavior.utils import n_classes, target_labels, target_labels_names


def temp_model_test(data_path, model_path):
    width = 30

    seed = 1234
    bu.set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    measurements, ldts = bd.load_csv(data_path)
    measurements, ldts = bd.get_specific_labesl(measurements, ldts, target_labels)

    dataset = bd.BirdDataset(measurements, ldts)
    data_loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )

    in_channel = next(iter(data_loader))[0].shape[1]
    model = bm.BirdModel(in_channel, width, n_classes).to(device)
    bm.load_model(model_path, model, device)
    model.eval()

    data, ldts = next(iter(data_loader))
    with torch.no_grad():
        labels = ldts[:, 0]
        labels = labels.to(device)  # N
        data = data.to(device)  # N x C x L
        outputs = model(data)  # N x C
        prob = torch.nn.functional.softmax(outputs, dim=-1).detach()  # N x C
        pred = torch.argmax(outputs.data, 1)  # N

    pred = pred.item()  # 1d. otherwise pred.cpu().numpy()
    return pred


def temp_database_model_test(
    database_url, device_id, start_time, end_time, model_path, glen=20
):
    width = 30

    seed = 1234
    bu.set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # gimus: M x C (C=4), idts: M x C (C=3)
    gimus, idts, _ = bd.get_data(database_url, device_id, start_time, end_time)
    gimus = gimus.reshape(-1, glen, 4)  # N x L x C (N x 20 x 4)
    idts = idts[::glen]  # N X C (N x 20 x 3)

    dataset = bd.BirdDataset(gimus, idts)  # fake label
    data_loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    in_channel = next(iter(data_loader))[0].shape[1]
    model = bm.BirdModel(in_channel, width, n_classes).to(device)
    bm.load_model(model_path, model, device)
    model.eval()

    data, ldts = next(iter(data_loader))
    with torch.no_grad():
        labels = ldts[:, 0]
        labels = labels.to(device)  # N
        data = data.to(device)  # N x C x L
        outputs = model(data)  # N x C
        probs = torch.nn.functional.softmax(outputs, dim=-1).cpu().numpy()  # N x C
        preds = torch.argmax(outputs.data, 1).cpu().numpy()  # N
    max_probs = np.array([probs[i, p] for i, p in enumerate(preds)])
    max_probs = np.round(max_probs, 2)
    return preds, max_probs


"""
CREATE OR REPLACE FUNCTION pytestdatabasemodelcsv()
RETURNS integer AS $$
from behavior import plpython3u as bp
from pathlib import Path
device_id = 534
start_time="2012-06-08 10:28:58"
database_url = "postgresql://username:pass@pub.e-ecology.nl:5432/eecology"
model_path = Path("/scratch/data/45_best.pth")
preds, probs = bp.temp_database_model_test(database_url, device_id, start_time, start_time, model_path)
return int(preds[0])
$$ LANGUAGE plpython3u;

CREATE OR REPLACE FUNCTION pytestmodelcsv()
RETURNS integer AS $$
from behavior import plpython3u as bp
from pathlib import Path
data_path = Path("/scratch/data/manouvre.csv")
model_path = Path("/scratch/data/45_best.pth")
pred = bp.temp_model_test(data_path, model_path)
return pred
$$ LANGUAGE plpython3u;


CREATE OR REPLACE FUNCTION pygetdata()
RETURNS float8 AS $$
import behavior.data as bd
device_id = 534
start_time="2012-06-08 10:28:58"
database_url = "postgresql://username:pass@pub.e-ecology.nl:5432/eecology"
gimus, idts, llat = bd.get_data(database_url, device_id, start_time, start_time)
print(gimus.shape)
return sum(gimus.shape)*.1
$$ LANGUAGE plpython3u;
"""
