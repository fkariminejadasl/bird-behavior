from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from behavior import data as bd
from behavior import model as bm
from behavior import utils as bu
from behavior.utils import n_classes, target_labels, target_labels_names


@pytest.mark.ignore
def test_model_csv():
    root_path = Path("/home/fatemeh/Downloads/bird/data/tests")
    data_path = root_path / "manouvre.csv"
    model_path = root_path / "45_best.pth"
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
    assert pred == 7
    assert target_labels_names[7] == "Manouvre"
