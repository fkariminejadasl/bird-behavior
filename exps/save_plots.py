from pathlib import Path

import torch
from torch.utils.data import DataLoader

from behavior import data as bd
from behavior import model as bm
from behavior import utils as bu
from behavior.utils import n_classes, target_labels

seed = 1234
exp = 107
save_name = f"{exp}"
width = 30
save_path = Path("/home/fatemeh/Downloads/bird/results/")
fail_path = save_path / f"failed/{save_name}"
fail_path.mkdir(parents=True, exist_ok=True)


bu.set_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

measurements, ldts = bd.load_csv(
    "/home/fatemeh/Downloads/bird/data/final/combined_unique.csv"  # s_data.csv"
)
measurements, ldts = bd.get_specific_labesl(measurements, ldts, target_labels)

dataset = bd.BirdDataset(measurements, ldts)
data_loader = DataLoader(
    dataset,
    batch_size=len(dataset),
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

criterion = torch.nn.CrossEntropyLoss()
in_channel = next(iter(data_loader))[0].shape[1]
model = bm.BirdModel(in_channel, width, n_classes).to(device)
bm.load_model(save_path / f"{exp}_best.pth", model, device)
model.eval()

data, ldts = next(iter(data_loader))
bu.save_plots_for_specific_label(
    data, ldts, model, device, fail_path, bu.target_labels_names, query_label="Soar"
)
