import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from behavior import data as bd
from behavior import model as bm
from behavior import utils as bu

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)


def process_config(config_path):
    with open(config_path, "r") as config_file:
        try:
            return yaml.safe_load(config_file)
        except yaml.YAMLError as error:
            print(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a config file.")
    parser.add_argument("config_file", type=Path, help="Path to the config file")

    args = parser.parse_args()
    config_path = args.config_file
    inputs = process_config(config_path)
    for key, value in inputs.items():
        print(f"{key}: {value}")
    inputs = SimpleNamespace(**inputs)

    model_checkpoint = Path(inputs.model_checkpoint)
    save_path = Path(inputs.save_path)

    ind2name = {
        0: "Flap",
        1: "ExFlap",
        2: "Soar",
        3: "Boat",
        4: "Float",
        5: "SitStand",
        6: "TerLoco",
        7: "Other",
        8: "Manouvre",
        9: "Pecking",
    }

    width = 30
    # target_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    target_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # no Other
    # target_labels = [0, 2, 3, 4, 5, 6] # no: Exflap:1, Other:7, Manauvre:8, Pecking:9
    # target_labels = [0, 3, 4, 5, 6]  # no: Exflap:1, Soar:2, Other:7, Manauvre:8, Pecking:9
    # target_labels = [0, 2, 4, 5]
    # target_labels = [8, 9]
    # target_labels = [0, 1, 2, 3, 4, 5, 6, 9]  # no Other:7; combine soar:2 and manuver:8
    target_labels_names = [ind2name[t] for t in target_labels]
    n_classes = len(target_labels)
    fail_path = save_path
    fail_path.mkdir(parents=True, exist_ok=True)

    gimus, idts, llat = bd.get_data(
        inputs.database_url, inputs.device_id, inputs.start_time, inputs.end_time
    )
    llat = np.array(llat).reshape(-1, 20, 4)[:, 0, :]
    idts = idts.reshape(-1, 20, 3)[:, 0, :]
    infer_measurements = gimus.reshape(-1, 20, 4)
    print(infer_measurements.shape)

    infer_dataset = bd.BirdDataset(infer_measurements, idts)

    infer_loader = DataLoader(
        infer_dataset,
        batch_size=len(infer_dataset),
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()

    print(f"data shape: {infer_dataset[0][0].shape}")  # 3x20
    in_channel = infer_dataset[0][0].shape[0]  # 3 or 4
    model = bm.BirdModel(in_channel, width, n_classes).to(device)
    model.eval()
    bm.load_model(model_checkpoint, model, device)

    print(device)

    data, _ = next(iter(infer_loader))
    bu.save_results(
        data,
        idts,
        llat,
        model_checkpoint.stem,
        model,
        device,
        fail_path,
        target_labels_names,
    )

    print(sum([p.numel() for p in model.parameters()]))
