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

    input_file = Path(inputs.input_file)
    model_checkpoint = Path(inputs.model_checkpoint)
    save_path = Path(inputs.save_path)

    save_path.mkdir(parents=True, exist_ok=True)

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

    target_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # no Other
    target_labels_names = [ind2name[t] for t in target_labels]
    n_classes = len(target_labels)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = bm.BirdModel(4, 30, n_classes).to(device)
    model.eval()
    bm.load_model(model_checkpoint, model, device)

    dev_st_ends = []
    #  encoding='utf-8-sig' needed because of windows editor
    with open(input_file, "r", encoding='utf-8-sig') as rfile:
        for row in rfile:
            dev, st, en = row.strip().split(",")
            dev_st_ends.append([int(dev), st, en])

    failures = []
    for device_id, start_time, end_time in dev_st_ends:
        try:
            gimus, idts, llat = bd.get_data(
                inputs.database_url, device_id, start_time, end_time
            )
            llat = np.array(llat).reshape(-1, 20, 4)[:, 0, :]
            idts = idts.reshape(-1, 20, 3)[:, 0, :]
            gimus = gimus.reshape(-1, 20, 4)
            print(f"data shape {gimus.shape} for {device_id}, {start_time}, {end_time}")

            infer_dataset = bd.BirdDataset(gimus, idts)

            infer_loader = DataLoader(
                infer_dataset,
                batch_size=len(infer_dataset),
                shuffle=False,
                num_workers=1,
                drop_last=True,
            )

            data, _ = next(iter(infer_loader))
            save_file = save_path / f"{device_id}_{start_time}_{end_time}.csv"
            bu.save_results(
                save_file,
                data,
                idts,
                llat,
                model_checkpoint.stem,
                model,
                device,
                target_labels_names,
            )
        except:
            failure = f"{device_id},{start_time},{end_time}\n"
            failures.append(failure)
            print("failed:\n", failure.strip())

    failed_file = save_path / "failures.csv"
    with open(failed_file, "w") as wfile:
        wfile.writelines(failures)

    print("model parameters: ", sum([p.numel() for p in model.parameters()]))
