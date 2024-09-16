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

    if hasattr(inputs, "data_file") and inputs.data_file:
        # Load from CSV
        data_file = Path(inputs.data_file)
        try:
            # Attempt to load data from CSV
            print(f"Loading data from {data_file}")
            gimus, idts = bd.load_csv(data_file)
            llats = None  # Since we don't have llats
        except FileNotFoundError:
            print(f"Data file not found: {data_file}")
            exit(1)
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            exit(1)

        # Validate loaded data
        if gimus is None or idts is None:
            print("Loaded data is None. Exiting.")
            exit(1)
        if len(gimus) == 0 or len(idts) == 0:
            print("Loaded data is empty. Exiting.")
            exit(1)

        print(f"Data shape: {gimus.shape}")

        infer_dataset = bd.BirdDataset(gimus, idts)

        infer_loader = DataLoader(
            infer_dataset,
            batch_size=len(infer_dataset),
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )

        data, _ = next(iter(infer_loader))
        save_file = save_path / f"results.csv"
        bu.save_results(
            save_file,
            data,
            idts,
            llats,
            model_checkpoint.stem,
            model,
            device,
            target_labels_names,
        )

    else:
        # Read from input_file and download data from database
        input_file = Path(inputs.input_file)
        dev_st_ends = []
        try:
            # Read device IDs and times from input_file
            # encoding='utf-8-sig' needed because of windows editor
            with open(input_file, "r", encoding="utf-8-sig") as rfile:
                for row in rfile:
                    try:
                        dev, st, en = row.strip().split(",")
                        dev_st_ends.append([int(dev), st, en])
                    except ValueError:
                        print(f"Invalid line in input file: {row.strip()}")
        except FileNotFoundError:
            print(f"Input file not found: {input_file}")
            exit(1)
        except Exception as e:
            print(f"Error reading input file: {e}")
            exit(1)

        failures = []
        for device_id, start_time, end_time in dev_st_ends:
            try:

                # Load data from database
                gimus, idts, llats = bd.get_data(
                    inputs.database_url, device_id, start_time, end_time
                )
                llats = np.array(llats).reshape(-1, 20, 4)[:, 0, :]
                idts = idts.reshape(-1, 20, 3)[:, 0, :]
                gimus = gimus.reshape(-1, 20, 4)

                print(
                    f"data shape {gimus.shape} for {device_id}, {start_time}, {end_time}"
                )

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
                    llats,
                    model_checkpoint.stem,
                    model,
                    device,
                    target_labels_names,
                )
            except Exception as e:
                failure = f"{device_id},{start_time},{end_time}\n"
                failures.append(failure)
                print(f"Failed to process {failure.strip()}: {e}")

        failed_file = save_path / "failures.csv"
        with open(failed_file, "w") as wfile:
            wfile.writelines(failures)

    print("model parameters: ", sum([p.numel() for p in model.parameters()]))
