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
            print(f"Error reading config file: {error}")
            exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a config file or command-line arguments."
    )

    # Config file argument (positional)
    parser.add_argument(
        "config_file",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the config file",
    )

    # Optional arguments (command-line key-value pairs)
    parser.add_argument("--database_url", type=str, help="Database URL")
    parser.add_argument("--input_file", type=Path, help="Path to the input CSV file")
    parser.add_argument("--data_file", type=Path, help="Path to the data CSV file")
    parser.add_argument(
        "--model_checkpoint", type=Path, help="Path to the model checkpoint"
    )
    parser.add_argument("--save_path", type=Path, help="Path to save the results")
    parser.add_argument("--username", type=str, help="Database username")
    parser.add_argument("--password", type=str, help="Database password")

    args = parser.parse_args()

    # Initialize inputs dictionary
    inputs = {}

    # Process config file if provided
    if args.config_file is not None:
        config_path = args.config_file
        inputs = process_config(config_path)
    else:
        print("No config file provided.")

    # Override with command-line arguments if they are provided
    cmd_args = vars(args)
    # Remove 'config_file' from cmd_args as it's not a config parameter
    cmd_args.pop("config_file", None)
    # Remove None values (arguments not provided)
    cmd_args = {k: v for k, v in cmd_args.items() if v is not None}

    # Merge command-line arguments into inputs, overriding config file values
    inputs.update(cmd_args)

    # Set default values if not provided in inputs
    defaults = {
        "input_file": Path("/content/input.csv"),
        "model_checkpoint": Path("/content/45_best.pth"),
        "save_path": Path("/content/result"),
        # 'username' and 'password' have no defaults
    }

    for key, value in defaults.items():
        inputs.setdefault(key, value)

    print("Inputs:")
    for key, value in inputs.items():
        print(f"  {key}: {value}")
    print("\n")

    # Convert inputs to SimpleNamespace for attribute-style access
    inputs = SimpleNamespace(**inputs)

    # Access parameters from inputs
    input_file = Path(inputs.input_file) if hasattr(inputs, "input_file") else None
    data_file = Path(inputs.data_file) if hasattr(inputs, "data_file") else None
    model_checkpoint = Path(inputs.model_checkpoint)
    save_path = Path(inputs.save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    # Fixed database parameters
    DB_HOST = "pub.e-ecology.nl"  # database_host
    DB_PORT = 5432  # database_port
    DB_NAME = "eecology"  # database_name

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
    model = bm.BirdModel(4, 30, n_classes).to(device)
    model.eval()
    bm.load_model(model_checkpoint, model, device)
    print(f"Using device: {device}")
    print("Number of model parameters: ", sum([p.numel() for p in model.parameters()]))

    if data_file is not None:
        # Load from CSV
        try:
            print(f"Loading data from {data_file}")
            gimus, idts = bd.load_csv(data_file)
            llats = None  # Since we don't have llats
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

        # Proceed with data processing
        try:
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
        except Exception as e:
            print(f"Error during data processing or saving results: {e}")
            exit(1)
    elif input_file is not None:
        # Ensure username and password are provided
        if not hasattr(inputs, "username") or not hasattr(inputs, "password"):
            print(
                "Error: Username and password must be provided when accessing the database."
            )
            exit(1)

        username = inputs.username
        password = inputs.password

        # Construct the database_url
        database_url = (
            f"postgresql://{username}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

        print(f"Loading data from {database_url}")
        dev_st_ends = []
        try:
            # Read device IDs and times from input_file
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
                gimus, idts, llats = bd.get_data(
                    database_url, device_id, start_time, end_time
                )
                llats = np.array(llats).reshape(-1, 20, 4)[:, 0, :]
                idts = idts.reshape(-1, 20, 3)[:, 0, :]
                gimus = gimus.reshape(-1, 20, 4)

                print(
                    f"Data shape {gimus.shape} for {device_id}, {start_time}, {end_time}"
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
    else:
        print("Error: Neither data_file nor input_file is provided.")
        exit(1)
