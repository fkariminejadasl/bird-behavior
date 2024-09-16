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

# Set random seeds for reproducibility
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

# Fixed database parameters
DB_HOST = "pub.e-ecology.nl"
DB_PORT = 5432
DB_NAME = "eecology"

# Indices to activity names mapping
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


def process_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as config_file:
        try:
            return yaml.safe_load(config_file)
        except yaml.YAMLError as error:
            print(f"Error reading config file: {error}")


def load_configuration():
    """Load and merge configuration from command-line arguments, config file, and defaults."""
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

    # Optional arguments (command-line key-value pairs) without default values
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
        inputs = process_config(args.config_file)
    else:
        print("No config file provided.")

    # Override with command-line arguments if they are provided
    cmd_args = vars(args)
    cmd_args.pop("config_file", None)  # Remove 'config_file' from cmd_args
    cmd_args = {
        k: v for k, v in cmd_args.items() if v is not None
    }  # Remove None values
    inputs.update(cmd_args)  # Merge command-line arguments into inputs

    # Set default values if not provided in inputs
    defaults = {
        "input_file": Path("/content/input.csv"),
        "data_file": None,
        "model_checkpoint": Path("/content/45_best.pth"),
        "save_path": Path("/content/result"),
    }
    for key, value in defaults.items():
        inputs.setdefault(key, value)

    print("Inputs:")
    for key, value in inputs.items():
        print(f"  {key}: {value}")
    print("\n")

    # Convert inputs to SimpleNamespace for attribute-style access
    inputs = SimpleNamespace(**inputs)

    # Ensure required parameters are present
    if not inputs.model_checkpoint:
        print("Error: Model checkpoint path must be provided.")
    return inputs


def initialize_model(model_checkpoint, n_classes):
    """Initialize the model and load the checkpoint."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = bm.BirdModel(4, 30, n_classes).to(device)
    model.eval()
    bm.load_model(model_checkpoint, model, device)
    print(f"Using device: {device}")
    print("Number of model parameters:", sum(p.numel() for p in model.parameters()))
    return model, device


def load_data_from_csv(data_file):
    """Load data from a CSV file."""
    try:
        print(f"Loading data from {data_file}")
        gimus, idts = bd.load_csv(data_file)
        llats = None  # Since we don't have llats
        if gimus is None or idts is None or len(gimus) == 0 or len(idts) == 0:
            print("Loaded data is empty or None.")
            return None, None, None
        print(f"Data shape: {gimus.shape}")
        return gimus, idts, llats
    except Exception as e:
        print(f"Error loading data from CSV: {e}")
        return None, None, None


def read_device_time_ranges(input_file):
    """Read device IDs and time ranges from the input file."""
    dev_st_ends = []
    try:
        with open(input_file, "r", encoding="utf-8-sig") as rfile:
            for row in rfile:
                try:
                    dev, st, en = row.strip().split(",")
                    dev_st_ends.append([int(dev), st, en])
                except ValueError:
                    print(f"Invalid line in input file: {row.strip()}")
        return dev_st_ends
    except FileNotFoundError:
        print(f"Input file not found: {input_file}")
        return None
    except Exception as e:
        print(f"Error reading input file: {e}")
        return None


def process_and_save_data(
    gimus, idts, llats, model, device, target_labels_names, save_file
):
    """Process data and save the results."""
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


if __name__ == "__main__":
    # Load and merge configuration
    inputs = load_configuration()

    # Prepare paths
    input_file = Path(inputs.input_file) if inputs.input_file else None
    data_file = Path(inputs.data_file) if inputs.data_file else None
    model_checkpoint = Path(inputs.model_checkpoint)
    save_path = Path(inputs.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Prepare labels and model
    target_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # Exclude 'Other'
    target_labels_names = [ind2name[t] for t in target_labels]
    n_classes = len(target_labels)
    model, device = initialize_model(model_checkpoint, n_classes)

    if data_file is not None:
        # Load data from CSV
        gimus, idts, llats = load_data_from_csv(data_file)
        if gimus is not None:
            # Process and save data
            save_file = save_path / "results.csv"
            process_and_save_data(
                gimus, idts, llats, model, device, target_labels_names, save_file
            )
        else:
            print("Data loading failed. Exiting.")
    elif input_file is not None:
        # Ensure username and password are provided
        if not inputs.username or not inputs.password:
            print(
                "Error: Username and password must be provided when accessing the database."
            )
        else:
            # Construct the database URL
            database_url = f"postgresql://{inputs.username}:{inputs.password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            print(f"Loading data from {database_url}")
            # Read device IDs and time ranges
            dev_st_ends = read_device_time_ranges(input_file)
            if dev_st_ends is None:
                print("Failed to read device time ranges. Exiting.")
            else:
                failures = []
                for device_id, start_time, end_time in dev_st_ends:
                    try:
                        gimus, idts, llats = bd.get_data(
                            database_url, device_id, start_time, end_time
                        )
                        if gimus is None or idts is None:
                            raise ValueError("No data retrieved from the database.")
                        llats = (
                            np.array(llats).reshape(-1, 20, 4)[:, 0, :]
                            if llats is not None
                            else None
                        )
                        idts = idts.reshape(-1, 20, 3)[:, 0, :]
                        gimus = gimus.reshape(-1, 20, 4)
                        print(
                            f"Data shape {gimus.shape} for {device_id}, {start_time}, {end_time}"
                        )
                        # Process and save data
                        save_file = (
                            save_path / f"{device_id}_{start_time}_{end_time}.csv"
                        )
                        process_and_save_data(
                            gimus,
                            idts,
                            llats,
                            model,
                            device,
                            target_labels_names,
                            save_file,
                        )
                    except Exception as e:
                        failure = f"{device_id},{start_time},{end_time}\n"
                        failures.append(failure)
                        print(f"Failed to process {failure.strip()}: {e}")
                # Write failures to a file
                if failures:
                    failed_file = save_path / "failures.csv"
                    with open(failed_file, "w") as wfile:
                        wfile.writelines(failures)
    else:
        print("Error: Neither data_file nor input_file is provided.")
