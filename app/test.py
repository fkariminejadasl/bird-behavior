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


def load_configuration():
    """Load and merge configuration from command-line arguments, config file, and defaults."""
    parser = argparse.ArgumentParser(
        description="Process a config file or command-line arguments."
    )

    # Optional arguments (command-line key-value pairs) without default values
    parser.add_argument(
        "--input_file",
        type=Path,
        help="Path to the input CSV file",
        default=Path("/home/fatemeh/Downloads/bird/data/classify_bird_3conv/input.csv"),
    )
    parser.add_argument(
        "--model_checkpoint",
        type=Path,
        help="Path to the model checkpoint",
        default=Path(
            "/home/fatemeh/Downloads/bird/data/classify_bird_3conv/45_best.pth"
        ),
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        help="Path to save the results",
        default=Path("/home/fatemeh/Downloads/bird/data/classify_bird_3conv/exp2"),
    )
    parser.add_argument("--username", type=str, help="Database username")
    parser.add_argument("--password", type=str, help="Database password")

    inputs = parser.parse_args()
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
    gimus, idts, llats, model, model_name, device, target_labels_names, save_file
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
            model_name,
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
    input_file = Path(inputs.input_file)
    model_checkpoint = Path(inputs.model_checkpoint)
    save_path = Path(inputs.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Prepare labels and model
    target_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # Exclude 'Other'
    target_labels_names = [ind2name[t] for t in target_labels]
    n_classes = len(target_labels)
    model, device = initialize_model(model_checkpoint, n_classes)

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
                    save_file = save_path / f"{device_id}_{start_time}_{end_time}.csv"
                    process_and_save_data(
                        gimus,
                        idts,
                        llats,
                        model,
                        model_checkpoint.stem,
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
