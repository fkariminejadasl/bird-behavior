# bird-behavior

A tool for classifying bird behaviors from sensor data.

## Installation

The minimum required Python version is **3.10**. If you don't have a newer version of Python installed, using **conda** or **miniconda** is the cleanest way to set it up. To install conda or miniconda, follow [these instructions](https://github.com/fkariminejadasl/ml-notebooks/blob/main/tutorial/python.md#setup-python).

The following commands are required when using conda:

```bash
conda create --name bird python=3.10 -y  # Create a virtual environment
conda activate bird                      # Activate the environment
conda deactivate                         # Deactivate the environment
conda remove --name bird --all           # Remove the virtual environment
```

Clone the repository and install the requirements:
```bash
git clone https://github.com/fkariminejadasl/bird-behavior.git
cd bird-behavior
# Ensure the conda environment is activated
# Install the software
pip install -e .
# Install additional requirements for notebooks (if needed)
pip install -e .[notebook]
```

# Usage

## General Users
If you do not have access to our database, you can use your own data in the format described below to classify bird behaviors. You need to download a pre-trained model, which can be found  here.

#### input format
The `data.csv` file is a simple text file where each row contains the following fields, separated by commas:
- device_id
- date_time (format: year-month-day hour:minute:second)
- index (can be 0 or increasing values)
- label (should be -1 indicating no label)
- IMU x, y, z acceleration
- GPS 2D speed
.
Here is an example (also available in `data/data.csv`)`:
```
537,2013-06-06 20:34:11,0,-1,0.36737926,0.03602193,1.59495666,9.84760679
537,2013-06-06 20:34:11,0,-1,0.38796516,0.11902897,1.41765169,9.84760679
```

#### Command to Run
```bash
python scripts/classify_birds.py --data_file /your_path/manouvre.csv  --model_checkpoint /your_path/model.pth --save_path /your_path/result
# Alternatively, with default values in the config file:
python scripts/classify_birds.py  configs/classification.yaml --data_file /your_path/data.csv 
# Or by specifying everything in the config file:
python scripts/classify_birds.py  configs/classification.yaml
```

### IBED Users

Use the `notebooks/classify_birds.ipynb` or follow the instruction below.

#### Input Format
The `input.csv` file is a simple text file where each row contains the following fields, separated by commas:
- device_id
- start_time (format: year-month-day hour:minute:second)
- end_time (format: year-month-day hour:minute:second)

Here is an example (also available in `data/input.csv`):
```
541,2012-05-17 00:00:59,2012-05-17 00:00:59
805,2014-06-05 11:16:27,2014-06-05 11:17:27
```

#### command to run
```bash
python scripts/classify_birds.py --input_file /your_path/input.csv --username your_user --password your_pass  --model_checkpoint /your_path/45_best.pth --save_path /your_path/result
# Alternatively, with default values in the config file:
python scripts/classify_birds.py  configs/classification.yaml --input_file /your_path/input.csv --username your_user --password your_pass 
# Or by specifying everything in the config file:
python scripts/classify_birds.py  configs/classification.yaml
```

