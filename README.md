# bird-behavior
bird behavior

Installation
------------
The minimum pyton version is 3.10. If there is not a new version of python, conda is the cleanest way.
These are only command requires in conda, which are different than python venv.
```bash
conda create --name bird python=3.8 -y # create virtualenv
conda activate bird # activate
conda deactivate # deactivate
conda remove --name bird --all # remove the virtualenv
```

Install only requirements:
```bash
git clone https://github.com/fkariminejadasl/bird-behavior.git
cd bird-behavior
# here the conda should be activated
pip install -r requirements.txt
pip install -e .
```

Usage
-----
```bash
git checkout make_features
```

In python: 
```python
from behavior import data as bd
labels, label_ids, device_ids, time_stamps, all_measurements = bd.read_data(data_file)
```