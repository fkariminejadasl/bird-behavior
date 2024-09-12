# bird-behavior
bird behavior

# Installation
The minimum pyton version is 3.10. If there is not a new version of python, conda or miniconda is the cleanest way. To install conda or miniconda follow [this instruction](https://github.com/fkariminejadasl/ml-notebooks/blob/main/tutorial/python.md#setup-python).
These are only command requires in conda, which are different than python venv.
```bash
conda create --name bird python=3.10 -y # create virtualenv
conda activate bird # activate
conda deactivate # deactivate
conda remove --name bird --all # remove the virtualenv
```

Install only requirements:
```bash
git clone https://github.com/fkariminejadasl/bird-behavior.git
cd bird-behavior
# here the conda should be activated
# Install the software
pip install -e .
# Install everything (some requirements are needed to run in notebook)
pip install -e .[notebook]
```

# Usage
Look at `notebooks/classify_birds.ipynb`