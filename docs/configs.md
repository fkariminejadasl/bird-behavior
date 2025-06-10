### Python Config

`omegaconf` package: 
- can handle nested configs 
- directly read configs
- return both object and dictionary

#### Convert a Dictionary to OmegaConf Object
```python
from omegaconf import OmegaConf

d = dict(k1=2, k2=4, k3=5)
config = OmegaConf.create(d)
```

#### Access Config

Both as dictionary and as an object.
```python
print(config.k1)  # Output: 2
print(config["k1"])  # Output: 2
```

#### Load OmegaConf from YAML File
```python
config = OmegaConf.load("config.yaml")
print(config)
```

#### Convert OmegaConf Object to YAML String
```python
yaml_str = OmegaConf.to_yaml(config)
print(yaml_str)  # Outputs YAML as a string
```

#### Save OmegaConf Object to YAML File.
```python
yaml_file_path = "config.yaml"
OmegaConf.save(config, yaml_file_path)
```

#### Save YAML String to File (Using yaml Package)
```python
import yaml

yaml_file_path = "config.yaml"
with open(yaml_file_path, "w") as f:
    yaml.safe_dump(yaml.safe_load(yaml_str), f)
```

#### Write YAML Directly (Without yaml Package)
```python
with open(yaml_file_path, "w") as f:
    f.write(yaml_str)
```

#### Convert Dataclass to Python Dictionary
```python
from dataclasses import dataclass, asdict

@dataclass
class Config:
    seed: int = 32984
    exp: int = 114

config_instance = Config()
config_dict = asdict(config_instance)
print(config_dict)  # Output: {'seed': 32984, 'exp': 114}
```

#### Structured Configs and Merge Configs

In the below exampel, `isinstance(cfg_paths.save_path, Path)` is true.
```python
from omegaconf import OmegaConf
from pathlib import Path
@dataclass
class PathConfig:
    save_path: Path
    data_file: Path
cfg = OmegaConf.load("config.yaml")
cfg_paths = OmegaConf.structured(PathConfig(save_path=cfg.save_path, data_file=cfg.data_file))
cfg = OmegaConf.merge(cfg, cfg_paths)
```

#### SimpleNamespace and OmegaConf

SimpleNamespace can make an object from dictionary. It is usefule for reading config files, the same as OmegaConf.

```python
from types import SimpleNamespace
with open('config.yaml', 'r') as file: 
    cfg_dict = yaml.safe_load(file)
cfg = SimpleNamespace(**cfg_dict)
```

But SimpleNamespace can't handle nested configs such as this one;
```bash
model:
  type: CNN
  parameters:
    input_size: 28
    num_classes: 10
    num_filters: 32
    kernel_size: 3
    activation: relu
    optimizer: adam
    learning_rate: 0.001
```

#### References

[omegaconf](https://omegaconf.readthedocs.io/en/latest/usage.html)