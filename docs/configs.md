### Python Config

Convert a Dictionary to OmegaConf Object
```python
from omegaconf import OmegaConf

d = dict(k1=2, k2=4, k3=5)
config = OmegaConf.create(d)
```

Access Config:
```python
print(config.k1)  # Output: 2
print(config["k1"])  # Output: 2
```

Load OmegaConf from YAML File
```python
config = OmegaConf.load("config.yaml")
print(config)
```

Convert OmegaConf Object to YAML String
```python
yaml_str = OmegaConf.to_yaml(config)
print(yaml_str)  # Outputs YAML as a string
```

Save OmegaConf Object to YAML File.
```python
yaml_file_path = "config.yaml"
OmegaConf.save(config, yaml_file_path)
```

Save YAML String to File (Using yaml Package)
```python
import yaml

yaml_file_path = "config.yaml"
with open(yaml_file_path, "w") as f:
    yaml.safe_dump(yaml.safe_load(yaml_str), f)
```

Write YAML Directly (Without yaml Package)
```python
with open(yaml_file_path, "w") as f:
    f.write(yaml_str)
```

Convert Dataclass to Python Dictionary
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