import itertools
import argparse
from pathlib import Path

import sys
script_path = Path(__file__).parent
exp_path = script_path.parent / "exps"
sys.path.append(str(script_path))
sys.path.append(str(exp_path))
import ss_cluster_behavior as cluster_module
import train as train_module

all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# cfg = train_module.get_config()

# # Exclude one label at a time: exp135-143
# for i, exclude in enumerate(all_labels):
#     cfg.exp = 135 + i 
#     cfg.labels_to_use = sorted(set(all_labels) - {exclude})
#     cfg.model.parameters.out_channels = len(cfg.labels_to_use)
#     print(f"Experiment {cfg.exp}: Excluding label {exclude}")
#     train_module.main(cfg)

# # Exclude two labels at a time: exp144-179
# # pairs = list(itertools.combinations(all_labels, 2))
# pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 9), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 8), (1, 9), (2, 3), (2, 4), (2, 5), (2, 6), (2, 8), (2, 9), (3, 4), (3, 5), (3, 6), (3, 8), (3, 9), (4, 5), (4, 6), (4, 8), (4, 9), (5, 6), (5, 8), (5, 9), (6, 8), (6, 9), (8, 9)]
# cfg.no_epochs = 2000
# for i, exclude in enumerate(pairs): #[(0,1)]
#     cfg.exp = 144 + i 
#     cfg.labels_to_use = sorted(set(all_labels) - set(exclude))
#     cfg.model.parameters.out_channels = len(cfg.labels_to_use)
#     # cfg.data_file = Path("/home/fkarimi/data/bird/starts.csv")
#     # cfg.save_path = Path("/home/fkarimi/exp/bird/runs")
#     print(f"Experiment {cfg.exp}: Excluding label {exclude}")
#     train_module.main(cfg)


cfg = cluster_module.get_config()

# # Exclude one label at a time: exp135-143
# all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]
# for i, exclude in enumerate(all_labels):
#     exp = 135 + i 
#     cfg.lt_labels = sorted(set(all_labels) - {exclude})
#     cfg.model_checkpoint = Path(f"/home/fatemeh/Downloads/bird/result/{exp}_best.pth")
#     cfg.model.name = "small"
#     cfg.model.channel_first = True
#     cfg.out_channel = len(cfg.lt_labels)
#     print(f"Experiment {exp}: Excluding label {exclude}")
#     cluster_module.main(cfg)

# Exclude two labels at a time: exp144-179
# pairs = list(itertools.combinations(all_labels, 2))
pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 9), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 8), (1, 9), (2, 3), (2, 4), (2, 5), (2, 6), (2, 8), (2, 9), (3, 4), (3, 5), (3, 6), (3, 8), (3, 9), (4, 5), (4, 6), (4, 8), (4, 9), (5, 6), (5, 8), (5, 9), (6, 8), (6, 9), (8, 9)]
for i, exclude in enumerate(pairs): #[(0,1)]
    exp = 144 + i
    cfg.lt_labels = sorted(set(all_labels) - set(exclude))
    cfg.model_checkpoint = Path(f"/home/fatemeh/Downloads/bird/result/{exp}_best.pth")
    cfg.model.name = "small"
    cfg.model.channel_first = True
    cfg.out_channel = len(cfg.lt_labels)
    print(f"Experiment {exp}: Excluding label {exclude}")
    cluster_module.main(cfg)

exp_exclude = dict()
all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]
pairs = list(itertools.combinations(all_labels, 1))
for i, exclude in enumerate(pairs):
    exp = 135 + i 
    exp_exclude[exp] = exclude
    print(f"Experiment {exp}: Excluding label {exclude}")

pairs = list(itertools.combinations(all_labels, 2))
for i, exclude in enumerate(pairs):
    exp = 144 + i
    exp_exclude[exp] = exclude
    print(f"Experiment {exp}: Excluding label {exclude}")

# def main():
#     parser = argparse.ArgumentParser(
#         description="Run the script in one of the modes: train, cluster."
#     )

#     parser.add_argument(
#     "mode",
#     choices=["train", "cluster"],
#     help="Mode to run: 'train', 'cluster'."
# )

#     args = parser.parse_args()
#     mode = args.mode

#     # Now you can branch on mode:
#     if mode == "train":
#         print("Starting training...")
#         train_module.run()
#     elif mode == "cluster":
#         print("Running SS Cluster...")
#     else:
#         print("Invalid: use 'train' or 'cluster'")

# if __name__ == "__main__":
#     main()