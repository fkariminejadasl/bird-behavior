import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt

script_path = Path(__file__).parent
exp_path = script_path.parent / "exps"
sys.path.append(str(script_path))
sys.path.append(str(exp_path))
import ss_cluster_behavior as cluster_module
import train as train_module

# all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# cfg = train_module.get_config()

# # Exclude one label at a time: exp135-143
# pairs = list(itertools.combinations(all_labels, 1))
# for i, exclude in enumerate(pairs):
#     cfg.no_epochs = 4000
#     cfg.exp = 135 + i
#     cfg.labels_to_use = sorted(set(all_labels) - set(exclude))
#     cfg.model.parameters.out_channels = len(cfg.labels_to_use)
#     cfg.save_path = Path("/home/fatemeh/Downloads/bird/result/1discover_2")
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

# # Exclude one label at a time: exp180-184
# all_labels = [0, 2, 4, 5, 6]
# cfg = train_module.get_config()
# pairs = list(itertools.combinations(all_labels, 1))
# for i, exclude in enumerate(pairs):
#     cfg.no_epochs = 2000
#     cfg.exp = 180 + i
#     cfg.labels_to_use = sorted(set(all_labels) - set(exclude))
#     cfg.model.parameters.out_channels = len(cfg.labels_to_use)
#     cfg.save_path = Path("/home/fatemeh/Downloads/bird/result/1discover_2")
#     print(f"Experiment {cfg.exp}: Excluding label {exclude}")
#     train_module.main(cfg)


# # For balanced data: create_balanced_data(df, [0, 2, 4, 5, 6]), min 6: 337
# # Exclude one label at a time: exp185-189
# all_labels = [0, 2, 4, 5, 6]
# cfg = train_module.get_config()
# cfg.data_file="/home/fatemeh/Downloads/bird/data/final/proc2/balanced_02456.csv"
# pairs = list(itertools.combinations(all_labels, 1))
# for i, exclude in enumerate(pairs):
#     cfg.no_epochs = 2000
#     cfg.exp = 185 + i
#     cfg.labels_to_use = sorted(set(all_labels) - set(exclude))
#     cfg.model.parameters.out_channels = len(cfg.labels_to_use)
#     cfg.save_path = Path("/home/fatemeh/Downloads/bird/result/1discover_2")
#     print(f"Experiment {cfg.exp}: Excluding label {exclude}")
#     train_module.main(cfg)

# Clustering (GCD)
# ================

# # Exclude one label at a time: exp135-143
# all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]
# cfg = cluster_module.get_config()
# pairs = list(itertools.combinations(all_labels, 1))
# accs = dict()
# for i, exclude in enumerate(pairs):
#     exp = 135 + i
#     cfg.discover_labels = list(exclude)
#     cfg.lt_labels = sorted(set(all_labels) - set(exclude))
#     cfg.model_checkpoint = Path(
#         f"/home/fatemeh/Downloads/bird/result/1discover_2/{exp}_best.pth"
#     )
#     cfg.model.name = "small"
#     cfg.model.channel_first = True
#     cfg.trained_labels = cfg.lt_labels.copy()  # cfg.all_labels, cfg.lt_labels
#     cfg.out_channel = len(cfg.trained_labels)
#     cfg.n_clusters = len(cfg.all_labels)
#     cfg.layer_name = "fc"  # avgpool, fc
#     cfg.save_path = Path("/home/fatemeh/Downloads/bird/result/1discover_3")
#     cfg.use_unlabel = False
#     cfg.data_file = Path("/home/fatemeh/Downloads/bird/data/ssl_mini")
#     print(f"Experiment {exp}: Excluding label {exclude}")
#     acc = cluster_module.main(cfg)
#     accs[exp] = acc
# print(accs)

# # Exclude two labels at a time: exp144-179
# # pairs = list(itertools.combinations(all_labels, 2))
# # fmt:off
# pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 9), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 8), (1, 9), (2, 3), (2, 4), (2, 5), (2, 6), (2, 8), (2, 9), (3, 4), (3, 5), (3, 6), (3, 8), (3, 9), (4, 5), (4, 6), (4, 8), (4, 9), (5, 6), (5, 8), (5, 9), (6, 8), (6, 9), (8, 9)]
# # fmt:on
# for i, exclude in enumerate(pairs):  # [(0,1)]
#     exp = 144 + i
#     cfg.lt_labels = sorted(set(all_labels) - set(exclude))
#     cfg.model_checkpoint = Path(f"/home/fatemeh/Downloads/bird/result/{exp}_best.pth")
#     cfg.model.name = "small"
#     cfg.model.channel_first = True
#     cfg.out_channel = len(cfg.lt_labels)
#     print(f"Experiment {exp}: Excluding label {exclude}")
#     cluster_module.main(cfg)


# # Exclude one label at a time: exp180-184
# all_labels = [0, 2, 4, 5, 6]
# cfg = cluster_module.get_config()
# pairs = list(itertools.combinations(all_labels, 1))
# accs = dict()
# for i, exclude in enumerate(pairs):
#     exp = 180 + i
#     cfg.discover_labels = list(exclude)
#     cfg.all_labels = all_labels.copy()
#     cfg.lt_labels = sorted(set(cfg.all_labels) - set(exclude))
#     cfg.model_checkpoint = Path(
#         f"/home/fatemeh/Downloads/bird/result/1discover_2/{exp}_best.pth"
#     )
#     cfg.model.name = "small"
#     cfg.model.channel_first = True
#     cfg.trained_labels = cfg.lt_labels.copy()  # cfg.all_labels, cfg.lt_labels
#     cfg.out_channel = len(cfg.trained_labels)
#     cfg.n_clusters = len(cfg.all_labels)
#     cfg.layer_name = "fc"  # avgpool, fc
#     cfg.save_path = Path("/home/fatemeh/Downloads/bird/result/1discover_2")
#     cfg.use_unlabel = False
#     print(f"Experiment {exp}: Excluding label {exclude}")
#     acc = cluster_module.main(cfg)
#     accs[exp] = acc
# print(accs)

# # # For balanced data: create_balanced_data(df, [0, 2, 4, 5, 6]), min 6: 337
# # # Exclude one label at a time: exp185-189
# all_labels = [0, 2, 4, 5, 6]
# cfg = cluster_module.get_config()
# cfg.test_data_file = "/home/fatemeh/Downloads/bird/data/final/proc2/balanced_02456.csv"
# pairs = list(itertools.combinations(all_labels, 1))
# accs = dict()
# for i, exclude in enumerate(pairs):
#     exp = 185 + i
#     cfg.discover_labels = list(exclude)
#     cfg.all_labels = all_labels.copy()
#     cfg.lt_labels = sorted(set(cfg.all_labels) - set(exclude))
#     cfg.model_checkpoint = Path(
#         f"/home/fatemeh/Downloads/bird/result/1discover_2/{exp}_best.pth"
#     )
#     cfg.model.name = "small"
#     cfg.model.channel_first = True
#     cfg.trained_labels = cfg.lt_labels.copy()  # cfg.all_labels, cfg.lt_labels
#     cfg.out_channel = len(cfg.trained_labels)
#     cfg.n_clusters = len(cfg.all_labels)
#     cfg.layer_name = "fc"  # avgpool, fc
#     cfg.save_path = Path("/home/fatemeh/Downloads/bird/result/1discover_2")
#     cfg.use_unlabel = False
#     print(f"Experiment {exp}: Excluding label {exclude}")
#     acc = cluster_module.main(cfg)
#     accs[exp] = acc
# print(accs)


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
