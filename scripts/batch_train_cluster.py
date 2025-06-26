import itertools
import argparse
from pathlib import Path

import sys, os
sys.path.append(os.path.abspath("scripts"))
sys.path.append(os.path.abspath("exps"))
import ss_cluster_behavior as cluster_module
import train as train_module

all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]

cfg = train_module.get_config()

# # Exclude one label at a time: exp135-143
# for i, exclude in enumerate(all_labels):
#     cfg.exp = 135 + i 
#     cfg.labels_to_use = sorted(set(all_labels) - {exclude})
#     cfg.model.parameters.out_channels = len(cfg.labels_to_use)
#     print(f"Experiment {cfg.exp}: Excluding label {exclude}")
#     train_module.main(cfg)

# Exclude two labels at a time: exp144-179
pairs = list(itertools.combinations(all_labels, 2))
for i, exclude in enumerate(pairs): #[(0,1)]
    cfg.exp = 144 + i 
    cfg.labels_to_use = sorted(set(all_labels) - set(exclude))
    cfg.model.parameters.out_channels = len(cfg.labels_to_use)
    cfg.data_file = Path("/home/fkarimi/data/bird/starts.csv")
    cfg.save_path = Path("/home/fkarimi/exp/bird/runs")
    print(f"Experiment {cfg.exp}: Excluding label {exclude}")
    # train_module.main(cfg)


# cfg = cluster_module.get_config()

# # Exclude one label at a time: exp135-143
# all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]
# for i, exclude in enumerate(all_labels):
#     cfg.lt_labels = sorted(set(all_labels) - {exclude})
#     exp = 135 + i 
#     cfg.model_checkpoint = Path(f"/home/fatemeh/Downloads/bird/result/{exp}_best.pth")
#     cfg.model.name = "small"
#     cfg.model.channel_first = True
#     cfg.out_channel = len(cfg.lt_labels)
#     print(f"Experiment {exp}: Excluding label {exclude}")
#     cluster_module.main(cfg)

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