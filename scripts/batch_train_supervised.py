from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils import tensorboard
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as tvt2

from behavior import data as bd
from behavior import data_augmentation as bau
from behavior import model as bm
from behavior import model1d as bm1
from behavior import utils as bu
from behavior.utils import new_label_inds

models = {
    "BirdModel": bm.BirdModel,
    "BirdModelWideRF": bm.BirdModelWideRF,
    "BirdModelSmallDilated": bm.BirdModelSmallDilated,
    "ResNet18_1D": bm.ResNet18_1D,
    "BirdModelTransformer": bm.BirdModelTransformer,
    "TransformerEncoderMAE": bm1.TransformerEncoderMAE,
    "BirdModelTransformer_": bm.BirdModelTransformer_,
}


@dataclass
class PathConfig:
    save_path: Path


def build_config(base_config, overrides=None):
    cfg = OmegaConf.create(deepcopy(base_config))
    if overrides is not None:
        cfg = OmegaConf.merge(cfg, overrides)
    cfg_paths = OmegaConf.structured(PathConfig(save_path=Path(cfg.save_path)))
    cfg = OmegaConf.merge(cfg, cfg_paths)
    cfg.min_lr = cfg.max_lr / 10 if cfg.min_lr is None else cfg.min_lr
    return cfg


def main(cfg):
    # Set seed and device
    bu.set_seed(cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Use one of the augmentation. transforms.Compose use all the augmentations.
    # Rotation augmentation: a full random 3D rotation (SO(3)) of the IMU
    # acceleration channels (x, y, z); the GPS 2D speed channel is left unchanged.
    # Motivation: UvA-BiTS/Ornitela tags are mounted at different orientations on
    # the bird (horizontal up to ~70 deg pitch) and Ornitela swaps x/y relative to
    # UvA-BiTS, so the classifier should be invariant to the accelerometer's
    # mounting frame. See "Accelerometer calibration" at
    # https://wiki.e-ecology.nl/index.php/UvA-BiTS_Tracking_Data . Applied to the
    # training set only (see the eval_dataset construction below).
    # These are the batched transforms: they take the whole channel-first
    # (N, C, T) batch on the GPU, which `GpuBatches` below hands them.
    transforms = tvt2.RandomChoice(
        [
            bau.BatchRandomRotation3D(),
            # bau.BatchRandomJitter(sigma=0.05),
            # bau.BatchRandomScaling(sigma=0.05),
            # bau.BatchTimeWarp(sigma=0.05),
            # bau.BatchMagnitudeWarp(sigma=0.05, knot=4),
        ]
    )
    # transforms = None  # set to None to disable augmentation

    # Prepare datasets
    if cfg.valid_file is not None:
        # The transform is not implemented here.
        train_dataset = bd.get_bird_dataset_from_csv(
            cfg.data_file, cfg.labels_to_use, channel_first=True
        )
        eval_dataset = bd.get_bird_dataset_from_csv(
            cfg.valid_file, cfg.labels_to_use, channel_first=True
        )
    else:
        print("No validation file provided, using train dataset for evaluation.")
        # train_dataset, eval_dataset = bd.prepare_train_valid_dataset(
        #     cfg.data_file,
        #     cfg.train_per,
        #     cfg.data_per,
        #     cfg.labels_to_use,
        #     channel_first=True,
        #     transforms=transforms,
        # )

        # Stratified split: ensures each class is represented with the same ratio
        # Note: This implementation is suboptimal—data flows through pandas (read),
        # PyTorch (stratified split), and then back to NumPy (for dataset creation).
        igs, ldts = bd.load_csv_pandas(cfg.data_file, cfg.labels_to_use, glen=20)
        igs = torch.tensor(igs, device=device)
        ldts = torch.tensor(ldts, device=device)
        split_ratios = [cfg.train_per, 1 - cfg.train_per]
        splits = bu.stratified_split(
            ldts[:, 0], split_ratios=split_ratios, seed=cfg.seed
        )
        idx1, idx2 = splits[0], splits[1]
        igs_train = igs[idx1].cpu().numpy()
        igs_eval = igs[idx2].cpu().numpy()
        ldts_train = ldts[idx1].cpu().numpy()
        ldts_valid = ldts[idx2].cpu().numpy()
        # The augmentation is applied per batch by GpuBatches, not per sample.
        # `add_magnitudes` appends mag / dyn_mag / jerk_mag (7 channels instead
        # of 4); it must match `cfg.model.parameters.in_channels`.
        train_dataset = bd.BirdDataset(
            igs_train,
            ldts_train,
            None,
            channel_first=True,
            add_magnitudes=cfg.add_magnitudes,
        )
        # Evaluation data must never be augmented.
        eval_dataset = bd.BirdDataset(
            igs_eval,
            ldts_valid,
            None,
            channel_first=True,
            add_magnitudes=cfg.add_magnitudes,
        )

    # Build the sampler: inversely weight by class frequency
    labels = train_dataset.ldts[:, 0]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]  # one weight per sample
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = bd.GpuBatches(train_dataset, device, cfg.batch_size, transforms)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=len(eval_dataset),
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    # Model setup
    # Number of input channels
    in_channel = train_dataset[0][0].shape[0]  # 3 or 4

    if cfg.model.name not in models:
        raise ValueError(f"Unknown model name: {cfg.model_name}")
    model = models[cfg.model.name](**cfg.model.parameters).to(device)
    print(cfg.model.name)

    # bm.load_model(save_path / f"{exp}_4000.pth", model, device) # start from a checkpoint

    # Loss function and optimizer
    if cfg.use_weighted_loss:
        weights = bd.get_labels_weights(new_label_inds)
        criterion = torch.nn.CrossEntropyLoss(torch.tensor(weights).to(device))
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Select optimizer based on configuration
    if cfg.optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer name: {cfg.optimizer_name}")

    # Select scheduler based on configuration
    if cfg.scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=0.1
        )
    elif cfg.scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.no_epochs, eta_min=cfg.min_lr
        )
    elif cfg.scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, cfg.warmup_epochs, eta_min=cfg.min_lr
        )
    elif cfg.scheduler_name == "SequentialLR":
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1, total_iters=cfg.warmup_epochs
        )
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.no_epochs - cfg.warmup_epochs, eta_min=cfg.min_lr
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[cfg.warmup_epochs],
        )
    else:
        raise ValueError(f"Unknown scheduler name: {cfg.scheduler_name}")

    # Print dataset sizes and device
    len_train, len_eval = len(train_dataset), len(eval_dataset)
    print(
        f"Device: {device}, Train samples: {len_train:,}, Validation samples: {len_eval:,}, "
        f"Train loader batches: {len(train_loader)}, Eval loader batches: {len(eval_loader)}"
    )
    # """
    # Training loop
    best_accuracy = 0
    with tensorboard.SummaryWriter(cfg.save_path / f"tensorboard/{cfg.exp}") as writer:
        for epoch in tqdm.tqdm(range(1, cfg.no_epochs + 1)):
            # tqdm.tqdm(range(4001, no_epochs + 1)): # start from a checkpoint
            start_time = datetime.now()
            print(f"Start time: {start_time}")

            # Train for one epoch
            bm.train_one_epoch(
                train_loader,
                model,
                criterion,
                device,
                epoch,
                cfg.no_epochs,
                writer,
                optimizer,
            )
            accuracy = bm.evaluate(
                eval_loader, model, criterion, device, epoch, cfg.no_epochs, writer
            )

            end_time = datetime.now()
            print(f"End time: {end_time}, Elapsed time: {end_time - start_time}")

            # Update scheduler and log learning rates
            scheduler.step()
            lr_optim = round(optimizer.param_groups[-1]["lr"], 6)
            lr_sched = scheduler.get_last_lr()[0]
            writer.add_scalar("lr/optim", lr_optim, epoch)
            writer.add_scalar("lr/sched", lr_sched, epoch)
            print(
                f"Optimizer LR: {optimizer.param_groups[-1]['lr']:.6f}, "
                f"Scheduler LR: {scheduler.get_last_lr()[0]:.6f}"
            )

            # Save model at intervals
            # if epoch % cfg.save_every == 0:
            #     bm.save_model(cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler)
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # 1-based save for epoch
                bm.save_model(
                    cfg.save_path,
                    cfg.exp,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best=True,
                )
                print(f"Best model accuracy: {best_accuracy:.2f}% at epoch: {epoch}")

    # Save the final model
    # 1-based save for epoch
    # bm.save_model(cfg.save_path, cfg.exp, epoch, model, optimizer, scheduler)
    # """

    bm.load_model(cfg.save_path / f"{cfg.exp}_best.pth", model, device)
    model.eval()
    name = f"{cfg.exp}_{Path(cfg.data_file).stem}"
    fail_path = cfg.save_path / f"failed/{name}"
    fail_path.mkdir(parents=True, exist_ok=True)

    datasets = dict()
    if cfg.valid_file is not None:
        data_files = {
            "train": cfg.data_file,
            "valid": cfg.valid_file,
            "test": cfg.test_file,
        }
        for stage, data_file in data_files.items():
            if data_file is not None:
                # del eval_loader, train_loader, train_dataset, eval_dataset
                dataset = bd.get_bird_dataset_from_csv(
                    data_file, cfg.labels_to_use, channel_first=True
                )
                datasets[stage] = dataset

    else:
        datasets = {"train": train_dataset, "valid": eval_dataset}

    for stage, dataset in datasets.items():
        # Never evaluate on augmented data. The augmentation now lives on the
        # GpuBatches train loader rather than on the dataset, so this only
        # guards the `valid_file` path above, which builds its own datasets.
        dataset.transform = None
        loader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
        )
        label_names = [bu.ind2name[i] for i in cfg.labels_to_use]
        data, ldts = next(iter(loader))
        probs, preds, labels, loss, accuracy = bu.evaluate(
            data, ldts, model, criterion, device
        )
        bu.save_confusion_matrix_other_stats(
            probs,
            preds,
            labels,
            loss,
            accuracy,
            fail_path,
            label_names,
            len(cfg.labels_to_use),
            stage=stage,
        )


def iter_batch_configs(base_config, experiments):
    for experiment in experiments:
        cfg = build_config(base_config, experiment)
        cfg.model.parameters.out_channels = len(cfg.labels_to_use)
        yield cfg


if __name__ == "__main__":
    all_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]

    base_config = {
        # Paths
        "save_path": Path("/home/fatemeh/Downloads/bird/results"),
        "data_file": "/home/fatemeh/Downloads/bird/data/final/starts.csv",
        "valid_file": None,
        "test_file": None,
        # General
        "seed": 32984,
        "exp": 192,
        "num_workers": 1,
        "no_epochs": 4000,
        "save_every": 4000,
        # Data
        "train_per": 0.9,
        "data_per": 1.0,
        "batch_size": None,
        "labels_to_use": all_labels,
        "add_magnitudes": False,  # True -> 7 channels, see bd.add_magnitude_features
        # Training
        "warmup_epochs": 1000,
        "step_size": 2000,
        "max_lr": 3e-4,
        "min_lr": None,
        "weight_decay": 1e-2,
        "use_weighted_loss": False,
        "optimizer_name": "AdamW",
        "scheduler_name": "StepLR",
        # Model
        "model": {
            "name": "BirdModel",
            "parameters": {
                "in_channels": 4,
                "mid_channels": 30,
                "out_channels": 9,
            },
            # Other model options from configs/train.yaml:
            # "name": "ResNet18_1D",
            # "parameters": {"dropout": 0.3, "num_classes": 9},
            # "name": "BirdModelTransformer",
            # "parameters": {"out_channels": 9, "embed_dim": 16, "drop": 0.7},
            # "name": "TransformerEncoderMAE",
            # "parameters": {
            #     "img_size": 20,
            #     "in_chans": 4,
            #     "out_chans": 9,
            #     "embed_dim": 16,
            #     "depth": 1,
            #     "num_heads": 8,
            #     "mlp_ratio": 4,
            #     "drop": 0.0,
            #     "layer_norm_eps": 1e-6,
            # },
            # "name": "BirdModelTransformer_",
            # "parameters": {"in_channels": 4, "out_channels": 9},
        },
    }

    # One entry per training run; each overrides base_config.
    experiments = [
        # {
        #     "exp": 196,
        #     "labels_to_use": all_labels,
        #     "add_magnitudes": False,
        #     "model": {
        #         "name": "BirdModelSmallDilated",
        #         "parameters": {
        #             "in_channels": 4,
        #             "mid_channels": 20,
        #             "out_channels": len(all_labels),
        #             "dropout": 0.15,
        #         },
        #     },
        # },
        {
            "exp": 197,
            "labels_to_use": all_labels,
            "add_magnitudes": True,
            "model": {
                "name": "BirdModelSmallDilated",
                "parameters": {
                    "in_channels": 7,
                    "mid_channels": 20,
                    "out_channels": len(all_labels),
                    "dropout": 0.15,
                },
            },
        },
    ]

    for cfg in iter_batch_configs(base_config, experiments):
        print(f"Experiment {cfg.exp}: {cfg.model.name}, data={cfg.data_file}")
        # import wandb
        # wandb.init(project="small-bird", config=OmegaConf.to_container(cfg, resolve=True))
        main(cfg)

    # Optional parallel version for small models/data.
    # This can run two trainings at the same time, but both jobs may compete for
    # the same GPU memory if only one GPU is available.
    #
    # from concurrent.futures import ProcessPoolExecutor
    #
    # max_parallel_runs = 2
    # configs = list(iter_batch_configs())
    # with ProcessPoolExecutor(max_workers=max_parallel_runs) as executor:
    #     executor.map(main, configs)
