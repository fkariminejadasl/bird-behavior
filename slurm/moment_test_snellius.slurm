#!/bin/bash
#SBATCH --job-name=moment-test
#SBATCH --gpus=2
#SBATCH --partition=gpu_a100
#SBATCH --time=00:05:00
#SBATCH --output=logs/moment_exp1_%j.log

module load 2024
module load CUDA/12.6.0

source $HOME/.bashrc
conda activate bird
cd $HOME/dev/bird-behavior/
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "Running on $(hostname) with GPUs: $CUDA_VISIBLE_DEVICES"
echo "which accelerate: $(which accelerate)"
echo "Slurm nodelist: $SLURM_JOB_NODELIST"
echo "Slurm nodes: $nodes"

# You *don’t* need to set CUDA_VISIBLE_DEVICES by hand
# Slurm will hand you GPUs and Accelerate will use both.
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/ds.yaml exps/moment_test.py
# CUDA_VISIBLE_DEVICES=0,1 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/ds.yaml exps/moment_test.py
