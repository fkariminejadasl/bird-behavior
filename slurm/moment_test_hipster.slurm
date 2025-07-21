#!/bin/bash
#SBATCH --job-name=moment-test
#SBATCH --gres=gpu:2              # request 2 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # one task per node; accelerate will spawn 2 procs
#SBATCH --cpus-per-task=32
#SBATCH --time=00:05:00
#SBATCH --output=logs/moment_test_%j.log

# module use /cvmfs/software.eessi.io/init/modules
# module load EESSI/2023.06
# module load Python/3.11.5-GCCcore-13.2.0 # Python/3.11.3-GCCcore-12.3.0
# module load CUDA/12.4.0

source $HOME/.bashrc
source $HOME/test/bin/activate
cd $HOME/dev/bird-behavior/
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "Running on $(hostname) with GPUs: $CUDA_VISIBLE_DEVICES"
echo "which accelerate: $(which accelerate)"
echo "Slurm nodelist: $SLURM_JOB_NODELIST"
echo "Slurm nodes: $nodes"


# You *don’t* need to set CUDA_VISIBLE_DEVICES by hand
# Slurm will hand you GPUs and Accelerate will use both.
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/ds.yaml exps/moment_test.py
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/ds.yaml exps/moment_test.py