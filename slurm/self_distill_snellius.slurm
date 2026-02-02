#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=72:00:00
#SBATCH -o /home/fkarimineja/exps/bird/self_distill_1_%j.out

cd "$HOME/dev/bird-behavior"
echo $(date)
echo $(git log -1 --pretty=%h)

echo "bash file: self_distill_1.sh"
cat /home/fkarimineja/exps/bird/self_distill_1.sh
echo "script"
cat /home/fkarimineja/dev/bird-behavior/scripts/train_self_distill.py

echo "cpu per node: $SLURM_CPUS_ON_NODE"

echo "source $HOME/.bashrc"
source $HOME/.bashrc
conda activate bird
echo "activate my virtual env: $CONDA_DEFAULT_ENV"

echo "start training"
python /home/fkarimineja/dev/bird-behavior/scripts/train_self_distill.py
echo "end training"

echo $(date)