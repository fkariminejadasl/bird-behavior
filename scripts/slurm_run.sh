#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=00:10:00
#SBATCH -o /home/fkarimi/exp/bird/p20_h100_s_1_%j.out

cd "$HOME/dev/bird-behavior"
echo $(date)
echo $(git log -1 --pretty=%h)

echo "bash file: p20_4.sh"
cat /home/fkarimi/exp/bird/p20_4.sh
echo "script"
cat /home/fkarimi/dev/bird-behavior/scripts/pretrain_memory_load.py

echo "cpu per node: $SLURM_CPUS_ON_NODE"

echo "source $HOME/.bashrc"
source $HOME/.bashrc
conda activate bird
echo "activate my virtual env: $CONDA_DEFAULT_ENV"

echo "start training"
python /home/fkarimi/dev/bird-behavior/scripts/pretrain_memory_load.py
echo "end training"

echo $(date)