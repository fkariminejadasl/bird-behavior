#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=00:10:00
#SBATCH -o /home/fkarimi/exp/bird/%x_h100_s_1_%j.out 

# %x is replaced with the script name (without path)
# %j represents the job ID

cd "$HOME/dev/bird-behavior"
echo $(date)
echo $(git log -1 --pretty=%h)

SCRIPT_PATH=$(realpath "$0")  # Get the full path of the script
echo -e "bash file: $SCRIPT_PATH \n"
cat "$SCRIPT_PATH"
echo -e "config: \n"
cat $HOME/dev/bird-behavior/configs/pretrain_memory_load.yaml
echo -e "script \n"
cat $HOME/dev/bird-behavior/scripts/pretrain_memory_load.py

echo "cpu per node: $SLURM_CPUS_ON_NODE"

echo "source $HOME/.bashrc"
source $HOME/.bashrc
conda activate bird
echo "activate my virtual env: $CONDA_DEFAULT_ENV"

echo "start training"
python $HOME/dev/bird-behavior/scripts/pretrain_memory_load.py
echo "end training"

echo $(date)