#!/bin/bash
#SBATCH -G 3                  # Number of GPUs
#SBATCH --nodes=1             # Always set to 1!
#SBATCH --gres=gpu:3          # This needs to match num GPUs. default 8
#SBATCH --ntasks-per-node=3   # This needs to match num GPUs. default 8
#SBATCH --mem=80000           # Requested Memory
#SBATCH -p gypsum-rtx8000     # Partition
#SBATCH -t 07-00              # Job time limit
#SBATCH -o slurm-%j.out       # %j = job ID

cd /project/pi_mfiterau_umass_edu/test

eval "$(conda shell.bash hook)"
conda activate base

export NGPU=3
export MASTER_PORT=44147

torchrun --nproc_per_node $NGPU --master_port=$MASTER_PORT run_distributed.py --exp_name test
