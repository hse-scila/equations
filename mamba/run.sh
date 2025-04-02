#!/bin/bash
#SBATCH --job-name=train_mamba

module load CUDA/11.6

source /home/aysurkov/.bashrc
conda init
source /home/aysurkov/venv/bin/activate
python /home/aysurkov/equations/mamba/train.py