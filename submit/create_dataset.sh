#!/bin/bash
#SBATCH --partition klab-cpu
#SBATCH -c 16
#SBATCH --mem 512G
#SBATCH --time 02:00:00
#SBATCH --output=./logs/%A.out
#SBATCH --error=./logs/%A.err


spack load miniconda3
spack load git
eval "$(conda shell.bash hook)"
conda activate amsdrift

python -u scripts/datasets/create_h5.py
