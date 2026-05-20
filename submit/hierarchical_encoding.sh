#!/bin/bash
#SBATCH --partition klab-cpu
#SBATCH -c 256
#SBATCH --mem 1000G
#SBATCH --time 24:00:00
#SBATCH --output=./logs/hierarchical_encoding_%A.out
#SBATCH --error=./logs/hierarchical_encoding_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=danthes@uos.de


spack load miniconda3
spack load git
eval "$(conda shell.bash hook)"
conda activate amsdrift
python -u scripts/allMUA_hierarchical_encoding.py
