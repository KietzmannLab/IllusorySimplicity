#!/bin/bash
#SBATCH --partition klab-cpu
#SBATCH -c 32
#SBATCH --mem 196G
#SBATCH --time 04:00:00
#SBATCH --output=./logs/dprime_N_%A.out
#SBATCH --error=./logs/dprime_N_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=danthes@uos.de


spack load miniconda3
spack load git
eval "$(conda shell.bash hook)"
conda activate amsdrift

monkey="monkeyN"
labels="category"
baseline=0
dprime_thresh=-1
sessions_channel_select="0 1 2 3 4 5"
session_number='0 1 2 3 4 5'
run_name="monkeyN"
rois="1 2 3"
arrays="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
neural_data='mua'


const_args="--baseline $baseline --monkey $monkey --good-channel-threshold $dprime_thresh --labels $labels --session-ids-for-channel-selection $sessions_channel_select --run-name $run_name --neural-data $neural_data --rois $rois --session-ids $session_number --monkey $monkey --array-indices $arrays"

python -u scripts/allMUA_dprime_new.py $const_args
