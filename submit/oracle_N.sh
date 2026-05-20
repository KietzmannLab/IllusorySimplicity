#!/bin/bash
#SBATCH --partition workq
#SBATCH -c 4
#SBATCH --mem 300G
#SBATCH --time 01:00:00
#SBATCH --output=./logs/oracle_N_%A.out
#SBATCH --error=./logs/oracle_N_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=danthes@uos.de


spack load miniconda3
spack load git
eval "$(conda shell.bash hook)"
conda activate amsdrift


monkey="monkeyN"
labels="filenames"
baseline=0
standardize_data=1
dprime_thresh=-1
sessions_channel_select="0 3 4 5"
session_number='0 3 4 5'
run_name="monkeyN"
rois="1 2 3"
arrays="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
neural_data='lfp'


const_args="--standardize-data $standardize_data --baseline $baseline --monkey $monkey --good-channel-threshold $dprime_thresh --labels $labels --session-ids-for-channel-selection $sessions_channel_select --run-name $run_name --neural-data $neural_data --rois $rois --session-ids $session_number --monkey $monkey --array-indices $arrays"


python -u scripts/allMUA_compute_oracle_correlation_per_electrode.py $const_args "$@"
