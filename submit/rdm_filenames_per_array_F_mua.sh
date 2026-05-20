#!/bin/bash
#SBATCH --array 1-16
#SBATCH --partition workq
#SBATCH -c 64
#SBATCH --mem 200G
#SBATCH --time 24:00:00
#SBATCH --output=./logs/rdm_fname_F_%A_arr_%a.out
#SBATCH --error=./logs/rdm_fname_F_%A_arr_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=danthes@uos.de


spack load miniconda3
spack load git
eval "$(conda shell.bash hook)"
conda activate amsdrift

monkey="monkeyF"
labels="filenames"
baseline=0
standardize_data=1
dprime_thresh=1.5
run_name=monkeyF_mua_minithings
neural_data='mua'
session_number='0 1 2 3 4 5'
sessions_channel_select="0 1 2 3 4 5"
rois="1 2 3"

const_args="--standardize-data $standardize_data --baseline $baseline --monkey $monkey --good-channel-threshold $dprime_thresh --labels $labels --session-ids-for-channel-selection $sessions_channel_select --run-name $run_name --neural-data $neural_data --session-ids $session_number --monkey $monkey --rois $rois"


echo run name: $run_name
python -u scripts/allMUA_compute_rdm_multi.py --array-indices $SLURM_ARRAY_TASK_ID $const_args
