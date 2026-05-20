#!/bin/bash
#SBATCH --partition klab-cpu
#SBATCH -c 128
#SBATCH --mem 256G
#SBATCH --time 12:00:00
#SBATCH --array=1-3
#SBATCH --output=./logs/dec_anim_N_%A_roi_%a.out
#SBATCH --error=./logs/dec_anim_N_%A_roi_%a.err

spack load miniconda3
spack load git
eval "$(conda shell.bash hook)"
conda activate amsdrift

# there are 6 sessions and 16 rois, map them to the jobs in the array
# account for 1 based indexing for ids (but not for sessions)
# array 6 is reported broken for monkeyN, exclude it
rois=$SLURM_ARRAY_TASK_ID
array_id='1 2 3 4 5 7 8 9 10 11 12 13 14 15 16'
monkey="monkeyN"
labels="filenames"
baseline=0
standardize_data=1
dprime_thresh=1.5
sessions_channel_select="0 3 4 5"
run_name="monkeyN_stratifyimages_allsess"
neural_data='lfp'
session_number='0 3 4 5'

const_args="--standardize-data $standardize_data --baseline $baseline --monkey $monkey --good-channel-threshold $dprime_thresh --labels $labels --session-ids-for-channel-selection $sessions_channel_select --run-name $run_name --neural-data $neural_data --array-indices $array_id --session-ids $session_number --monkey $monkey"

python -u scripts/allMUA_decoding.py \
            --rois $rois \
            $const_args
