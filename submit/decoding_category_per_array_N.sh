#!/bin/bash
#SBATCH --partition klab-cpu
#SBATCH -c 64
#SBATCH --mem 64G
#SBATCH --time 04:00:00
#SBATCH --array=1-16
#SBATCH --output=./logs/dec_cat_N_%A_arr_%a.out
#SBATCH --error=./logs/dec_cat_N_%A_arr_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=danthes@uos.de


spack load miniconda3
spack load git
eval "$(conda shell.bash hook)"
conda activate amsdrift

# array 6 is reported broken for monkeyN, so we skip it
if [ $SLURM_ARRAY_TASK_ID -eq 6 ]
then
    echo "Skipping broken array id 6 for monkeyN"
    exit 0
fi

# there are 6 sessions and 16 rois, map them to the jobs in the array
# account for 1 based indexing for ids (but not for sessions)
array_id=$SLURM_ARRAY_TASK_ID
monkey="monkeyN"
labels="category"
baseline=0
standardize_data=1
dprime_thresh=1.5
sessions_channel_select="0 3 4 5"
run_name="monkeyN_stratifyimages_allsess"
rois="1 2 3"
neural_data='lfp'
session_number='0 3 4 5'

const_args="--standardize-data $standardize_data --baseline $baseline --monkey $monkey --good-channel-threshold $dprime_thresh --labels $labels --session-ids-for-channel-selection $sessions_channel_select --run-name $run_name --neural-data $neural_data --rois $rois --session-ids $session_number --monkey $monkey"

# print the command before submitting
python -u scripts/allMUA_decoding.py \
            --array-indices $array_id \
            $const_args
