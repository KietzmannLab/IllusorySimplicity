#!/bin/bash
#SBATCH --partition klab-gpu
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem 64G
#SBATCH --time 02:00:00
#SBATCH --array=13-16
#SBATCH --output=./logs/dec_rnn_single_array_N_%A_%a.out
#SBATCH --error=./logs/dec_rnn_single_array_N_%A_%a.err

spack load miniconda3
spack load git
eval "$(conda shell.bash hook)"
conda activate amsdrift

# array 6 is broken for monkeyN, excluded
monkey="monkeyN"
labels="category"
baseline=0
standardize_data=1
dprime_thresh=1.5
sessions_channel_select="0 3 4 5"
run_name="monkeyN_rnn_single_array"
neural_data="lfp"
session_number="0 3 4 5"
array_id=$SLURM_ARRAY_TASK_ID
rois="3"
rnn_config="./config/rnn_decoder_config_s.yaml"
decode_time_first=50
decode_time_last=250
decode_time_step=10
# avg_input: set to 1 to use averaged input control (no dynamics); also set avg_input_n_steps
avg_input=0
avg_input_n_steps=20

avg_input_args=""
if [ "$avg_input" -eq 1 ]; then
    avg_input_args="--avg-input --avg-input-n-steps $avg_input_n_steps"
fi

python -u scripts/allMUA_decoding_rnn.py \
    --standardize-data $standardize_data \
    --baseline $baseline \
    --monkey $monkey \
    --good-channel-threshold $dprime_thresh \
    --labels $labels \
    --session-ids-for-channel-selection $sessions_channel_select \
    --run-name $run_name \
    --neural-data $neural_data \
    --array-indices $array_id \
    --session-ids $session_number \
    --rois $rois \
    --rnn-config $rnn_config \
    --decode-time-first $decode_time_first \
    --decode-time-last $decode_time_last \
    --decode-time-step $decode_time_step \
    $avg_input_args
