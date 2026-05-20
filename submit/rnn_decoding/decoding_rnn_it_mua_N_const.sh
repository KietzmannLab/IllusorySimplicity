#!/bin/bash
#SBATCH --partition klab-gpu
#SBATCH --gres gpu:1
#SBATCH -c 16
#SBATCH --mem 128G
#SBATCH --time 02:00:00
#SBATCH --output=./logs/dec_rnn_N_%A.out
#SBATCH --error=./logs/dec_rnn_N_%A.err

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
run_name="monkeyN_rnn"
neural_data="mua"
session_number="0 3 4 5"
array_id="1 2 3 4 5 7 8 9 10 11 12 13 14 15 16"
rois="3"
rnn_config="./config/rnn_decoder_config.yaml"
decode_time_first=70
decode_time_last=171
decode_time_step=10
# avg_input: set to 1 to use averaged input control (no dynamics); also set avg_input_n_steps
avg_input=1
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
