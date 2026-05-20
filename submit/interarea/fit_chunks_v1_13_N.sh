#!/bin/bash
#SBATCH --partition workq
#SBATCH -c 128
#SBATCH --mem 200G
#SBATCH --time 48:00:00
#SBATCH --output=./logs/interarea_N_v1_13_%A_%a.out
#SBATCH --error=./logs/interarea_N_v1_13_%A_%a.err
#SBATCH --array 0-150

spack load miniconda3
spack load git
eval "$(conda shell.bash hook)"
conda activate amsdrift

start=$((SLURM_ARRAY_TASK_ID * 2))
end=$(((SLURM_ARRAY_TASK_ID + 1) * 2))
stride=2
target_array=13

source_arrays="1 2 3 4 5 7 8 9 10 11 12 13 14 15 16"
sessions="0 3 4 5"
sessions_for_dprime="0 3 4 5"
monkey=monkeyN
neuraldata=lfp
interaction_features="0"

# make sure variables are escaped
run_name="${monkey}_v1_target_array_${target_array}_stride_${stride}_ridgecv_${neuraldata}_threefold/chunk${SLURM_ARRAY_TASK_ID}"
# only call cleaning script for first job in array
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    # clean tmp files
    echo "cleaning /tmp to ensure multiprocessing happiness ..."
    python /share/klab/danthes/danthes/clean_tmp.py
    exit_code=$?
    # if previous exited with error, cancel
    if [ $exit_code -ne 0 ]; then
        exit $exit_code
    fi
    echo "done. moving on to script"
fi


python -u scripts/allMUA_inter_area_fit.py --run-name $run_name --target-times $start $end $stride --delays $stride 81 $stride --source-rois 1 --target-rois 3 --monkey $monkey --session-ids $sessions --session-ids-for-channel-selection $sessions_for_dprime --dataset allMUA --neural-data $neuraldata --source-arrays $source_arrays --target-arrays $target_array --trial-averaged 0 --include-pairwise-interaction-features $interaction_features

# get a variable storing exit code of the previous line
exit_code=$?
# store an empy file in run_dir indicating success if python cmd exits with code 0
if [ $exit_code -eq 0 ]; then
    touch results/inter_area/$run_name/success
fi
# exit with the same error code. (Without this line the script will always report success.)
exit $exit_code
