# Dataset Creation

## MUA and LFP Datasets

`scripts/datasets/create_h5.py` is used to create h5 datasets from session wise .mat files. These .mat files are the final output of the preprocessing scripts in `./matlab_preprocessing`.
These scripts are adapted from Paolo's preprocessing at the NIN, slightly modified to run on our HPC.

Preprocessing scripts are submitted with `submit_matlab_preproc.sh`.
With options specified in `matlab_preprocessing/_code/meta_preproc.m` (and throughout the scripts called from this file).

The preprocessing pipeline stores intermediate results in the dataset folder structure under `_temps`. When rerunning preprocessing with different parameters
make sure to delete intermediate .mat files to avoid skipping sessions that were already processed.

### MonkeyN

For monkeyN, the last session was recorded twice on a single day. Indexing by date by default concatenates all data from these two sessions to a single session. The script `scripts/datasets/patch_h5_monkeyN.py` adjusts the sess_idx column of the dataset to accurately refflect that the data comes from two separate sessions (sess_idx 4 and 5 are then the two sessions on the same day).
