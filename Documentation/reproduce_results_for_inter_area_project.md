# Steps to obtain inter-area results

## 1. Dataset creation

### Neural Data

See [here](./Documentation/dataset_creation.md).

### Stimulus Information

For creating model RDMs and decoding 'higher-level' categories we created a hierarchical labeling of each stimulus from concrete to abstract. This labeling is stored in `datasets/stimulus_information.csv` and is created by running `scripts/datasets/create_stim_info_dataframe.py`.

## 2. Selecting 'good' channels

Most scripts rely on selecting good channels based on signal amplitude relative to baseline (with d-prime).
The d-prime values are computed in `scripts/allMUA_dprime_new.py`.

Results are stored in `results/dprime_new`. Many analyses expect this information to be available so this script must be run first.

## 3. Decoding analyses

Decoding is performed by running `scripts/allMUA_decoding.py`. Decoding can be performed for each column in `stimulus_information.csv` by specifying its name as `label` in the data cfg.

## 4. Signal Reliability / Oracle Correlation

Oracle correlation is computed separately at each electrode, resolved in time using the script `scripts/allMUA_compute_oracle_correlation_per_electrode.py`.

## 5. Inter-area models

Inter-area models are fit using `scripts/allMUA_inter_area_fit.py`.

## 6. RDMs

RDMs are computed in `scripts/allMUA_compute_rdm_multi.py`.

## 7. Analysis of inter-area models

Subspace RDM trajectories are computed in `scripts/inter_area_analysis/analyse_inter_area_model.py`.

## 8. RNN decoding

The RNN decoder pipeline is available as a notebook in `notebooks/recurrent_decoding.ipynb`. Additionally the pipeline is also available as a script in `scripts/allMUA_decoding_rnn.py`.

## 9. Plotting

Plots are generated using notebooks in `notebooks/figs/`

## 10. Submission and parameters

Submission scripts for analyses and parameters for different conditions are included in `./submit`.
