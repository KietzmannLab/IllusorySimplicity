import argparse
import os
import pickle as pkl
import sys
from os import path
from pprint import pprint

import numpy as np
from ephyslib.interactive_utils import show_struct
from ephyslib.reliability import response_reliability_for_electrode_sampled
from macaquethings.data_util.load_data import cli_data_parser, load_data
from macaquethings.data_util.process_data import process_data

# ------------- SETUP

# --- data config
data_cfg = dict(
    monkey="monkeyN",
    labels="filenames",
    baseline=0,
    session_ids=np.array([0, 3, 4, 5]),
    array_indices=np.arange(16) + 1,
    rois=np.array([1, 2, 3]),
    good_channel_threshold=-1,
    session_ids_for_channel_selection=np.array([0, 3, 4, 5]),
    neural_data="lfp",
    dataset="allMUA",
)
data_cfg = cli_data_parser(defaults=data_cfg)  # cmd line can overwrite args

# --- oracle config
oracle_cfg = dict(
    standardize_data=1,
    run_name=f"{data_cfg['monkey']}_{data_cfg['labels']}_allMUA_avg_10ms_lfp",
    nsamples=30,
    parallel_samples=False,
    parallel_bins=False,
    trial_averaged=False,  # must be false, for compatibility with data preprocessing
    avg_time=10,  # window size for averaging in time
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--standardize-data",
    type=int,
    default=oracle_cfg["standardize_data"],
    help="whether to z-score data per session for each bin (type: int. all values > 0 are considered True)",
)
parser.add_argument(
    "--run-name",
    type=str,
    default=oracle_cfg["run_name"],
    help="subfolder for run. Will not overwrite results with same name.",
)

args, _ = parser.parse_known_args()
args = vars(args)

oracle_cfg.update(args)

print("computing oracle correlations with params:")
print("--------------------------------------------- DATA")
pprint(data_cfg)  # shows content of arrays
print("--------------------------------------------- ORACLE")
show_struct(oracle_cfg)  # shows shape of arrays
print("---------------------------------------------")

time, X, y, groups, im_number, sess_idx, info, data_strs, h5_handle = load_data(
    data_cfg
)
roi_str, arr_str, sess_str = data_strs

# process data according to cfg
X, groups = process_data(X, sess_idx, im_number, groups, oracle_cfg)

savedir = path.join("results", "oracle_electrode", oracle_cfg["run_name"])
svgdir = path.join(savedir, "svg")
pngdir = path.join(savedir, "png")
os.makedirs(svgdir, exist_ok=True)
os.makedirs(pngdir, exist_ok=True)

results_fname = f"{data_cfg['monkey']}-labels_{data_cfg['labels']}-sessions_{sess_str}-rois_{roi_str}-arrays_{arr_str}-baseline_{data_cfg['baseline']}-standardize_{oracle_cfg['standardize_data']}.pkl"

resultspath = path.join(savedir, results_fname)

if path.exists(resultspath):
    print("file exists and will not be overwritten. Exiting.")
    sys.exit(0)

oracle_correlations = response_reliability_for_electrode_sampled(
    X,
    y,
    # parallel=oracle_cfg["parallel_samples"],
    parallel=False,
    nsamples=oracle_cfg["nsamples"],
    progress=True,
)

oracle_correlations, oracle_corr_stds = oracle_correlations

out = dict(
    oracle=oracle_correlations,
    std=oracle_corr_stds,
    time=time,
    data_cfg=data_cfg,
    oracle_cfg=oracle_cfg,
)
pkl.dump(out, open(resultspath, "wb"))
