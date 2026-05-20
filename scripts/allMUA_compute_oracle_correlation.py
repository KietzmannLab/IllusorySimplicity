import sys
import os
from os import path
import numpy as np
import seaborn as sns
import pickle as pkl
from joblib import Parallel, delayed
import argparse
from pprint import pprint
from ephyslib.interactive_utils import show_struct


from macaquethings.data_util.load_data import load_data, cli_data_parser

from ephyslib import preprocessing
from ephyslib.reliability import response_reliability_for_stimuli

sns.set_theme(context="notebook", style="white")

# ------------- SETUP

# --- data config
data_cfg = dict(
    monkey="monkeyF",
    labels="filenames",
    baseline=0,
    session_ids=np.array([0, 1, 2, 3, 4, 5]),
    array_indices=np.arange(16) + 1,
    rois=np.array([3]),
    good_channel_threshold=1.5,
    session_ids_for_channel_selection=np.array([0, 1, 2, 3, 4, 5]),
    neural_data="lfp",
)
data_cfg = cli_data_parser(defaults=data_cfg)  # cmd line can overwrite args

# --- oracle config
bin_centers = np.arange(-50, 400, 1)[:, None]
bin_range = np.array([[0]])  # no temporal averaging

oracle_cfg = dict(
    standardize_data=1,
    time_select=bin_centers + bin_range,
    bin_centers=bin_centers,
    bin_range=bin_range,
    run_name="monkeyF_lfp",
)

# todo: parse rdm args from cmd line
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

time, X, y, groups, im_number, sess_idx, info, data_strs, h5_handle = load_data(
    data_cfg
)
roi_str, arr_str, sess_str = data_strs

savedir = path.join("results", "oracle", oracle_cfg["run_name"])
svgdir = path.join(savedir, "svg")
pngdir = path.join(savedir, "png")
os.makedirs(svgdir, exist_ok=True)
os.makedirs(pngdir, exist_ok=True)

results_fname = f"{data_cfg['monkey']}-labels_{data_cfg['labels']}-sessions_{sess_str}-rois_{roi_str}-arrays_{arr_str}-baseline_{data_cfg['baseline']}-standardize_{oracle_cfg['standardize_data']}.pkl"

resultspath = path.join(savedir, results_fname)

if path.exists(resultspath):
    print("file exists and will not be overwritten. Exiting.")
    sys.exit(0)

unique_labels = np.sort(np.unique(y))


def select_time_and_compute(time_s):

    Xtime = X[..., np.isin(time, time_s)]
    assert np.prod(Xtime.shape) > 0, "at least one axis has no data."
    Xselect = Xtime.mean(-1)  # avg over time window

    if oracle_cfg["standardize_data"]:
        print("standardizing data ...")
        Xselect = preprocessing.zscore_per_sess(Xselect, sess_idx, copy=False)

    oracle_correlations = response_reliability_for_stimuli(Xselect, y, unique_labels)
    return oracle_correlations


oracles = Parallel(n_jobs=-1, verbose=100, temp_folder="./tmp")(
    delayed(select_time_and_compute)(t) for t in oracle_cfg["time_select"]
)
oracles = np.array(oracles)

out = dict(
    oracle=oracles,
    data_cfg=data_cfg,
    oracle_cfg=oracle_cfg,
    oracle_stimuli=unique_labels,
)
pkl.dump(out, open(resultspath, "wb"))
