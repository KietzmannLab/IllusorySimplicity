import argparse
import os
import pickle as pkl
import sys
from os import path
from pprint import pprint

import numpy as np
import seaborn as sns
from ephyslib.interactive_utils import show_struct
from ephyslib.rdm import compute_robust_rdm
from ephyslib.stimulus_response import compute_noise_covariance
from joblib import Parallel, delayed
from macaquethings.data_util.load_data import cli_data_parser, load_data
from macaquethings.data_util.process_data import process_data
from tqdm import tqdm

sns.set_theme(context="notebook", style="white")

# ------------- SETUP

# --- data config
data_cfg = dict(
    monkey="monkeyF",
    labels="filenames",
    baseline=0,
    session_ids=np.array([0, 1, 2, 3, 4, 5]),
    array_indices=np.arange(16) + 1,
    rois=np.array([1, 2, 3]),
    good_channel_threshold=1.5,
    session_ids_for_channel_selection=np.array([0, 1, 2, 3, 4, 5]),
    neural_data="lfp",
    dataset="allMUA",
)
data_cfg = cli_data_parser(defaults=data_cfg)  # cmd line can overwrite args


# --- rdm config
time_select = np.arange(-50, 400, 2)
rdm_cfg = dict(
    standardize_data=1,
    metric="correlation",
    time_select=time_select,
    avg_time=10,
    navg=10,
    nsamples=1_000,
    run_name=f"{data_cfg['monkey']}_minithings_10msavg",
    whiten=False,
    parallel_samples=True,
    parallel_times=False,
    trial_averaged=False,  # default, needed for process_data
)

# todo: parse rdm args from cmd line
parser = argparse.ArgumentParser()
parser.add_argument(
    "--standardize-data",
    type=int,
    default=rdm_cfg["standardize_data"],
    help="whether to z-score data per session for each bin (type: int. all values > 0 are considered True)",
)
parser.add_argument(
    "--run-name",
    type=str,
    default=rdm_cfg["run_name"],
    help="subfolder for run. Will not overwrite results with same name.",
)
parser.add_argument(
    "--metric",
    type=str,
    default=rdm_cfg["metric"],
    help="metric for RDMs",
)
parser.add_argument(
    "--navg",
    type=int,
    default=rdm_cfg["navg"],
    help="number of stimuli to average for each stimulus in a given sample",
)
parser.add_argument(
    "--nsamples",
    type=int,
    default=rdm_cfg["nsamples"],
    help="how many sample RDMs to compute.",
)

args, _ = parser.parse_known_args()
args = vars(args)

rdm_cfg.update(args)

print("computing RDMs with params:")
print("--------------------------------------------- DATA")
pprint(data_cfg)  # shows content of arrays
print("--------------------------------------------- RDMs")
show_struct(rdm_cfg)  # shows shape of arrays


# ----------------------------------

time, X, y, groups, im_number, sess_idx, info, data_strs, h5_handle = load_data(
    data_cfg
)
roi_str, arr_str, sess_str = data_strs

savedir = path.join("results", "rdm", rdm_cfg["run_name"])
svgdir = path.join(savedir, "svg")
pngdir = path.join(savedir, "png")
os.makedirs(svgdir, exist_ok=True)
os.makedirs(pngdir, exist_ok=True)

results_fname = f"{data_cfg['monkey']}-labels_{data_cfg['labels']}-sessions_{sess_str}-rois_{roi_str}-arrays_{arr_str}-baseline_{data_cfg['baseline']}-standardize_{rdm_cfg['standardize_data']}-metric_{rdm_cfg['metric']}-neural_{data_cfg['neural_data']}.pkl"

resultspath = path.join(savedir, results_fname)

if path.exists(resultspath):
    print("file exists and will not be overwritten. Exiting.")
    sys.exit(0)

unique_labels = np.sort(np.unique(y))

# prepare data
X, groups = process_data(X, sess_idx, im_number, groups, rdm_cfg)


def compute_rdm(time_s):
    Xtime = X[..., np.isin(time, time_s)]
    assert np.prod(Xtime.shape) > 0, "at least one axis has no data."
    Xselect = Xtime.mean(-1)  # avg over time window

    _, counts = np.unique(y, return_counts=True)
    navg = np.min(
        (rdm_cfg["navg"], np.min(counts))
    )  # if classes have missing samples, make sure we do not crash

    if rdm_cfg["whiten"]:
        cov = compute_noise_covariance(Xselect, im_number)
    else:
        cov = None

    rdm = compute_robust_rdm(
        Xselect,
        y,
        unique_labels,
        navg=navg,
        nsamples=rdm_cfg["nsamples"],
        metric=rdm_cfg["metric"],
        whiten=rdm_cfg["whiten"],
        cov=cov,
        progress=False,
        parallel=rdm_cfg["parallel_samples"],
    )
    return rdm


pool = Parallel(
    n_jobs=-1 if rdm_cfg["parallel_times"] else 1,
    verbose=100,
    temp_folder="./tmp",
    return_as="generator",
)
joblist = [delayed(compute_rdm)(t) for t in rdm_cfg["time_select"]]
rdms = np.empty(
    (
        len(rdm_cfg["time_select"]),
        int(len(unique_labels) * (len(unique_labels) - 1) * 0.5),
    )
)
results = pool(joblist)  # this is a generator of results

for i, res in tqdm(enumerate(results), total=len(rdms), file=sys.stdout):
    rdms[i] = res

out = dict(rdms=rdms, data_cfg=data_cfg, rdm_cfg=rdm_cfg, time=rdm_cfg["time_select"])
pkl.dump(out, open(resultspath, "wb"))
