import argparse
import os
import sys
from os import path
from pprint import pprint

import numpy as np
from ephyslib import crossvalidation
from ephyslib.interactive_utils import show_struct
from macaquethings.data_util.load_data import cli_data_parser, load_data
from macaquethings.data_util.process_data import process_data
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline

# ------------- SETUP

data_cfg = dict(
    monkey="monkeyF",
    labels="category",
    baseline=0,
    session_ids=np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
        ]
    ),
    array_indices=np.arange(16) + 1,
    rois=np.array([3]),
    good_channel_threshold=1.5,
    session_ids_for_channel_selection=np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
        ]
    ),
    neural_data="lfp",
    dataset="allMUA",
    avg_time=10,
    standardize_data=1,
    trial_averaged=False,
)

data_cfg = cli_data_parser(data_cfg)
print(data_cfg)

# ------------------------------ ARGS FOR DECODING

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-name",
    type=str,
    default="default",
    help="subfolder for run. Will not overwrite results with same name.",
)

args, _ = parser.parse_known_args()
decode_cfg = vars(args)

decode_cfg["model"] = RidgeClassifier(
    random_state=42, max_iter=10000, class_weight="balanced"
)
decode_cfg["param_grid"] = {
    "ridgeclassifier__alpha": np.logspace(0, 8, 8),
}

decode_cfg["time_bins_spatiotemporal"] = np.arange(50, 200, 5)

# set the seed and pass it to CV. This allows to get the same splits when
# loading the results
seed = np.random.randint(99999)
decode_cfg["cv"] = crossvalidation.ImageStratifiedShuffleSplit(5, random_state=seed)

# ------------------------------ ARGS FOR DECODING

savedir = path.join("results", "decoding_spatiotemporal", decode_cfg["run_name"])
svgdir = path.join(savedir, "svg")
pngdir = path.join(savedir, "png")

os.makedirs(svgdir, exist_ok=True)
os.makedirs(pngdir, exist_ok=True)

print("running decoding analysis with params:")
print("--------------------------------------------- DATA")
pprint(data_cfg)  # shows content of arrays
print("--------------------------------------------- DECODING")
show_struct(decode_cfg)  # shows shape of arrays
print("---------------------------------------------")

# -------------------------------------- load data

time, X, y, groups, im_number, sess_idx, info, data_strs, h5_handle = load_data(
    data_cfg
)

# apply preprocessing
X, groups = process_data(X, sess_idx, im_number, groups, data_cfg)

roi_str, arr_str, sess_str = data_strs

results_fname = f"{data_cfg['monkey']}-labels_{data_cfg['labels']}-sessions_{sess_str}-rois_{roi_str}-arrays_{arr_str}-baseline_{data_cfg['baseline']}-standardize_{data_cfg['standardize_data']}-{data_cfg['neural_data']}.pkl"

resultspath = path.join(savedir, results_fname)

unique_y = np.sort(np.unique(y))

# check whether this file already exists and exit without error if it does.
if path.exists(resultspath):
    print("RUN ALREADY EXISTS. EXITING.")
    sys.exit(0)

print("will save results to:", resultspath)

#######################################################################
n_chans = X.shape[1]
t_mask = np.isin(time, decode_cfg["time_bins_spatiotemporal"])

Xflat = X[..., t_mask].reshape((X.shape[0], -1))

pca = PCA(n_components=0.9)
pipe = make_pipeline(pca, decode_cfg["model"])

grid_search = GridSearchCV(
    pipe,
    decode_cfg["param_grid"],
    cv=decode_cfg["cv"],
    scoring="accuracy",
    n_jobs=-1,
    refit=False,
    verbose=100,
)

grid_search.fit(Xflat, y, groups=groups)

# fit with found alpha and return scores for all folds
pipe.set_params(**grid_search.best_params_)
crossval_results = cross_validate(
    pipe,
    Xflat,
    y,
    groups=groups,
    cv=decode_cfg["cv"],
    scoring="accuracy",
    n_jobs=-1,
)

split_scores = crossval_results["test_score"]
print(split_scores)

# # select single time point and refit
# Xsel = X[..., np.isin(time, np.arange(0, 200))]
# print(Xsel.shape)
# Xsel = Xsel.reshape((X.shape[0], -1))
# grid_search.fit(Xsel, y, groups=groups)
# grid_search.best_score_
