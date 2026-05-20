import argparse
import os
import pickle as pkl
import sys
from copy import deepcopy
from os import path
from pprint import pprint

import numpy as np
from ephyslib import crossvalidation
from ephyslib.interactive_utils import show_struct
from macaquethings.data_util.load_data import cli_data_parser, load_data
from macaquethings.data_util.process_data import process_data
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    cross_validate,
)
from tqdm import tqdm

# ------------- SETUP

data_cfg = dict(
    monkey="monkeyF",
    labels="animate",
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

decode_cfg["decoding_times"] = np.arange(-50, 400, 5)
decode_cfg["model"] = RidgeClassifier(
    random_state=42, max_iter=10000, class_weight="balanced"
)
decode_cfg["param_grid"] = {
    "alpha": np.logspace(0, 8, 128),
}

# set the seed and pass it to CV. This allows to get the same splits when
# loading the results
seed = np.random.randint(99999)
decode_cfg["cv"] = crossvalidation.ImageStratifiedShuffleSplit(50, random_state=seed)

# ------------------------------ ARGS FOR DECODING

grid_search = GridSearchCV(
    decode_cfg["model"],
    decode_cfg["param_grid"],
    cv=decode_cfg["cv"],
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
)

savedir = path.join("results", "decoding", decode_cfg["run_name"])
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

# ---------------------------------------- fit the model for each time point

accuracies = np.zeros(decode_cfg["decoding_times"].shape[0], dtype=float)
fold_accuracies = []
best_params = []
best_models = []
confusion_matrices = []

for i, t in enumerate(tqdm(decode_cfg["decoding_times"], file=sys.stdout)):
    time_mask = np.isin(time, t)
    # there should only be a single match
    assert time_mask.sum() == 1, (
        f"should match only a single time point. Matched: {time_mask.sum()}"
    )
    X_time = X[..., time_mask].squeeze()  # trials x channels

    # fit model with grid search CV
    print("grid search...", end=" ")
    grid_search.fit(X_time, y, groups=groups)

    best_model = deepcopy(decode_cfg["model"])
    best_model.set_params(**grid_search.best_params_)
    # grid_search.fit(X_time, y, groups=groups)
    print("cross validation...", end=" ")
    crossval_results = cross_validate(
        best_model,
        X_time,
        y,
        groups=groups,
        cv=decode_cfg["cv"],
        scoring="accuracy",
        n_jobs=-1,
        return_estimator=True,
        return_indices=True,
    )

    split_scores = crossval_results["test_score"]
    cv_estimators = crossval_results["estimator"]
    cv_indices = crossval_results["indices"]

    print()
    print("confusion matrices...", end=" ")
    ytrues, ypreds = [], []
    for est, idx in zip(cv_estimators, cv_indices["test"]):
        Xtest = X_time[idx]
        ytest = y[idx]
        ypred = est.predict(Xtest)
        ytrues.append(ytest)
        ypreds.append(ypred)

    confusions = np.array(
        [
            confusion_matrix(true, pred, labels=unique_y)
            for true, pred in zip(ytrues, ypreds)
        ]
    )
    print("done.")

    # store results
    accuracies[i] = np.mean(split_scores)
    best_params.append(grid_search.best_params_)
    best_models.append(grid_search.best_estimator_)
    fold_accuracies.append(split_scores)
    confusion_matrices.append(confusions)
    print()  # force a newline for tqdm
    print(
        "time window center",
        t.mean(),
        "accuracy:",
        grid_search.best_score_,
        "mean (manual):",
        split_scores.mean(),
        "median (manual):",
        np.median(split_scores),
    )

# save results
results = {
    "times": decode_cfg["decoding_times"],
    "accuracies": accuracies,
    "accuracies_per_fold": np.array(fold_accuracies),
    "confusion_matrices": confusion_matrices,
    "confusion matrix labels": unique_y,
    "best_params": best_params,
    "best_models": best_models,
    "data_cfg": data_cfg,
    "decode_cfg": decode_cfg,
}

with open(resultspath, "wb") as f:
    pkl.dump(results, f)
