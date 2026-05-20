import argparse
import os
import pickle
import sys
from copy import deepcopy
from os import path

import numpy as np
from ephyslib.connectivity import fit_inter_area_models_over_time
from ephyslib.interactive_utils import show_struct
from macaquethings.data_util.load_data import cli_data_parser, load_data
from macaquethings.data_util.process_data import process_data_for_interarea_fit

# import sklearn
# sklearn.set_config(array_api_dispatch=True)
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures

# ------------- SETUP

# NOTE, this script needs to select two datasets containing different channels. The array_indices and rois fields in the data_cfg dict will be overwritten
data_cfg_defaults = dict(
    monkey="monkeyF",
    labels="category",
    baseline=0,
    session_ids=np.array([0, 1, 2, 3, 4, 5]),
    array_indices=np.arange(16) + 1,
    rois=np.array([1, 2, 3]),
    good_channel_threshold=1.5,
    session_ids_for_channel_selection=np.array([0, 1, 2, 3, 4, 5]),
    neural_data="lfp",
    dataset="allMUA",
)

data_cfg = cli_data_parser(data_cfg_defaults)

inter_area_cfg = {
    "enable_gpu": False,
    "run_name": "test_fixsplit",
    "standardize_data": 1,
    "trial_averaged": 0,
    "avg_time": 10,
    "source_rois": np.array([2]),
    "target_rois": np.array([3]),
    "source_arrays": np.array([15]),
    "target_arrays": np.array([11]),
    "target_times": (0, 350, 2),  # arange
    "delays": (1, 61, 2),  # arange
    "model": RidgeCV,
    "efficient_ridgecv": True,
    "include_pairwise_interaction_features": 0,
    "grid": {
        "alpha": np.logspace(-3, 8, 2048),
        "fit_intercept": False,
        # "solver": ['svd']
    },
    "cv": "fixed_manual_split",
    "cfg_grid_search_dict": {"scoring": "r2", "n_jobs": None, "verbose": 100},
    "cfg_parallel": {
        "n_jobs": 1,
        "return_as": "generator",
        "verbose": 100,  # above 50 prints to stdout. Otherwise prints end up in error log
    },
    "dry_run": False,  # if True, set up folders, print cfg and exit.
}

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default=inter_area_cfg["run_name"])
parser.add_argument(
    "--standardize-data", type=int, default=inter_area_cfg["standardize_data"]
)
parser.add_argument(
    "--trial-averaged", default=inter_area_cfg["trial_averaged"], type=int
)
parser.add_argument(
    "--source-rois", type=int, nargs="+", default=inter_area_cfg["source_rois"]
)
parser.add_argument(
    "--target-rois", type=int, nargs="+", default=inter_area_cfg["target_rois"]
)
parser.add_argument(
    "--source-arrays", type=int, nargs="+", default=inter_area_cfg["source_arrays"]
)
parser.add_argument(
    "--target-arrays", type=int, nargs="+", default=inter_area_cfg["target_arrays"]
)
parser.add_argument(
    "--target-times", type=int, nargs=3, default=inter_area_cfg["target_times"]
)
parser.add_argument("--delays", type=int, nargs=3, default=inter_area_cfg["delays"])
parser.add_argument("--dry-run", action="store_true")
parser.add_argument(
    "--include-pairwise-interaction-features",
    type=int,
    default=inter_area_cfg["include_pairwise_interaction_features"],
)
parser.add_argument("--enable-gpu", action="store_true")

args, _ = parser.parse_known_args()
inter_area_cfg.update(vars(args))

inter_area_cfg["target_times"] = np.arange(*inter_area_cfg["target_times"])
inter_area_cfg["delays"] = np.arange(*inter_area_cfg["delays"])

# ----------------------------------- set up paths

savedir = path.join("results", "inter_area", inter_area_cfg["run_name"])
svgdir = path.join(savedir, "svg")
pngdir = path.join(savedir, "png")

os.makedirs(svgdir, exist_ok=True)
os.makedirs(pngdir, exist_ok=True)

# ------------- LOAD DATA

# create two copies of the data cfg dict and overwrite selected channels.
data_cfg_source = deepcopy(data_cfg)
data_cfg_target = deepcopy(data_cfg)

data_cfg_source["array_indices"] = inter_area_cfg["source_arrays"]
data_cfg_source["rois"] = inter_area_cfg["source_rois"]

data_cfg_target["array_indices"] = inter_area_cfg["target_arrays"]
data_cfg_target["rois"] = inter_area_cfg["target_rois"]


print("******************** CONFIG ***************************")
print(">>> source data config:")
show_struct(data_cfg_source)
print()
print("-------------------------------")
print()
print(">>> target data config:")
show_struct(data_cfg_target)
print()
print("-------------------------------")
print()
print(">>> inter area config:")
show_struct(inter_area_cfg)
print("******************** CONFIG ***************************")

# set up feature expansion
if inter_area_cfg["include_pairwise_interaction_features"]:

    def feature_expansion(X):
        print("source region features will include first-order interaction terms.")
        interactions = PolynomialFeatures(
            degree=(2, 2), interaction_only=True, include_bias=False
        ).fit_transform(X)
        print(f"interaction terms: {interactions.shape}")

        # condense feature space, many interactions will be redundant
        pca = PCA(n_components=0.95)
        interactions = pca.fit_transform(interactions)

        print(f"interaction term pca: {interactions.shape}")
        Xexpand = np.concatenate([X, interactions], axis=1)
        print(f"Expanded feature dataset: {Xexpand.shape}")
        return Xexpand


else:
    feature_expansion = None

if inter_area_cfg["dry_run"]:
    print("DRY RUN. Exiting...")
    sys.exit(0)

time, Xsource, _, groups, im_number, sess_idx, info, data_strs, h5_handle = load_data(
    data_cfg_source
)
source_roi_str, source_arr_str, sess_str = data_strs


_, Xtarget, _, _, _, _, _, data_strs, _ = load_data(data_cfg_target)
target_roi_str, target_arr_str, _ = data_strs

print(f"Loaded data: source  {Xsource.shape}. target: {Xtarget.shape}")

if inter_area_cfg["cv"] == "fixed_manual_split":
    print("computing custom split: Stratify over images and sessions.")
    from ephyslib.crossvalidation import FixedTrainTestSplits
    from sklearn.model_selection import StratifiedKFold
    # compute a fixed split that balances images and sessions
    # the same split is used for all time points and delays

    unique_ims = np.unique(im_number)
    unique_sess = np.unique(sess_idx)
    print(f"Found {len(unique_ims)} unique stimuli in {len(unique_sess)} sessions.")

    # create unique labels for each stimulus per session so that we can stratify over both variables jointly
    # find out the order of magnitude of the number of stimuli
    order = 0
    n = 999
    while n > 9:
        order += 1
        check = 10**order
        n = len(unique_ims) // check  # integer division

    print(f"found unique stimuli on the order of {order} ({10**order})")

    # add session offset to image labels.
    # this will create unique labels for the same images in each session
    sess_idx_multiplier = sess_idx * (10 ** (order + 1))
    unique_sess_ims = sess_idx_multiplier + im_number

    # generate three distinct splits. train and test are flipped so that we
    # train on one fold and leave the remaining n-1 folds for analysis.
    # fix the random seed to ensure that we use the same splits for all runs
    # with the same parameters.
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits = splitter.split(Xsource, unique_sess_ims)

    train_indices = []
    test_indices = []
    for test, train in splits:
        train_indices.append(train)
        test_indices.append(test)
    inter_area_cfg["cv"] = FixedTrainTestSplits(train_indices, test_indices)
    inter_area_cfg["train_splits"] = train_indices
    inter_area_cfg["test_splits"] = test_indices

    print("Set up custom splits.")

Xsource, Xtarget, groups = process_data_for_interarea_fit(
    Xsource, Xtarget, sess_idx, im_number, groups, inter_area_cfg
)

cfg = dict(
    data_cfg_source=data_cfg_source,
    data_cfg_target=data_cfg_target,
    inter_area_cfg=inter_area_cfg,
)
with open(path.join(savedir, "_cfg.pkl"), "wb") as f:
    pickle.dump(cfg, f)

fit_inter_area_models_over_time(
    time=time,
    area2_times=inter_area_cfg["target_times"],
    delays=inter_area_cfg["delays"],
    Xarea1=Xsource,
    Xarea2=Xtarget,
    model_class=inter_area_cfg["model"],
    grid=inter_area_cfg["grid"],
    cv=inter_area_cfg["cv"],
    groups=groups,
    cfg_grid_search_dict=inter_area_cfg["cfg_grid_search_dict"],
    cfg_parallel=inter_area_cfg["cfg_parallel"],
    verbose=True,
    savedir=savedir,
    efficient_ridgecv=inter_area_cfg["efficient_ridgecv"],
    feature_expansion=feature_expansion,
    enable_gpu=inter_area_cfg["enable_gpu"],
    exit_if_exists=False,  # already checked in submission script. Do not enable here as this will exit for partial success caused by HPC instability
)
