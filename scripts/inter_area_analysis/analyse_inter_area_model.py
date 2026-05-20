import argparse
import os
import pickle
import sys
from glob import glob
from os import path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ephyslib.connectivity import (
    load_memmap_from_filename,
    select_data_for_inter_area_fit,
)
from ephyslib.rdm import compute_robust_rdm
from ephyslib.stimulus_response import compute_stimulus_responses
from joblib import Parallel, delayed
from macaquethings.data_util.load_data import load_data
from macaquethings.data_util.process_data import process_data_for_interarea_fit
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.linalg import orth
from scipy.spatial.distance import cdist
from tqdm import tqdm

#  -------------------------------------------------------------------------------
# PARAMETERS AND DATA SELECTION

# --- inter-area model selection
run_name = "monkeyF_v4_target_array_11_stride_2_ridgecv_lfp_threefold"
t_model = 122
d_model = 35

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default=run_name, type=str)
parser.add_argument("--time", default=t_model, type=int)
parser.add_argument("--delay", default=d_model, type=int)
parser.add_argument("--exist-ok", action="store_true")

args = parser.parse_args()

run_name = args.run_name
t_model = args.time
d_model = args.delay
exist_ok = args.exist_ok

print("STARTING ANALYSIS ...")
print("ARGUMENTS:")
print(f">>> run name: {run_name}")
print(f">>> target time: {t_model}")
print(f">>> target delay: {d_model}")


# --- RDM parameters
rdm_times = np.arange(
    -50, 450, 2
)  # time points for which to compute RDMs. Longer than for regular RDM computation to account for delays
metric = "correlation"
nsamples = 1_000
navg = 10 if run_name.startswith("monkeyF") else 7  # monkeyN has fewer trials available

n_worker_rdm = 16

# --- Select DNN for RDM model correlation

model_rdm_folder = path.join("datasets", "TIMM", "resnet18")
model_rdm_name = "rdms-resnet18-metric_cosine-normalization_None-pca_1000.pkl"


#  ------------------------------------------------------------------------------
# PLOTTING DEFAULTS


DPI = 200


def figure_style(font_size=7):
    """
    Set style for plotting figures
    """
    sns.set(
        style="ticks",
        context="paper",
        font="sans-serif",
        rc={
            "font.size": font_size,
            "figure.titlesize": font_size,
            "figure.labelweight": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1,
            "lines.markersize": 3,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "savefig.transparent": True,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "legend.fontsize": font_size,
            "legend.title_fontsize": font_size,
            "legend.frameon": False,
        },
    )
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["figure.dpi"] = DPI


figure_style()


#  ------------------------------------------------------------------------------
# UTILITY


import logging

# Define ANSI escape code for orange (bright yellow)
ORANGE = "\033[33;1m"
RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        if record.levelno == logging.WARNING:
            msg = f"{ORANGE}{msg}{RESET}"
        return msg


# Set up logger
logger = logging.getLogger("warn_logger")
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.WARNING)


def warn(msg):
    logger.warning(msg)


def save_as_pickle(data, fpath):
    with open(fpath, "wb") as f:
        pickle.dump(data, f)
    print("saved data to:", fpath)


#  ------------------------------------------------------------------------------
# LOAD INTER AREA PERFORMANCES

print("\n" + "=" * 60)
print("STEP 1: Loading inter-area performances...")
print("=" * 60)

inter_area_dir = path.join("results", "inter_area")
rundir = path.join(inter_area_dir, run_name)

# create analysis dir
analysis_dir = path.join(rundir, "analysis")

os.makedirs(analysis_dir, exist_ok=True)

# create subdir for time and delay
subspace_analysis_dir = path.join(analysis_dir, f"subspace_t-{t_model}_d-{d_model}")

# if --exist-ok  has not been set,
# check whether the analysis dir exists and contains an (empty) file
# named SUCCESS. This is created at the end of this script to
# indicate the analysis has completed successfully.
# If it exists, exit without error.
if path.isdir(subspace_analysis_dir) and not exist_ok:
    # check whether there is a file called SUCCESS
    success_fpath = path.join(subspace_analysis_dir, "SUCCESS")
    success_exists = path.isfile(success_fpath)
    if success_exists:
        print("****** ANALYSIS DIR:")
        print(subspace_analysis_dir)
        print("ANALYSIS WAS ALREADY MARKED COMPLETE, EXITING.")
        sys.exit(0)  # do not error on early exit

os.makedirs(subspace_analysis_dir, exist_ok=True)


chunks = [f for f in os.listdir(rundir) if f.startswith("chunk")]
params, inter_area_results, recurrent_results = [], [], []
for i, chunk in enumerate(tqdm(chunks)):
    chunkdir = path.join(rundir, chunk)
    if i == 0:
        cfg = pickle.load(open(path.join(chunkdir, "_cfg.pkl"), "rb"))
    if path.isfile(
        path.join(chunkdir, "success")
    ):  # only load chunks that were successfully completed
        recurrent_file = [
            x for x in os.listdir(chunkdir) if x.endswith(".npy") and "recurrent" in x
        ][0]
        inter_area_file = [
            x for x in os.listdir(chunkdir) if x.endswith(".npy") and "inter_area" in x
        ][0]
        recur_result = np.array(
            load_memmap_from_filename(path.join(chunkdir, recurrent_file))
        )
        inter_result = np.array(
            load_memmap_from_filename(path.join(chunkdir, inter_area_file))
        )
        param = np.load(path.join(chunkdir, "parameters_time_area2_delay.npy"))
        params.append(param)
        inter_area_results.append(inter_result)
        recurrent_results.append(recur_result)

inter_area_results = np.concatenate(inter_area_results, axis=0)
recurrent_results = np.concatenate(recurrent_results, axis=0)
params = np.concatenate(params, axis=0)


print("loaded results:")
print(inter_area_results.shape, recurrent_results.shape, params.shape)


#  -------------------------------------------------------------------------------
# PLOT INTER-AREA RESULTS

print("\n" + "=" * 60)
print("STEP 2: Plotting inter-area results...")
print("=" * 60)

time_area2 = params[:, 0]
delay = params[:, 1]
time_area1 = params[:, 0] - params[:, 1]

assert not np.isnan(recurrent_results).any()
assert not np.isnan(inter_area_results).any()

recurrent_has_neg_r2 = (recurrent_results < 0).any()
inter_area_has_neg_r2 = (inter_area_results < 0).any()

if recurrent_has_neg_r2:
    warn("recurrent results have negative r^2 values.")

if inter_area_has_neg_r2:
    warn("inter-area results have negative r^2 values.")


def plot_inter_area_performance_surface(time, delay, perfs):
    avg_perf = perfs.mean(axis=(1, 2))
    fig = plt.figure(figsize=(3.5, 2), dpi=DPI)  # half width
    plt.tricontourf(time, delay, avg_perf, levels=100, cmap="magma")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")  # equal aspect ratio
    plt.xlabel("time (ms)")
    plt.ylabel("delay")
    plt.colorbar(label="r^2")
    return fig, {"time": time, "delay": delay, "avg_perf": avg_perf}


fig_perf_recur, fig_perf_recur_data = plot_inter_area_performance_surface(
    time_area2, delay, recurrent_results
)
fig_perf_inter, fig_perf_inter_data = plot_inter_area_performance_surface(
    time_area2, delay, inter_area_results
)
fig_perf_granger, fig_perf_granger_data = plot_inter_area_performance_surface(
    time_area2, delay, inter_area_results - recurrent_results
)

fig_perf_recur.savefig(path.join(analysis_dir, "inter_area_performance_r2_recur.svg"))
save_as_pickle(
    fig_perf_recur_data, path.join(analysis_dir, "inter_area_performance_r2_recur.pkl")
)

fig_perf_inter.savefig(path.join(analysis_dir, "inter_area_performance_r2_inter.svg"))
save_as_pickle(
    fig_perf_inter_data, path.join(analysis_dir, "inter_area_performance_r2_inter.pkl")
)

fig_perf_granger.savefig(
    path.join(analysis_dir, "inter_area_performance_r2_granger.svg")
)
save_as_pickle(
    fig_perf_granger_data,
    path.join(analysis_dir, "inter_area_performance_r2_granger.pkl"),
)


# add a marker for selected time and delay
ax = fig_perf_granger.gca()
ax.scatter([t_model], [d_model], color="r", edgecolor="white", linewidth=0.2)
fig_perf_granger.savefig(
    path.join(subspace_analysis_dir, "inter_area_performance_r2_granger.svg")
)


#  ------------------------------------------------------------------------------
# SELECT A TIME POINT, LOAD MODEL AND EXTRACT WEIGHTS

print("\n" + "=" * 60)
print("STEP 3: Selecting time point and extracting model weights...")
print("=" * 60)

#  ------------------------------------------extract the weight matrices


# get the models for a time point
modelfile = glob(
    f"**/time_area2_{str(t_model)}_delay_{d_model}.joblib", root_dir=rundir
)
assert len(modelfile) == 1, "glob match for model is not unique"
modelfile = modelfile[0]
model_results = joblib.load(path.join(rundir, modelfile))
models = model_results["models"]["inter_area"]

all_weights = np.array([m.coef_ for m in models])
print(
    "loaded weights have shapes:", all_weights.shape, "[fold, n_target, n_predictors]"
)

# full model includes recurrent inputs, get number of channels from source region
n_folds, n_target_chans, n_predictor_chans = all_weights.shape
n_source_chans = n_predictor_chans - n_target_chans

# get inter-area weight matrices (excluding recurrent predictors)
all_inter_weights = all_weights[..., :n_source_chans]


#  ------------------------------------------ get cv splits


train_splits = cfg["inter_area_cfg"]["train_splits"]
test_splits = cfg["inter_area_cfg"]["test_splits"]


#  ------------------------------------------ get cv splits
# -- load the neural data
#
# variables without suffix _t indicate data for the time and delay at which the model was fit
# variables with suffix refer to the whole time course


time, Xsource, y, groups, im_number, sess_idx, info, data_strs, h5_handle = load_data(
    cfg["data_cfg_source"], root="."
)

_, Xtarget, _, _, _, _, _, data_strs, _ = load_data(cfg["data_cfg_target"], root=".")

# apply same processing as during inter-area fit
# this returns the full time courses
Xsource_t, Xtarget_t, groups = process_data_for_interarea_fit(
    Xsource, Xtarget, sess_idx, im_number, groups, cfg["inter_area_cfg"]
)

# select the time points that the model was fit on
Xsource, Xrecur, Xtarget, _, _, _ = select_data_for_inter_area_fit(
    Xsource_t, Xtarget_t, t_model - d_model, t_model, time
)

print("time course data:")
print("source:", Xsource_t.shape)
print("target:", Xtarget_t.shape)
print()
print("selected time point data")
print("source:", Xsource.shape)
print("target:", Xtarget.shape)
print("recurrent", Xrecur.shape)


#  ------------------------------------------------------------------------------
# PCA ANALYSIS, ESTIMATE DIMENSIONALITIES FOR MODEL TIME

print("\n" + "=" * 60)
print("STEP 4: Running PCA analysis to estimate dimensionalities...")
print("=" * 60)

from ephyslib.pca import cvPCA
from ephyslib.subspace import compute_reduced_rank_regression_curves

# compute separately over folds set up during inter-area fit

vars_cv_source = []
vars_nocv_source = []
vars_cv_target = []
vars_nocv_target = []

for i, (train_idx, test_idx) in enumerate(zip(train_splits, test_splits)):
    print(f"Fold: {i + 1} / {n_folds}")

    # get pca and crossvalidated pca for fold
    Xsource_fold = Xsource[test_idx]
    Xtarget_fold = Xtarget[test_idx]
    im_number_fold = im_number[test_idx]

    vars_cv_fold_source, vars_nocv_fold_source = cvPCA(Xsource_fold, im_number_fold)
    vars_cv_fold_target, vars_nocv_fold_target = cvPCA(Xtarget_fold, im_number_fold)

    vars_cv_source.append(vars_cv_fold_source.mean(axis=0))
    vars_nocv_source.append(vars_nocv_fold_source.mean(axis=0))
    vars_cv_target.append(vars_cv_fold_target.mean(axis=0))
    vars_nocv_target.append(vars_nocv_fold_target.mean(axis=0))

vars_cv_source = np.array(vars_cv_source)
vars_nocv_source = np.array(vars_nocv_source)
vars_cv_target = np.array(vars_cv_target)
vars_nocv_target = np.array(vars_nocv_target)

print("crossvalidated PCAs source:", vars_cv_source.shape)
print("crossvalidated PCAs target:", vars_cv_target.shape)

print("regular PCAs source:", vars_nocv_source.shape)
print("regular PCAs target:", vars_nocv_target.shape)

signal_dim_results = {
    "variances_source_cvPCA": vars_cv_source,
    "variances_source_PCA": vars_nocv_source,
    "variances_target_cvPCA": vars_cv_target,
    "variances_target_PCA": vars_nocv_target,
}

save_as_pickle(
    signal_dim_results, path.join(subspace_analysis_dir, "signal_dim_data.pkl")
)


# --------------------- plot

signal_dims_99 = {
    "source_nocv": [],
    "target_nocv": [],
    "source_cv": [],
    "target_cv": [],
}

fig_source_pca = plt.figure(figsize=(3.5, 2))
dims = np.arange(vars_nocv_source.shape[1]) + 1
for i, vars in enumerate(vars_nocv_source):
    cumvars = np.cumsum(vars)
    n99 = dims[np.where(cumvars > (0.99 * cumvars[-1]))[0][0]]

    signal_dims_99["source_nocv"].append(n99)

    plt.plot(dims, cumvars, color="tab:blue")
    plt.axvline(n99, color="r", label=f"fold {i + 1}: {n99} comp. 99% var")


plt.legend()
plt.xlabel("n components")
plt.ylabel("cumulative variance explained")
plt.title(f"Source signal dimensionality ({t_model - d_model}ms)")
fig_source_pca.savefig(
    path.join(subspace_analysis_dir, "dimensionality_source_nocv.svg")
)


fig_target_pca = plt.figure(figsize=(3.5, 2))
dims = np.arange(vars_nocv_target.shape[1]) + 1
for i, vars in enumerate(vars_nocv_target):
    cumvars = np.cumsum(vars)
    n99 = dims[np.where(cumvars > (0.99 * cumvars[-1]))[0][0]]

    signal_dims_99["target_nocv"].append(n99)

    plt.plot(dims, cumvars, color="tab:blue")
    plt.axvline(n99, color="r", label=f"fold {i + 1}: {n99} comp. 99% var")

plt.legend()
plt.xlabel("n components")
plt.ylabel("cumulative variance explained")
plt.title(f"Target signal dimensionality ({t_model}ms)")
fig_target_pca.savefig(
    path.join(subspace_analysis_dir, "dimensionality_target_nocv.svg")
)

fig_source_cvpca = plt.figure(figsize=(3.5, 2))
dims = np.arange(vars_cv_source.shape[1]) + 1
for i, vars in enumerate(vars_cv_source):
    cumvars = np.cumsum(vars)
    n99 = dims[np.where(cumvars > (0.99 * cumvars[-1]))[0][0]]

    signal_dims_99["source_cv"].append(n99)

    plt.plot(dims, cumvars, color="tab:blue")
    plt.axvline(n99, color="r", label=f"fold {i + 1}: {n99} comp. 99% var")

plt.legend()
plt.xlabel("n components")
plt.ylabel("cumulative variance explained")
plt.title(f"Source signal dimensionality (cvPCA) ({t_model - d_model}ms)")
fig_source_cvpca.savefig(
    path.join(subspace_analysis_dir, "dimensionality_source_cv.svg")
)

fig_target_cvpca = plt.figure(figsize=(3.5, 2))
dims = np.arange(vars_cv_target.shape[1]) + 1
for i, vars in enumerate(vars_cv_target):
    cumvars = np.cumsum(vars)
    n99 = dims[np.where(cumvars > (0.99 * cumvars[-1]))[0][0]]

    signal_dims_99["target_cv"].append(n99)

    plt.plot(dims, cumvars, color="tab:blue")
    plt.axvline(n99, color="r", label=f"fold {i + 1}: {n99} comp. 99% var")

plt.legend()
plt.xlabel("n components")
plt.ylabel("cumulative variance explained")
plt.title(f"Target signal dimensionality (cvPCA) ({t_model}ms)")
fig_target_cvpca.savefig(
    path.join(subspace_analysis_dir, "dimensionality_target_cv.svg")
)

save_as_pickle(
    signal_dims_99,
    path.join(subspace_analysis_dir, "dimensionality_estimates_99perc_var.pkl"),
)

#  ------------------------------------------------------------------------------
# REDUCED RANK REGRESSION, GET RANK CURVES AND PROJECTION MATRICES

print("\n" + "=" * 60)
print("STEP 5: Computing reduced rank regression curves and projections...")
print("=" * 60)

from ephyslib.subspace import (
    compute_projection_matrix_onto_n_columns,
)

reduced_rank_r2s_train = []
reduced_rank_r2s_test = []
reduced_rank_decomps = []

for i, (train_idx, test_idx) in enumerate(zip(train_splits, test_splits)):
    print(f"Fold: {i + 1} / {n_folds}")

    Xsource_train_fold = Xsource[train_idx]
    Xsource_test_fold = Xsource[test_idx]

    Xtarget_train_fold = Xtarget[train_idx]
    Xtarget_test_fold = Xtarget[test_idx]

    # predict with weight matrix for fold
    W_fold = all_inter_weights[i]

    Xpred_train_fold = Xsource_train_fold @ W_fold.T
    Xpred_test_fold = Xsource_test_fold @ W_fold.T

    # compute curves
    r2s_train_fold, r2s_test_fold, dims, decomp_fold = (
        compute_reduced_rank_regression_curves(
            Xpred_train_fold, Xtarget_train_fold, Xpred_test_fold, Xtarget_test_fold
        )
    )

    # store
    reduced_rank_r2s_train.append(r2s_train_fold)
    reduced_rank_r2s_test.append(r2s_test_fold)
    reduced_rank_decomps.append(decomp_fold)

reduced_rank_r2s_train = np.array(reduced_rank_r2s_train)
reduced_rank_r2s_test = np.array(reduced_rank_r2s_test)

# get best avg. dimensionality by crossvalidation
best_rank_train = dims[np.argmax(np.mean(reduced_rank_r2s_train, axis=0))]

best_rank_test = dims[np.argmax(np.mean(reduced_rank_r2s_test, axis=0))]

print(
    f"Best (crossvalidated) ranks: train - {best_rank_train}, test - {best_rank_test}"
)


# get projection for each fold
target_low_rank_projections = []

for i in range(n_folds):
    P_fold = compute_projection_matrix_onto_n_columns(
        reduced_rank_decomps[i]["Vh"].T, best_rank_test
    )
    target_low_rank_projections.append(P_fold)

target_low_rank_projections = np.array(target_low_rank_projections)

# apply projection to inter-area weight matrices
# NOTE: by convention, sklearn weight matrices are defined as [n_target, n_predictor]
# here we apply the projection to the transposed weight matrix and then transpose back
# to stay with this convention
inter_weights_low_dim = np.array(
    [(W.T @ P).T for W, P in zip(all_inter_weights, target_low_rank_projections)]
)

# get source region readout projections
# this is a projection onto the basis of the low rank weight matrices
readout_basis_projections = []
for W in inter_weights_low_dim:
    W_basis = orth(W.T)
    P_source = W_basis @ W_basis.T
    readout_basis_projections.append(P_source)

readout_basis_projections = np.array(readout_basis_projections)


#  ------------------------------------------ sanity checks
# assert that matrix ranks are correct

for P in readout_basis_projections:
    assert np.linalg.matrix_rank(P) == best_rank_test, (
        "source readout projection does not have the correct rank."
    )
    assert (np.array(P.shape) == n_source_chans).all(), (
        "source readout projection matrix does not have the correct shape."
    )

for W in inter_weights_low_dim:
    assert (np.array(W.shape) == np.array([n_target_chans, n_source_chans])).all(), (
        "low-rank inter-area weight matrix does not have the correct shape."
    )
    assert np.linalg.matrix_rank(W) == best_rank_test, (
        "low-rank weight matrix does not have the correct rank."
    )

# save to disk
reduced_rank_results = {
    "reduced_rank_r2s_train": reduced_rank_r2s_train,
    "reduced_rank_r2s_test": reduced_rank_r2s_test,
    "svd_decompositions": reduced_rank_decomps,
    "best_rank_train": best_rank_train,
    "best_rank_test": best_rank_test,
    "train_indices": train_splits,
    "test_indices": test_splits,
    "target_low_rank_projections": target_low_rank_projections,
    "readout_subspace_projections": readout_basis_projections,
    "reduced_rank_Ws": inter_weights_low_dim,
}

save_as_pickle(
    reduced_rank_results, path.join(subspace_analysis_dir, "reduced_rank_data.pkl")
)


# --------------------- plot

fig_rrr_curves = plt.figure(figsize=(3.5, 2))
dims = np.arange(reduced_rank_r2s_test.shape[1]) + 1
for rsquares in reduced_rank_r2s_test:
    plt.plot(dims, rsquares, color="tab:blue", alpha=0.3)
plt.plot(dims, reduced_rank_r2s_test.mean(axis=0), color="tab:blue")
plt.axvline(best_rank_test, color="r", label=f"best rank: {best_rank_test} dims.")
plt.legend()
plt.xlabel("model rank")
plt.ylabel("coefficient of determination")
plt.title("Reduced Rank Regression performance")
fig_rrr_curves.savefig(path.join(subspace_analysis_dir, "reduced_rank_r2_curves.svg"))


#  -------------------------------------------------------------------------------
# COMPUTE RDM TIME COURSES

print("\n" + "=" * 60)
print("STEP 6: Computing RDM time courses...")
print("=" * 60)

# compute four sets of RDMS:
# - for the full source time course
# - for the full target time course
# - for the readout subspace source time course
# - for the low-rank prediction of the target time course

# for all RDMs, crossvalidation folds are treated separately, such that the
# output shapes will be: [n_folds, n_times, rdm_size]


unique_ims = np.sort(np.unique(im_number))
n_ims = len(unique_ims)
rdm_size = n_ims * (n_ims - 1) // 2
n_times = len(rdm_times)

source_rdms = np.empty((n_folds, n_times, rdm_size))
target_rdms = np.empty_like(source_rdms)
source_readout_sub_rdms = np.empty_like(source_rdms)
target_prediction_rdms = np.empty_like(source_rdms)

sig_norms_full = np.empty((n_folds, n_times))
sig_norms_readout = np.empty((n_folds, n_times))
sig_norms_null = np.empty((n_folds, n_times))

sig_norms_full_stim_avg = np.empty((n_folds, n_times))
sig_norms_readout_stim_avg = np.empty((n_folds, n_times))
sig_norms_null_stim_avg = np.empty((n_folds, n_times))

source_rdms.fill(np.nan)
target_rdms.fill(np.nan)
source_readout_sub_rdms.fill(np.nan)
target_prediction_rdms.fill(np.nan)

sig_norms_full.fill(np.nan)
sig_norms_readout.fill(np.nan)
sig_norms_null.fill(np.nan)

sig_norms_full_stim_avg.fill(np.nan)
sig_norms_readout_stim_avg.fill(np.nan)
sig_norms_null_stim_avg.fill(np.nan)


def select_data_and_compute_rdms(
    time, t_rdm, Xsource_t_fold, Xtarget_t_fold, im_number_fold
):
    assert ((time == t_rdm).sum()) == 1, (
        f"expected a unique time point, matched {(time == t_rdm).sum()}"
    )
    # select data for time point
    Xsource_fold_select = Xsource_t_fold[
        ..., time == t_rdm
    ].squeeze()  # squeeze unitary time axis
    Xtarget_fold_select = Xtarget_t_fold[
        ..., time == t_rdm
    ].squeeze()  # squeeze unitary time axis

    Xreadout = (
        Xsource_fold_select @ P_readout_sub_fold
    )  # project source data onto readout subspace
    Xnull = Xsource_fold_select - Xreadout  # project source data onto readout nullspace
    Xpred = (
        Xsource_fold_select @ W_lowrank_fold.T
    )  # project source data onto target space

    stimuli = np.sort(np.unique(im_number_fold))
    stim_responses_full = compute_stimulus_responses(
        Xsource_fold_select, im_number_fold, stimuli
    )
    stim_responses_readout = stim_responses_full @ P_readout_sub_fold
    stim_responses_null = stim_responses_full - stim_responses_readout

    # compute norms
    norm_full = np.linalg.norm(Xsource_fold_select, axis=1).mean()
    norm_readout = np.linalg.norm(Xreadout, axis=1).mean()
    norm_null = np.linalg.norm(Xnull, axis=1).mean()

    norm_full_stim_resp = np.linalg.norm(stim_responses_full, axis=1).mean()
    norm_readout_stim_resp = np.linalg.norm(stim_responses_readout, axis=1).mean()
    norm_null_stim_resp = np.linalg.norm(stim_responses_null, axis=1).mean()

    # compute RDMs
    rdm_source = compute_robust_rdm(
        Xsource_fold_select,
        im_number_fold,
        unique_ims,
        navg=navg,
        nsamples=nsamples,
        metric=metric,
        parallel=False,
    )
    rdm_target = compute_robust_rdm(
        Xtarget_fold_select,
        im_number_fold,
        unique_ims,
        navg=navg,
        nsamples=nsamples,
        metric=metric,
        parallel=False,
    )
    rdm_sub = compute_robust_rdm(
        Xreadout,
        im_number_fold,
        unique_ims,
        navg=navg,
        nsamples=nsamples,
        metric=metric,
        parallel=False,
    )
    rdm_pred = compute_robust_rdm(
        Xpred,
        im_number_fold,
        unique_ims,
        navg=navg,
        nsamples=nsamples,
        metric=metric,
        parallel=False,
    )

    return (
        rdm_source,
        rdm_target,
        rdm_sub,
        rdm_pred,
        norm_full,
        norm_readout,
        norm_null,
        norm_full_stim_resp,
        norm_readout_stim_resp,
        norm_null_stim_resp,
    )


with Parallel(
    n_jobs=n_worker_rdm, verbose=0, return_as="generator"
) as pool:  # use the same pool for all RDM computations
    for i, test_idx in enumerate(test_splits):
        print(f"fold {i + 1} / {n_folds} ...")
        # select test data for fold
        Xsource_t_fold = Xsource_t[test_idx]
        Xtarget_t_fold = Xtarget_t[test_idx]
        im_number_fold = im_number[test_idx]

        # get projection matrices
        P_readout_sub_fold = readout_basis_projections[i]
        W_lowrank_fold = inter_weights_low_dim[i]

        args_iter = []
        for t in rdm_times:
            args_iter.append((time, t, Xsource_t_fold, Xtarget_t_fold, im_number_fold))

        res_generator = pool(
            delayed(select_data_and_compute_rdms)(*args) for args in args_iter
        )  # submit jobs to pool

        # collect results
        for j, res in enumerate(
            tqdm(res_generator, total=len(args_iter), file=sys.stdout)
        ):
            (
                rdm_source,
                rdm_target,
                rdm_sub,
                rdm_pred,
                norm_full,
                norm_readout,
                norm_null,
                norm_full_stim,
                norm_readout_stim,
                norm_null_stim,
            ) = res
            source_rdms[i, j] = rdm_source
            target_rdms[i, j] = rdm_target
            source_readout_sub_rdms[i, j] = rdm_sub
            target_prediction_rdms[i, j] = rdm_pred

            sig_norms_full[i, j] = norm_full
            sig_norms_readout[i, j] = norm_readout
            sig_norms_null[i, j] = norm_null

            sig_norms_full_stim_avg[i, j] = norm_full_stim
            sig_norms_readout_stim_avg[i, j] = norm_readout_stim
            sig_norms_null_stim_avg[i, j] = norm_null_stim

assert not np.isnan(source_rdms).any(), "results array was not filled completely."
assert not np.isnan(target_rdms).any(), "results array was not filled completely."
assert not np.isnan(source_readout_sub_rdms).any(), (
    "results array was not filled completely."
)
assert not np.isnan(target_prediction_rdms).any(), (
    "results array was not filled completely."
)

assert not np.isnan(sig_norms_full).any(), "results array was not filled completely."
assert not np.isnan(sig_norms_readout).any(), "results array was not filled completely."
assert not np.isnan(sig_norms_null).any(), "results array was not filled completely."

rdm_results = {
    "stimuli": unique_ims,
    "time": rdm_times,
    "rdms_source": source_rdms,
    "rdms_target": target_rdms,
    "rdms_source_readout_subspace": source_readout_sub_rdms,
    "rdms_target_prediction": target_prediction_rdms,
    "rdm_cfg": {"metric": metric, "nsamples": nsamples, "navg": navg},
}

norm_results = {
    "time": rdm_times,
    "norm_full": sig_norms_full,
    "norm_readout": sig_norms_readout,
    "norm_null": sig_norms_null,
    "norm_stim_avg_full": sig_norms_full_stim_avg,
    "norm_stim_avg_readout": sig_norms_readout_stim_avg,
    "norm_stim_avg_null": sig_norms_null_stim_avg,
}

save_as_pickle(rdm_results, path.join(subspace_analysis_dir, "rdms.pkl"))
save_as_pickle(norm_results, path.join(subspace_analysis_dir, "norms.pkl"))


#  ------------------------------------------------------------------------------
# Correlate with DNN RDMs

print("\n" + "=" * 60)
print("STEP 7: Correlating RDMs with DNN model RDMs...")
print("=" * 60)

with open(path.join(model_rdm_folder, model_rdm_name), "rb") as f:
    model_rdm_data = pickle.load(f)


def correlate_rdm_movie_with_models(rdm_timecourse, target_rdms, model_keys):
    print(f"time course shape: {rdm_timecourse.shape}")
    # extract model rdms from dict
    models = np.array([target_rdms[key] for key in model_keys])
    print(f"models shape: {models.shape}")
    return 1 - cdist(rdm_timecourse, models, metric="correlation")


model_keys = model_rdm_data["selected_nodes"]
model_rdm_dict = model_rdm_data["rdms"]


rdm_corrs_source = np.empty((n_folds, source_rdms.shape[1], len(model_keys)))
rdm_corrs_source.fill(np.nan)

rdm_corrs_target = np.empty((n_folds, target_rdms.shape[1], len(model_keys)))
rdm_corrs_target.fill(np.nan)

rdm_corrs_sub = np.empty((n_folds, source_readout_sub_rdms.shape[1], len(model_keys)))
rdm_corrs_sub.fill(np.nan)

rdm_corrs_pred = np.empty((n_folds, target_prediction_rdms.shape[1], len(model_keys)))
rdm_corrs_pred.fill(np.nan)

for i in range(n_folds):
    rdm_corrs_source[i] = correlate_rdm_movie_with_models(
        source_rdms[i], model_rdm_dict, model_keys
    )
    rdm_corrs_target[i] = correlate_rdm_movie_with_models(
        target_rdms[i], model_rdm_dict, model_keys
    )
    rdm_corrs_sub[i] = correlate_rdm_movie_with_models(
        source_readout_sub_rdms[i], model_rdm_dict, model_keys
    )
    rdm_corrs_pred[i] = correlate_rdm_movie_with_models(
        target_prediction_rdms[i], model_rdm_dict, model_keys
    )


rdm_corr_results = {
    "stimuli": unique_ims,
    "time": rdm_times,
    "rdm_cfg": {"metric": metric, "nsamples": nsamples, "navg": navg},
    "model_folder": model_rdm_folder,
    "model_file": model_rdm_name,
    "model_keys": model_keys,
    "layer_indices": model_rdm_data["node_indices"],
    "rdm_corrs_source": rdm_corrs_source,
    "rdm_corrs_target": rdm_corrs_target,
    "rdm_corrs_sub": rdm_corrs_sub,
    "rdm_corrs_pred": rdm_corrs_pred,
}

save_as_pickle(
    rdm_corr_results,
    path.join(subspace_analysis_dir, f"rdm_corrs_{model_rdm_name}.pkl"),
)


# --------------------- plot

xlims = (
    rdm_times[0],
    rdm_times[-1] + d_model,
)  # fix xlims to include the same data range for all panels

max_corr = np.concatenate(
    [
        rdm_corrs_source.flatten(),
        rdm_corrs_target.flatten(),
        rdm_corrs_sub.flatten(),
        rdm_corrs_pred.flatten(),
    ]
).max()

ylims = (0, max_corr)


colors = np.array(
    sns.color_palette("Blues", len(model_keys) + 1)[1:]
)  # exclude first color, since it is too faint
# sort colors by  layer position
layer_idx = np.argsort(model_rdm_data["node_indices"])
colors = colors[layer_idx]

# cmap for colorbar
cmap = ListedColormap(colors)
bounds = np.arange(len(colors) + 1)
norm = BoundaryNorm(bounds, cmap.N)


fig_rdm_corrs_source = plt.figure(figsize=(3.5, 2))
avg_corrs_source = rdm_corrs_source.mean(axis=0)

for corrs, color, name, idx in zip(
    avg_corrs_source.T, colors, model_keys, model_rdm_data["node_indices"]
):
    plt.plot(rdm_times, corrs, label=f"{idx}: {name}", color=color)
plt.xlabel("time (ms)")
plt.ylabel("pearson correlation")

plt.axvline(t_model - d_model, color="r")

cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=plt.gca(),
)

cb.ax.minorticks_off()
cb.set_ticks(np.arange(len(model_rdm_data["node_indices"])) + 0.5)  # center each tick
cb.set_ticklabels(model_keys)
cb.ax.tick_params(labelsize=5)

plt.xlim(xlims)
plt.ylim(ylims)

plt.title("Source Signal")

fig_rdm_corrs_source.savefig(
    path.join(subspace_analysis_dir, "rdm_correlations_source.svg")
)


fig_rdm_corrs_target = plt.figure(figsize=(3.5, 2))
avg_corrs_target = rdm_corrs_target.mean(axis=0)

for corrs, color, name, idx in zip(
    avg_corrs_target.T, colors, model_keys, model_rdm_data["node_indices"]
):
    plt.plot(rdm_times, corrs, label=f"{idx}: {name}", color=color)
plt.xlabel("time (ms)")
plt.ylabel("pearson correlation")

plt.axvline(t_model, color="r")

cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=plt.gca(),
)

cb.ax.minorticks_off()
cb.set_ticks(np.arange(len(model_rdm_data["node_indices"])) + 0.5)  # center each tick
cb.set_ticklabels(model_keys)
cb.ax.tick_params(labelsize=5)

plt.xlim(xlims)
plt.ylim(ylims)

plt.title("Target Signal")

fig_rdm_corrs_target.savefig(
    path.join(subspace_analysis_dir, "rdm_correlations_target.svg")
)

fig_rdm_corrs_sub = plt.figure(figsize=(3.5, 2))
avg_corrs_sub = rdm_corrs_sub.mean(axis=0)

for corrs, color, name, idx in zip(
    avg_corrs_sub.T, colors, model_keys, model_rdm_data["node_indices"]
):
    plt.plot(rdm_times, corrs, label=f"{idx}: {name}", color=color)
plt.xlabel("time (ms)")
plt.ylabel("pearson correlation")

plt.axvline(t_model - d_model, color="r")

cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=plt.gca(),
)

cb.ax.minorticks_off()
cb.set_ticks(np.arange(len(model_rdm_data["node_indices"])) + 0.5)  # center each tick
cb.set_ticklabels(model_keys)
cb.ax.tick_params(labelsize=5)

plt.xlim(xlims)
plt.ylim(ylims)

plt.title("Readout Subspace Signal")

fig_rdm_corrs_sub.savefig(
    path.join(subspace_analysis_dir, "rdm_correlations_readout_subspace.svg")
)


fig_rdm_corrs_pred = plt.figure(figsize=(3.5, 2))
avg_corrs_pred = rdm_corrs_pred.mean(axis=0)

for corrs, color, name, idx in zip(
    avg_corrs_pred.T, colors, model_keys, model_rdm_data["node_indices"]
):
    plt.plot(rdm_times + d_model, corrs, label=f"{idx}: {name}", color=color)
plt.xlabel("time (ms)")
plt.ylabel("pearson correlation")

plt.axvline(t_model, color="r")

cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=plt.gca(),
)

cb.ax.minorticks_off()
cb.set_ticks(np.arange(len(model_rdm_data["node_indices"])) + 0.5)  # center each tick
cb.set_ticklabels(model_keys)
cb.ax.tick_params(labelsize=5)

plt.xlim(xlims)
plt.ylim(ylims)

plt.title("Predicted Target Signal")

fig_rdm_corrs_pred.savefig(
    path.join(subspace_analysis_dir, "rdm_correlations_target_prediction.svg")
)

open(path.join(subspace_analysis_dir, "SUCCESS"), "w").close()  # indicator file
