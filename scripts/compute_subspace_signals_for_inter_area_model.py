"""
Given a set of inter-area models and a time and delay, project the neural time courses
onto two spaces:

    1. The subspace that is read out from a source region
    2. The prediction for the target region, projected in the same dimensionality

The two datasets differ only by reweighting: The first is a selection of directions in
the source space. The second is the same data after allowing for 'stretching' by the weight matrix.

The script stores the projected data in a subfolder of the inter-area run from which
the models (and subspaces) originate.

Additionally, RDMs can be computed from the projected signal.

"""

import os
import pickle
from os import path

import numpy as np
from ephyslib.connectivity import select_data_for_inter_area_fit
from ephyslib.rdm import compute_robust_rdm
from ephyslib.subspace import find_subspace
from macaquethings.data_util.load_data import (
    load_data,
    load_inter_area_fit_chunks,
    load_models_for_time_and_delay,
)
from macaquethings.data_util.process_data import process_data_for_interarea_fit
from scipy.linalg import orth
from tqdm import tqdm

STORE_SIGNALS = True

# run from which to get inter-area models
inter_area_dir = path.join("results", "inter_area")
run_name = "monkeyN_v4_target_array_14_stride_2_ridgecv_lfp_hugegrid_avg10ms_allsess"

# time and delay of the model to select
t = 134
d = 28

# subspace settings
mode = "max"
threshold = 0.99  # not used if mode max

subsample_time = 5

# RDM parameters
navg = 10
nsamples = 1_000
metric = "correlation"

# ----------------------------------------------- load inter-area results

rundir = path.join(inter_area_dir, run_name)
inter_area_results, recurrent_results, params, cfg = load_inter_area_fit_chunks(rundir)

print("loaded results:")
print(inter_area_results.shape, recurrent_results.shape, params.shape)

# ----------------------------------------------- get config

# we want category and image information to be available.
# Force y to be category, im_number is always available from
# a separate variable

overrides = {
    "labels": "category",
}
data_cfg_source = cfg["data_cfg_source"]
data_cfg_target = cfg["data_cfg_target"]

# update data selection
data_cfg_source.update(overrides)
data_cfg_target.update(overrides)

if d >= 0:
    data_cfg_predictors = cfg["data_cfg_source"]
else:
    data_cfg_predictors = cfg["data_cfg_target"]  # reverse for feedback

# ----------------------------------------------- load models for the selected time and delay

full_Ws, recurrent_Ws, recurrent_models, full_models, model_results = (
    load_models_for_time_and_delay(t, d, rundir)
)

# ----------------------------------------------- load data

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
    Xsource_t, Xtarget_t, t - d, t, time
)

time = h5_handle["time"][:]

print("time course data:")
print("source:", Xsource_t.shape)
print("target:", Xtarget_t.shape)
print()
print("selected time point data")
print("source:", Xsource.shape)
print("target:", Xtarget.shape)
print("recurrent", Xrecur.shape)


def predict_per_fold(
    model_results, Xsource, full_Ws, P=None, return_source_subspace=False
):
    """
    predict data with the inter-area weight matrix of an inter-area model,
    respecting the folds that were used during fitting.
    Each trial is predicted using the weight matrix from the model where
    this trial was in the test split
    """
    W_is = []
    folds = model_results["folds"]["test_folds"]
    Xp_i = np.empty((Xsource.shape[0], full_Ws.shape[1]))
    Xp_i.fill(np.nan)

    if return_source_subspace:
        Xsub = np.empty(Xsource.shape)
        Xsub.fill(np.nan)

    for i in range(len(full_models)):
        W_i_fold = full_Ws[i][:, : Xsource.shape[1]]
        test_idx = folds[i]

        if P is not None:
            W_i_fold = (W_i_fold.T @ P).T  # projection onto subspace

        Xp_i[test_idx] = Xsource[test_idx] @ W_i_fold.T
        W_is.append(W_i_fold)

        if return_source_subspace:
            W_basis = orth(W_i_fold.T)
            P_source = W_basis @ W_basis.T
            Xsub[test_idx] = Xsource[test_idx] @ P_source

    assert not np.isnan(Xp_i).any()

    if return_source_subspace:
        assert not np.isnan(Xsub).any()
        print(np.linalg.matrix_rank(Xsub))
        return Xp_i, Xsub, W_is

    else:
        return Xp_i, np.array(W_is)


Xp_i, W_is = predict_per_fold(model_results, Xsource, full_Ws)

print("predicted data (before rank reduction):", Xp_i.shape)
print("inter-area weight matrices (before rank reduction):", W_is.shape)

# ----------------------------------------------- get ideal subspace size

P_inter, n_dims_inter, perfs_inter, svd_inter = find_subspace(
    Xp_i, Xtarget, threshold=threshold, mode=mode
)
print(
    f"found best subsapce in mode {mode} (with threshold {threshold}): {n_dims_inter} dimensions."
)

# ----------------------------------------------- project full time course with reduced rank


def project_timecourse(X, model_results, Ws, P=None):
    # init results array
    Xp = np.empty((X.shape[0], Ws.shape[1], X.shape[2]))  # target space
    Xsub = np.empty(X.shape)  # source space
    Xp.fill(np.nan)

    for i in tqdm(range(X.shape[2])):
        Xp_t, Xsub_t, _ = predict_per_fold(
            model_results, X[..., i], Ws, P=P, return_source_subspace=True
        )
        Xp[..., i] = Xp_t
        Xsub[..., i] = Xsub_t

    # make sure the array was filled completely
    assert not np.isnan(Xp).any()
    assert not np.isnan(Xsub).any()
    return Xp, Xsub


Xpred_t, Xsub_t = project_timecourse(Xsource_t, model_results, full_Ws, P_inter)

# ----------------------------------------------- compute RDMs

# subsample if set
if subsample_time:
    time = time[::subsample_time]
    Xpred_t = Xpred_t[..., ::subsample_time]
    Xsub_t = Xsub_t[..., ::subsample_time]

# compute rdms
rdms_pred = []
rdms_sub = []

unique_stim = np.sort(np.unique(im_number))
for i in tqdm(range(Xpred_t.shape[-1])):
    Xp_t = Xpred_t[..., i]
    Xs_t = Xsub_t[..., i]
    rdm_p = compute_robust_rdm(
        Xp_t, im_number, unique_stim, navg, nsamples, metric, parallel=True
    )
    rdm_s = compute_robust_rdm(
        Xs_t, im_number, unique_stim, navg, nsamples, metric, parallel=True
    )
    rdms_pred.append(rdm_p)
    rdms_sub.append(rdm_s)

rdms_pred = np.array(rdms_pred)
rdms_sub = np.array(rdms_sub)

# save
savedir = path.join(rundir, f"subspace_data-t_{t}-d_{d}")
os.makedirs(savedir, exist_ok=True)

if STORE_SIGNALS:
    data = {
        "time": time,
        "Xpred": Xpred_t,
        "Xsub": Xsub_t,
        "category": y,
        "im_number": im_number,
    }

    with open(path.join(savedir, "neural_data.pkl"), "wb") as f:
        pickle.dump(data, f)

rdm_data = {
    "stimuli": unique_stim,
    "time": time,
    "cfg": {"navg": navg, "nsamples": nsamples, "metric": metric},
    "rdm_pred": rdms_pred,
    "rdm_sub": rdms_sub,
}

with open(path.join(savedir, "rdms.pkl"), "wb") as f:
    pickle.dump(rdm_data, f)
