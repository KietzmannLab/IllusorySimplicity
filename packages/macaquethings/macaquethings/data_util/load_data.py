import os
import pickle as pkl
import sys
from glob import glob
from os import path

import h5py as h5
import joblib
import numpy as np
import pandas as pd
from ephyslib.connectivity import load_memmap_from_filename
from tqdm import tqdm

defaults = dict(
    monkey="monkeyF",
    labels="category",
    baseline=0,
    session_ids=np.array([0]),
    array_indices=np.arange(16) + 1,
    rois=np.array([3]),
    good_channel_threshold=1.5,
    session_ids_for_channel_selection=np.array([0]),
    neural_data="lfp",
    dataset="allMUA",
)


def cli_data_parser(defaults=defaults):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--monkey", type=str, default=defaults["monkey"], help="monkey identifier"
    )
    parser.add_argument(
        "--labels", type=str, default=defaults["labels"], help="label column to use"
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=defaults["baseline"],
        help="whether to apply baseline correction (type: int. all values > 0 are considered True)",
    )
    parser.add_argument(
        "--session-ids",
        type=int,
        nargs="+",
        default=defaults["session_ids"],
        help="session ids to include",
    )
    parser.add_argument(
        "--array-indices",
        type=int,
        nargs="+",
        default=defaults["array_indices"],
        help="array indices to include",
    )
    parser.add_argument(
        "--rois", type=int, nargs="+", default=defaults["rois"], help="ROIs to include"
    )

    parser.add_argument(
        "--good-channel-threshold",
        type=float,
        default=defaults["good_channel_threshold"],
        help="dprime threshold for channel inclusion",
    )

    parser.add_argument(
        "--session-ids-for-channel-selection",
        type=int,
        nargs="+",
        default=defaults["session_ids_for_channel_selection"],
        help="sessions to consider for channel inclusion. Criterion is applied to the worst session per electrode.",
    )
    parser.add_argument(
        "--neural-data",
        type=str,
        default=defaults["neural_data"],
        help="type of neural data to load. Must be lfp or mua",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=defaults["dataset"],
        help="select a dataset. options are 'allMUA', 'allMUA_22k_train', 'allMUA_22k_test'",
    )

    args, _ = parser.parse_known_args()
    defaults.update(vars(args))
    return defaults


def get_label_map(cfg, label, root="."):
    if cfg["dataset"] == "allMUA":
        stim_info_suffix = ""
    elif (
        cfg["dataset"] == "allMUA_22k_train"
        or cfg["dataset"] == "allMUA_22k_lfp_mua_train"
    ):
        stim_info_suffix = "_22k_train"
    else:
        stim_info_suffix = "_22k_test"
    stim_info = pd.read_csv(
        path.join(root, "datasets", f"stimulus_information{stim_info_suffix}.csv")
    )
    datapath = path.join(root, "datasets", cfg["monkey"], f"{cfg['dataset']}.h5")
    data = h5.File(datapath, "r")
    im_number = np.array(data["im_number"])
    label_map = {
        imnum: cat
        for imnum, cat in zip(
            np.arange(1, len(np.unique(im_number)) + 1), stim_info[label].values
        )
    }

    print(f"using label column {label}")
    print(f"{len(np.unique(list(label_map.values())))} unique labels")

    def map(imnums):
        return np.array([label_map[imnum] for imnum in imnums])

    return map


def load_data(cfg, root="."):
    datapath = path.join(root, "datasets", cfg["monkey"], f"{cfg['dataset']}.h5")
    data = h5.File(datapath, "r")
    if cfg["dataset"] == "allMUA":
        stim_info_suffix = ""
    elif (
        cfg["dataset"] == "allMUA_22k_train"
        or cfg["dataset"] == "allMUA_22k_lfp_mua_train"
    ):
        stim_info_suffix = "_22k_train"
    else:
        stim_info_suffix = "_22k_test"
    stim_info = pd.read_csv(
        path.join(root, "datasets", f"stimulus_information{stim_info_suffix}.csv")
    )

    # strings for easy referencing
    sess_str = "_".join([str(s) for s in cfg["session_ids"]])
    roi_str = "_".join([str(r) for r in cfg["rois"]])
    arr_str = "_".join([str(a) for a in cfg["array_indices"]])

    # get handles for h5 file. Load small datasets.
    neural_data = data[cfg["neural_data"]]  # handle only
    array_ids = np.array(data["array_ids"])
    rois = np.array(data["rois"])
    sess_idx = np.array(data["sess_idx"])
    time = np.array(data["time"]).astype(int)
    im_number = np.array(data["im_number"])

    unique_sess = np.sort(np.unique(sess_idx))

    # some of the indexing below will not work if sess_idx values are incomplete.
    # Especially the dprime scripts index directly without checking for available indices.
    # Note that indices are not session names, but should enumerate the sessions in order so that there are no
    assert len(unique_sess) == unique_sess[-1] + 1, (
        f"session indices should be 0-indexed and not contain any missing indices in the raw datasets. Found indices: {unique_sess}. Last index: {unique_sess[-1]}"
    )

    if cfg["dataset"] == "allMUA":
        im_number_in_class = im_number % 10  # enumerates images in each category
    elif (
        cfg["dataset"] == "allMUA_22k_train"
        or cfg["dataset"] == "allMUA_22k_lfp_mua_train"
    ):
        im_number_in_class = im_number % 12
    else:
        im_number_in_class = np.zeros_like(
            im_number
        )  # one image per category in test set.

    label_map = {
        imnum: cat
        for imnum, cat in zip(
            np.arange(1, len(np.unique(im_number)) + 1), stim_info[cfg["labels"]].values
        )
    }

    print(f"using label column {cfg['labels']}")
    print(f"{len(np.unique(list(label_map.values())))} unique labels")

    # select data
    sess_mask = np.isin(sess_idx, cfg["session_ids"])
    array_mask = np.isin(array_ids, cfg["array_indices"])
    roi_mask = np.isin(rois, cfg["rois"])

    if cfg["good_channel_threshold"] > 0:
        if cfg["dataset"] == "allMUA":
            dset_suffix = ""
        elif "lfp_mua" not in cfg["dataset"]:
            dset_suffix = "22k_"
        else:
            dset_suffix = "22klfpmua_"
        signal_quality_estimates = pkl.load(
            open(
                f"{root}/results/dprime_new/{cfg['monkey']}_{cfg['neural_data']}_{dset_suffix}dprimes.pkl",
                "rb",
            )
        )
        dprimes = signal_quality_estimates["dprime"]
        # get best stimulus per electrode, and select only sessions that were specified
        best_dprimes = dprimes.max(axis=1)[
            ..., cfg["session_ids_for_channel_selection"]
        ]
        worst_over_sessions = np.min(best_dprimes, axis=1)
        good_channels = worst_over_sessions >= cfg["good_channel_threshold"]
    else:
        good_channels = np.ones(1024).astype(bool)

    channel_mask = array_mask & roi_mask & good_channels

    # for allMUA data from monkeyN we have a block of bad trials in session 4.
    # Exclude them from any further procesing - UPDATE: in the new dataset this should no longer be necessary
    good_trials = np.ones(len(im_number)).astype(bool)

    """
    if (cfg["dataset"] == "allMUA") & (cfg["monkey"] == "monkeyN"):
        print("****** allMUA - monkeyN has a block of bad trials in sess_idx == 4.")
        print("... setting up a mask to drop trials.")
        # get a mask for all trials
        idx_bad = np.where(sess_idx == 4)[0][:3000]
        good_trials[idx_bad] = False
    """

    trial_mask = sess_mask & good_trials
    sess_idx_select = sess_idx[trial_mask]

    # save for later reference
    info = dict()
    info["good_channels"] = good_channels
    info["trial_mask"] = trial_mask
    info["array_mask"] = array_mask
    info["roi_mask"] = roi_mask
    info["channel_mask"] = channel_mask

    # now load into memory only the channels we need, then slice trials for included sessions
    X = neural_data[channel_mask][:, trial_mask]
    # ensure what we got is a numpy array. This is mostly so Pyright stops complaining.
    X = np.array(X)
    # move trial axis first
    X = np.moveaxis(a=X, source=0, destination=1)

    if cfg["baseline"]:
        print("Applying baseline correction in [-100,0]")
        baseline_activity = X[..., (time < 0) & (time >= -100)].mean(
            axis=-1, keepdims=True
        )
        X -= baseline_activity

    y = np.array([label_map[imnum] for imnum in im_number])[trial_mask]
    groups = im_number_in_class[trial_mask]
    im_number = im_number[trial_mask]

    print(f"number of groups: {len(np.unique(groups))}")
    print(f"X: {X.shape}, y: {y.shape}")
    assert np.prod(X.shape) > 0, (
        f"at least one axis of X has zero elements, no data loaded (got {X.shape})"
    )

    return (
        time,
        X,
        y,
        groups,
        im_number,
        sess_idx_select,
        info,
        (roi_str, arr_str, sess_str),
        data,
    )


def get_channel_masks(cfg, root="."):
    """
    this function is part of the load_data function above.
    It is intended to set up the data masks without subsequent loading of the data.
    """
    datapath = path.join(root, "datasets", cfg["monkey"], f"{cfg['dataset']}.h5")
    data = h5.File(datapath, "r")
    if cfg["dataset"] == "allMUA":
        stim_info_suffix = ""
    elif (
        cfg["dataset"] == "allMUA_22k_train"
        or cfg["dataset"] == "allMUA_22k_lfp_mua_train"
    ):
        stim_info_suffix = "_22k_train"
    else:
        stim_info_suffix = "_22k_test"
    stim_info = pd.read_csv(
        path.join(root, "datasets", f"stimulus_information{stim_info_suffix}.csv")
    )

    # strings for easy referencing
    sess_str = "_".join([str(s) for s in cfg["session_ids"]])
    roi_str = "_".join([str(r) for r in cfg["rois"]])
    arr_str = "_".join([str(a) for a in cfg["array_indices"]])

    # get handles for h5 file. Load small datasets.
    neural_data = data[cfg["neural_data"]]  # handle only
    array_ids = np.array(data["array_ids"])
    rois = np.array(data["rois"])
    sess_idx = np.array(data["sess_idx"])
    time = np.array(data["time"]).astype(int)
    im_number = np.array(data["im_number"])

    unique_sess = np.sort(np.unique(sess_idx))

    # some of the indexing below will not work if sess_idx values are incomplete.
    # Especially the dprime scripts index directly without checking for available indices.
    # Note that indices are not session names, but should enumerate the sessions in order so that there are no
    # missing indices.
    assert len(unique_sess) == unique_sess[-1] + 1, (
        f"session indices should be 0-indexed and not contain any missing indices in the raw datasets. Found indices: {unique_sess}. Last index: {unique_sess[-1]}"
    )

    if cfg["dataset"] == "allMUA":
        im_number_in_class = im_number % 10  # enumerates images in each category
    elif (
        cfg["dataset"] == "allMUA_22k_train"
        or cfg["dataset"] == "allMUA_22k_lfp_mua_train"
    ):
        im_number_in_class = im_number % 12
    else:
        im_number_in_class = np.zeros_like(
            im_number
        )  # one image per category in test set.

    label_map = {
        imnum: cat
        for imnum, cat in zip(
            np.arange(1, len(np.unique(im_number)) + 1), stim_info[cfg["labels"]].values
        )
    }

    print(f"using label column {cfg['labels']}")
    print(f"{len(np.unique(list(label_map.values())))} unique labels")

    # select data
    sess_mask = np.isin(sess_idx, cfg["session_ids"])
    array_mask = np.isin(array_ids, cfg["array_indices"])
    roi_mask = np.isin(rois, cfg["rois"])

    if cfg["good_channel_threshold"] > 0:
        if cfg["dataset"] == "allMUA":
            dset_suffix = ""
        elif "lfp_mua" not in cfg["dataset"]:
            dset_suffix = "22k_"
        else:
            dset_suffix = "22klfpmua_"
        signal_quality_estimates = pkl.load(
            open(
                f"{root}/results/dprime_new/{cfg['monkey']}_{cfg['neural_data']}_{dset_suffix}dprimes.pkl",
                "rb",
            )
        )
        dprimes = signal_quality_estimates["dprime"]
        # get best stimulus per electrode, and select only sessions that were specified
        best_dprimes = dprimes.max(axis=1)[
            ..., cfg["session_ids_for_channel_selection"]
        ]
        worst_over_sessions = np.min(best_dprimes, axis=1)
        good_channels = worst_over_sessions >= cfg["good_channel_threshold"]
    else:
        good_channels = np.ones(1024).astype(bool)

    channel_mask = array_mask & roi_mask & good_channels
    trial_mask = sess_mask

    return {
        "channel_mask": channel_mask,
        "array_mask": array_mask,
        "roi_mask": roi_mask,
        "good_channels": good_channels,
        "trial_mask": trial_mask,
        "rois": rois,
        "array_ids": array_ids,
    }


def load_inter_area_fit_chunks(rundir, with_tqdm=True):
    """
    Load all performance results, parameters and config files for results of an inter_area model fit.
    When scheduling analyses with SLURM, the grid of time - delay pairs is sliced into chunks, each
    of which is submitted as one task in an array job.
    This function reads partial results for all chunks and returns them together.
    """
    chunks = [f for f in os.listdir(rundir) if f.startswith("chunk")]
    params, inter_area_results, recurrent_results = [], [], []
    if with_tqdm:
        chunk_iter = tqdm(chunks, file=sys.stdout)
    else:
        chunk_iter = chunks
    for i, chunk in enumerate(chunk_iter):
        chunkdir = path.join(rundir, chunk)
        if i == 0:
            cfg = pkl.load(open(path.join(chunkdir, "_cfg.pkl"), "rb"))
        if path.isfile(
            path.join(chunkdir, "success")
        ):  # only load chunks that were successfully completed
            recurrent_file = [
                x
                for x in os.listdir(chunkdir)
                if x.endswith(".npy") and "recurrent" in x
            ][0]
            inter_area_file = [
                x
                for x in os.listdir(chunkdir)
                if x.endswith(".npy") and "inter_area" in x
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

    return inter_area_results, recurrent_results, params, cfg


def load_models_for_time_and_delay(t, d, rundir):
    """
    Given a results folder from an inter-area fit, return the stored models for a time, delay pair.
    """
    modelfile = glob(f"**/time_area2_{str(t)}_delay_{d}.joblib", root_dir=rundir)
    assert len(modelfile) == 1
    modelfile = modelfile[0]

    model_results = joblib.load(path.join(rundir, modelfile))
    models = model_results["models"]

    recurrent_models = models["recurrent"]
    full_models = models["inter_area"]

    full_Ws = np.array([m.coef_ for m in full_models])
    recurrent_Ws = np.array([m.coef_ for m in recurrent_models])

    print("full model:", full_Ws.shape, "recurrent model:", recurrent_Ws.shape)

    return full_Ws, recurrent_Ws, recurrent_models, full_models, model_results


if __name__ == "__main__":
    cfg = cli_data_parser()
    print(cfg)
    print("--------------------------")
    time, X, y, groups, im_number, sess_idx, info, data_strs, data = load_data(cfg)
