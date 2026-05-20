import os
import pickle
from os import path
from pprint import pprint

import macaquethings.plotting.anatomical as anatomy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from macaquethings.data_util.load_data import cli_data_parser, load_data

sns.set_theme(context="notebook", style="whitegrid")

# ------------- SETUP

roidict = {1: "V1", 2: "V4", 3: "IT"}

dprime_cfg = dict(
    peak_intervals={
        1: np.arange(25, 125),  # V1
        2: np.arange(50, 150),  # V4
        3: np.arange(75, 175),  # IT
    },
    baseline_interval=np.arange(-100, 0),
    threshold=1.5,
)

# what data are we loading?
data_cfg = dict(
    monkey="monkeyF",
    labels="category",
    baseline=0,
    session_ids=np.array([0, 1, 2, 3, 4, 5]),
    array_indices=np.arange(16) + 1,
    rois=np.array([1, 2, 3]),
    good_channel_threshold=-1,
    session_ids_for_channel_selection=np.array([0, 1, 2, 3, 4, 5]),
    neural_data="lfp",
    dataset="allMUA",
)

# update with CLI args
data_cfg = cli_data_parser(data_cfg)


print("--------------------------------------------- CONFIG")
pprint(data_cfg)  # shows content of arrays
print("---------------------------------------------")

savedir = path.join("results", "dprime_new")
svgdir = path.join(savedir, "svg")
pngdir = path.join(savedir, "png")

os.makedirs(svgdir, exist_ok=True)
os.makedirs(pngdir, exist_ok=True)

if data_cfg["dataset"] == "allMUA":
    dataset_suffix = ""
elif "lfp_mua" not in data_cfg["dataset"]:
    dataset_suffix = "22k_"
else:
    dataset_suffix = "22klfpmua_"

# ------------------------------ LOAD DATA

time, X, y, groups, im_number, sess_idx, info, data_strs, h5_handle = load_data(
    data_cfg
)
X = np.moveaxis(X, 0, 1)  # channels first for this script
roi_str, arr_str, sess_str = data_strs

unique_labels = np.sort(np.unique(y))
unique_sessions = np.sort(np.unique(sess_idx))
rois = np.array(h5_handle["rois"])
array_ids = np.array(h5_handle["array_ids"])
# dates = h5_handle["date"][:]
# datestrings = [str(d) for d in dates]
# datestrings = [f"{ds[:4]}-{ds[4:6]}-{ds[6:]}" for ds in datestrings]

# ------------------------------

# init a results array
dprimes = np.empty(
    (len(X), len(unique_labels), len(unique_sessions))
)  # shape: channels x classes x sessions

# init an average responses array - used for plotting later
average_responses = np.empty(
    (len(X), len(unique_labels), len(time), len(unique_sessions))
)  # shape: channels x classes x time x sessions

for i_sess, sess in enumerate(unique_sessions):
    # select session data
    sess_mask = sess_idx == sess
    sess_X = X[:, sess_mask]
    sess_y = y[sess_mask]

    # select stimulus data
    for i_stim, stim in enumerate(unique_labels):
        print(f"Session idx: {i_sess}. Stim idx: {i_stim}")
        stim_mask = sess_y == stim
        stim_X = sess_X[:, stim_mask]

        baseline_mean = stim_X[
            ..., np.isin(time, dprime_cfg["baseline_interval"])
        ].mean(axis=(1, 2))  # mean over all trials in baseline period
        stim_X_baseline = (
            stim_X - baseline_mean[:, np.newaxis, np.newaxis]
        )  # expectation over trial-wise baseline averages is 0
        average_responses[:, i_stim, :, i_sess] = stim_X_baseline.mean(
            axis=1
        )  # average response for this stimulus [channels x time]

        Xbaseline = stim_X_baseline[
            ..., np.isin(time, dprime_cfg["baseline_interval"])
        ]  # [channels x stim_trials x baseline_time]
        Xsignal = np.zeros_like(
            Xbaseline
        )  # [channels x stim_trials x baseline_time] - note that all signal windows have the same number of timesteps as the baseline for equal sampling

        # fill Xsignal - signal time windows differ per ROI
        for r in dprime_cfg["peak_intervals"].keys():
            chan_mask = rois == r
            stim_X_baseline_roi = stim_X_baseline[chan_mask]
            time_mask = np.isin(time, dprime_cfg["peak_intervals"][r])
            Xsignal[chan_mask] = stim_X_baseline_roi[..., time_mask]

        # ----- the data going into this computation is:
        # Xbaseline - trials for the current stimulus, time slice from baseline interval [channel x trial x time]
        # Xsignal   - trials for the current stimulus, time slice from signal interval (for each ROI) [channel x trial x time]

        # compute dprime for the stimulus
        # take the absolute value over the response to account for both increase
        # or decrease in spike rate relative to baseline
        # average over response window
        Xb = np.abs(Xbaseline).mean(
            axis=-1
        )  # avg over selected time window - [channels x trials]
        Xs = np.abs(Xsignal).mean(
            axis=-1
        )  # avg over selected time window   - [channels x trials]

        Xb_m = Xb.mean(axis=1)
        Xb_std = Xb.std(
            axis=1
        )  # standard deviation - fluctuation of mean baseline activity over trials

        Xs_m = Xs.mean(axis=1)  # mean over 'signal' time window across trials
        Xs_std = Xs.std(
            axis=1
        )  # standard deviation - fluctuation of mean 'signal' time window activity over trials

        dprime_stim = (Xs_m - Xb_m) / (0.5 * np.sqrt(Xs_std**2 + Xb_std**2))
        dprimes[:, i_stim, i_sess] = dprime_stim

dprime_best_stim = np.max(dprimes, axis=1)


for i in range(dprimes.shape[-1]):
    plt.figure(figsize=(15, 10))
    electrode_idx = np.ones((dprimes.shape[0], dprimes.shape[1]))
    electrode_idx *= (np.arange(dprimes.shape[0]) + 1)[:, np.newaxis]
    plt.scatter(electrode_idx.flatten(), dprimes[..., i].flatten(), marker=".")
    plt.scatter(np.arange(1024) + 1, dprime_best_stim[:, i], marker=".")
    plt.axhline(dprime_cfg["threshold"], color="k")
    plt.xlabel("channel idx")
    plt.ylabel("d-prime")
    plt.title(f" Session {unique_sessions[i]}")
    plt.savefig(
        path.join(
            pngdir,
            f"{data_cfg['monkey']}_{data_cfg['neural_data']}_{dataset_suffix}dprimedist_{i}.png",
        ),
        dpi=300,
    )

    # show the mask for each session
    fig = anatomy.plot_data_on_anatomy(
        data_cfg["monkey"],
        dprime_best_stim[:, i] >= dprime_cfg["threshold"],
        vmin=0,
        vmax=1,
    )
    fig.savefig(
        path.join(
            pngdir,
            f"{data_cfg['monkey']}_{data_cfg['neural_data']}_{dataset_suffix}threshold_dprime_{i}.png",
        ),
        dpi=300,
    )

    # plot average responses for included and excluded channels to check whether exclusion criterion is sensible
    fig, axes = plt.subplots(2, 3, figsize=(30, 15), sharey=True)
    best_stims = dprimes[..., i].argmax(axis=1)
    worst_stims = dprimes[..., i].argmin(axis=1)

    good_channels = dprime_best_stim[:, i] >= 1.5
    bad_channels = ~good_channels

    average_responses_sess = average_responses[..., i]
    strongest_responses = average_responses_sess[
        np.arange(len(average_responses)), best_stims
    ]
    weakest_responses = average_responses_sess[
        np.arange(len(average_responses)), worst_stims
    ]

    print(strongest_responses.shape, weakest_responses.shape)
    # show the good channels
    for r in roidict.keys():
        ax = axes[0, r - 1]
        roi_good_mask = good_channels & (rois == r)
        ax.plot(
            time,
            strongest_responses[roi_good_mask].T,
            color="tab:orange",
            label="strongest response",
        )
        ax.plot(
            time,
            weakest_responses[roi_good_mask].T,
            color="tab:blue",
            label="weakest response",
        )
        ax.set_title(f"{roidict[r]} - good channels: {np.sum(roi_good_mask)}")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("average response (a. u.)")

    # show the bad channels
    for r in roidict.keys():
        ax = axes[1, r - 1]
        roi_bad_mask = bad_channels & (rois == r)
        ax.plot(
            time,
            strongest_responses[roi_bad_mask].T,
            color="tab:orange",
            label="strongest response",
        )
        ax.plot(
            time,
            weakest_responses[roi_bad_mask].T,
            color="tab:blue",
            label="weakest response",
        )
        ax.set_title(f"{roidict[r]} - bad channels: {np.sum(roi_bad_mask)}")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("average response (a. u.)")
    plt.suptitle(f"Session {unique_sessions[i]}")
    plt.savefig(
        path.join(
            pngdir,
            f"{data_cfg['monkey']}_{data_cfg['neural_data']}_{dataset_suffix}average_responses_best_worst_{i}.png",
        ),
        dpi=300,
    )


# save all info
results = {
    "sessions": unique_sessions,
    "labels": unique_labels,
    "dprime": dprimes,
    "data_cfg": data_cfg,
    "dprime_cfg": dprime_cfg,
}

with open(
    path.join(
        savedir,
        f"{data_cfg['monkey']}_{data_cfg['neural_data']}_{dataset_suffix}dprimes.pkl",
    ),
    "wb",
) as f:
    pickle.dump(results, f)
