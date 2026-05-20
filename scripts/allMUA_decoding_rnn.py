import argparse
import os
import pickle as pkl
import subprocess
import sys
from os import path
from pprint import pprint

import numpy as np
import torch
import yaml
from ephyslib import crossvalidation
from ephyslib.decoding.rnn import GRU_Decoder
from ephyslib.decoding.torch_util import LabelEncoder, TrialDataset
from ephyslib.interactive_utils import show_struct
from macaquethings.data_util.load_data import cli_data_parser, load_data
from macaquethings.data_util.process_data import process_data
from torch.utils.data import DataLoader

import wandb


def _git_hash(repo_dir):
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def zscore_over_time(X, stats=None):
    """Z-score [trials, time, channels] data over (trials, time) axes per channel.
    Returns z-scored X and (mean, std) stats. Pass stats to apply train stats to test set.
    """
    if stats is None:
        mean = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1), keepdims=True)
        stats = (mean, std)
    mean, std = stats
    return (X - mean) / (std + 1e-8), stats


_SCRIPT_DIR = path.dirname(path.abspath(__file__))
_EPHYSLIB_DIR = path.join(_SCRIPT_DIR, "..", "packages", "ephyslib")
git_hashes = {
    "amsdrift": _git_hash(_SCRIPT_DIR),
    "ephyslib": _git_hash(_EPHYSLIB_DIR),
}
print("git hashes:", git_hashes)

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
parser.add_argument(
    "--rnn-config",
    type=str,
    default="./config/rnn_decoder_config.yaml",
    help="path to yaml file specifying rnn decoder configuration.",
)

parser.add_argument(
    "--decode-time-first",
    type=int,
    default=50,
    help="first time (in ms) to decode from",
)
parser.add_argument(
    "--decode-time-last", type=int, default=250, help="last time (in ms) to decode from"
)
parser.add_argument(
    "--decode-time-step", type=int, default=10, help="stride between bins (in ms)"
)
parser.add_argument(
    "--temporal-diff",
    action="store_true",
    default=False,
    help="compute temporal differences before passing to RNN (incompatible with standardize_data)",
)
parser.add_argument(
    "--shuffle-time",
    action="store_true",
    default=False,
    help="shuffle time axis of each training trial on every access (val/test never shuffled)",
)
parser.add_argument(
    "--avg-input",
    action="store_true",
    default=False,
    help="average neural data over the decode time interval and repeat for --avg-input-n-steps timesteps (control: no dynamics, matched compute)",
)
parser.add_argument(
    "--avg-input-n-steps",
    type=int,
    default=None,
    help="number of timesteps to repeat the averaged input for (required when --avg-input is set)",
)
args, _ = parser.parse_known_args()
decode_cfg = vars(args)

with open(decode_cfg["rnn_config"]) as f:
    rnn_cfg = yaml.safe_load(f)
print("loaded rnn config:")
pprint(rnn_cfg)

assert not (data_cfg["standardize_data"] and decode_cfg["temporal_diff"]), (
    "standardize_data and temporal_diff cannot both be enabled."
)
assert not (decode_cfg["avg_input"] and decode_cfg["avg_input_n_steps"] is None), (
    "--avg-input-n-steps must be set when --avg-input is enabled."
)

# set the seed and pass it to CV. This allows to get the same splits when
# loading the results
seed = 42
decode_cfg["seed"] = seed
decode_cfg["cv"] = crossvalidation.ImageStratifiedShuffleSplit(5, random_state=seed)

savedir = path.join("results", "rnn_decoding", decode_cfg["run_name"])
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

# select strided time points and reshape for RNN: [trials, time, channels]
target_times = np.arange(
    decode_cfg["decode_time_first"],
    decode_cfg["decode_time_last"],
    decode_cfg["decode_time_step"],
)
time_mask = np.isin(time, target_times)
decode_cfg["decoding_times"] = time[time_mask]
X_seq = X[:, :, time_mask]  # [trials, channels, time]
X_seq = np.transpose(X_seq, (0, 2, 1))  # [trials, time, channels]

if decode_cfg["avg_input"]:
    X_avg = X_seq.mean(axis=1, keepdims=True)  # [trials, 1, channels]
    X_seq = np.repeat(
        X_avg, decode_cfg["avg_input_n_steps"], axis=1
    )  # [trials, n_steps, channels]
    print(
        f"avg_input: averaged over time, repeated for {decode_cfg['avg_input_n_steps']} steps"
    )

print(f"X_seq: {X_seq.shape}  (trials x time x channels)")

roi_str, arr_str, sess_str = data_strs

rnn_config_name = path.splitext(path.basename(decode_cfg["rnn_config"]))[0]
results_fname = f"{data_cfg['monkey']}-labels_{data_cfg['labels']}-sessions_{sess_str}-rois_{roi_str}-arrays_{arr_str}-baseline_{data_cfg['baseline']}-standardize_{data_cfg['standardize_data']}-{data_cfg['neural_data']}-temporal_diff_{decode_cfg['temporal_diff']}-shuffle_time_{decode_cfg['shuffle_time']}-avg_input_{decode_cfg['avg_input']}-avg_input_n_steps_{decode_cfg['avg_input_n_steps']}-config_{rnn_config_name}.pkl"

resultspath = path.join(savedir, results_fname)

unique_y = np.sort(np.unique(y))
shared_label_encoder = LabelEncoder(y)

# patch rnn_cfg with data-derived dimensions
n_trials, n_time, n_channels = X_seq.shape
n_classes = len(unique_y)
rnn_cfg["rnn_params"]["input_size"] = n_channels
rnn_cfg["classifier_params"]["out_features"] = n_classes
print(f"patched rnn_cfg: input_size={n_channels}, out_features={n_classes}")

# check whether this file already exists and exit without error if it does.
if path.exists(resultspath):
    print("RUN ALREADY EXISTS. EXITING.")
    sys.exit(0)

print("will save results to:", resultspath)

# ---------------------------------------- train RNN decoder

# keys passed directly to GRU_Decoder constructor
MODEL_KEYS = (
    "rnn_params",
    "classifier_params",
    "proj_dim",
    "dropout_p",
    "loss_at_last_timestep",
    "device",
    "patience",
)

np.random.seed(seed)
torch.manual_seed(seed)

fold_accuracies = []  # best val acc per fold (scalar)
fold_splits = []  # (train_idx, test_idx) per fold

for cv_split_idx, (train_idx, test_idx) in enumerate(
    decode_cfg["cv"].split(X_seq, y, groups=groups)
):
    print("\n" * 3)
    print(f"--------------- STARTING SPLIT {cv_split_idx + 1}")

    if decode_cfg["temporal_diff"]:
        # temporal difference + per-session z-score: [trials, time-1, channels]
        X_diff = X_seq[:, 1:, :] - X_seq[:, :-1, :]

        sess_idx_train = sess_idx[train_idx]
        sess_idx_test = sess_idx[test_idx]

        X_train = np.full_like(X_diff[train_idx], np.nan)
        X_test = np.full_like(X_diff[test_idx], np.nan)

        for s in np.unique(sess_idx):
            print(f"  session {s}", end=" ")
            sess_mask_train = sess_idx_train == s
            sess_mask_test = sess_idx_test == s

            Xtrain_z, z_stats = zscore_over_time(X_diff[train_idx][sess_mask_train])
            Xtest_z, _ = zscore_over_time(X_diff[test_idx][sess_mask_test], z_stats)

            X_train[sess_mask_train] = Xtrain_z
            X_test[sess_mask_test] = Xtest_z

        print()
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()
    else:
        X_train = X_seq[train_idx]
        X_test = X_seq[test_idx]

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    dset_train = TrialDataset(
        X_train,
        y[train_idx],
        shuffle_time=decode_cfg["shuffle_time"],
        label_encoder=shared_label_encoder,
    )
    dset_test = TrialDataset(
        X_test,
        y[test_idx],
        shuffle_time=False,
        label_encoder=shared_label_encoder,
    )

    wandb.init(
        project="miniTHINGS_rnn_decoding",
        name=f"{results_fname[:-4]}-split_{cv_split_idx}",
        reinit=True,
        config={
            **rnn_cfg,
            **data_cfg,
            "temporal_diff": decode_cfg["temporal_diff"],
            "shuffle_time": decode_cfg["shuffle_time"],
            "decoding_times": decode_cfg["decoding_times"].tolist(),
            "split": cv_split_idx,
        },
    )

    model = GRU_Decoder(**{k: rnn_cfg[k] for k in MODEL_KEYS})
    model.to(rnn_cfg["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=rnn_cfg["lr"], weight_decay=rnn_cfg["weight_decay"]
    )

    dloader_train = DataLoader(
        dset_train, batch_size=rnn_cfg["batch_size"], shuffle=True
    )
    dloader_test = DataLoader(
        dset_test, batch_size=rnn_cfg["batch_size"], shuffle=False
    )

    model.train_model(dloader_train, dloader_test, rnn_cfg["n_epochs"], optimizer)
    wandb.finish()
    fold_accuracies.append(model.best_val.item())
    fold_splits.append((train_idx, test_idx))
    print(f"\n  fold {cv_split_idx + 1} best val acc: {model.best_val:.4f}")

    model_fname = results_fname.replace(".pkl", f"-fold_{cv_split_idx}.pt")
    torch.save(model, path.join(savedir, model_fname))

# save results
# temporal diff reduces time axis by 1: times[1:] correspond to diff time points
saved_times = (
    decode_cfg["decoding_times"][1:]
    if decode_cfg["temporal_diff"]
    else decode_cfg["decoding_times"]
)
results = {
    "times": saved_times,
    "accuracies_per_fold": np.array(fold_accuracies),
    "fold_splits": fold_splits,
    "data_cfg": data_cfg,
    "decode_cfg": decode_cfg,
    "rnn_cfg": rnn_cfg,
    "git_hashes": git_hashes,
}

with open(resultspath, "wb") as f:
    pkl.dump(results, f)
