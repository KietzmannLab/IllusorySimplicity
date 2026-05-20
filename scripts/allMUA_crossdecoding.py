from os import path
import sys
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pickle as pkl
import argparse

from ephyslib import preprocessing
from macaquethings.data_util.load_data import load_data

# defaults
run_name = "monkeyF_decode_category_stratifiedshuffle"
condition = "monkeyF-labels_category-sessions_0-rois_3-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-baseline_0-standardize_1-lfp.pkl"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-name",
    type=str,
    help="run folder containing decoding results.",
    default=run_name,
)
parser.add_argument(
    "--run-file",
    type=str,
    help=".pkl file containing config and results.",
    default=condition,
)
args = parser.parse_args()

# update defaults
run_name = args.run_name
condition = args.run_file

# get decoding results
respath = path.join("results", "decoding", run_name, condition)
decode_results = pkl.load(open(respath, "rb"))

# extract config files
data_cfg = decode_results["data_cfg"]
decode_cfg = decode_results["decode_cfg"]

# crossdecoding is performed for all sessions that were
# used to create the good channel mask
# this ensures matching channel masks for all sessions
crossdecode_sessions = data_cfg["session_ids_for_channel_selection"]
decoding_times = decode_cfg["decoding_times"]

crossdecode_results = np.empty((len(decoding_times), len(crossdecode_sessions)))
crossdecode_results.fill(np.nan)

for i_sess, session in enumerate(crossdecode_sessions):
    print(f"target session id: {session}")
    # load data for the target session
    cross_data_cfg = deepcopy(data_cfg)  # make sure we do not overwrite data cfg
    cross_data_cfg["session_ids"] = [session]

    # load data for cross decoding
    time, X, y, groups, im_number, sess_idx, info, data_strs, h5_handle = load_data(
        cross_data_cfg
    )

    # decode for all time points the original model was trained on
    for i_time, t in enumerate(tqdm(decode_cfg["decoding_times"], file=sys.stdout)):
        time_mask = np.isin(time, t)
        X_time = X[..., time_mask].mean(axis=-1)  # trials x channels

        if decode_cfg["standardize_data"]:
            if i_time == 0:  # only print once
                print("z-scoring per channel and session")

            X_time = preprocessing.zscore_per_sess(X_time, sess_idx)

        # get the model
        model = decode_results["best_models"][i_time]
        acc = model.score(X_time, y)
        crossdecode_results[i_time, i_sess] = acc

# add results to the pkl file and save back to disk
decode_results["crossdecode_accuracies"] = crossdecode_results

with open(respath, "wb") as f:
    pkl.dump(decode_results, f)
