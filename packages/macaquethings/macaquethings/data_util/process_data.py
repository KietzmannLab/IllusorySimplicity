import numpy as np
from ephyslib import preprocessing
from ephyslib.stimulus_response import compute_stimulus_responses
from scipy.ndimage import convolve1d
from scipy.stats import zscore


def process_data(X, sess_idx, im_number, groups, cfg):
    if cfg["standardize_data"]:
        print("Z-scoring each bin and session.")
        X = preprocessing.zscore_per_sess(X, sess_idx, copy=False)

    if cfg["trial_averaged"]:
        print("converting dataset to trial-averages per stimulus.")
        print(
            "WARNING: This assumes that all repeats of the same stimulus belong to the same group!"
        )
        print(
            "         Be careful when using this flag with a crossvalidator relying on groups"
        )
        unique_ims = np.unique(im_number)
        X, groups = compute_stimulus_responses(X, im_number, unique_ims, groups)

    if cfg["avg_time"] is not None:
        print(f"averaging data in time with bin size {cfg['avg_time']}")
        print(f"data have shapes {X.shape}.")
        print("applying convolution to axis -1")
        kernel = np.ones((cfg["avg_time"])) / cfg["avg_time"]
        X = convolve1d(X, kernel, axis=-1, mode="constant")

    # ensure that all channels still have equal variance and are mean centered
    if cfg["standardize_data"]:
        X = zscore(X)

    # make sure data is contiguous in memory
    X = np.ascontiguousarray(X)

    return X, groups


def process_data_for_interarea_fit(
    Xsource, Xtarget, sess_idx, im_number, groups, inter_area_cfg
):
    """
    Process source and target datasets using the same pipeline implemented in
    `process_data`. Process the source first so that if `trial_averaged=True`
    the `groups` returned from source-processing are used for target-processing,
    ensuring stimulus-group alignment between source and target.
    Returns (Xsource_processed, Xtarget_processed, groups).
    """

    # if trial averaging is enabled, groups will be updated to match the shape of the new dataset
    Xs, groups_new = process_data(Xsource, sess_idx, im_number, groups, inter_area_cfg)
    Xt, groups_new = process_data(Xtarget, sess_idx, im_number, groups, inter_area_cfg)

    return Xs, Xt, groups_new


# legacy compatibility
def process_target_data_for_encoding_model(
    Xtarget, sess_idx, im_number, groups, encoding_cfg
):
    return process_data(Xtarget, sess_idx, im_number, groups, encoding_cfg)
