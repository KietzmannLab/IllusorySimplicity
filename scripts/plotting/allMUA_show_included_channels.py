import matplotlib.pyplot as plt
import numpy as np
from macaquethings.data_util.load_data import get_channel_masks
from macaquethings.plotting.anatomical import plot_data_on_anatomy
from macaquethings.plotting.default_styles import *

figure_style()  # set consistent plotting defaults for all figs


cfg_F_lfp = dict(
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

cfg_F_mua = dict(
    monkey="monkeyF",
    labels="category",
    baseline=0,
    session_ids=np.array([0, 1, 2, 3, 4, 5]),
    array_indices=np.arange(16) + 1,
    rois=np.array([1, 2, 3]),
    good_channel_threshold=1.5,
    session_ids_for_channel_selection=np.array([0, 1, 2, 3, 4, 5]),
    neural_data="mua",
    dataset="allMUA",
)


cfg_N_lfp = dict(
    monkey="monkeyN",
    labels="category",
    baseline=0,
    session_ids=np.array([0, 3, 4, 5]),
    array_indices=np.arange(16) + 1,
    rois=np.array([1, 2, 3]),
    good_channel_threshold=1.5,
    session_ids_for_channel_selection=np.array([0, 3, 4, 5]),
    neural_data="lfp",
    dataset="allMUA",
)

cfg_N_mua = dict(
    monkey="monkeyN",
    labels="category",
    baseline=0,
    session_ids=np.array([0, 3, 4, 5]),
    array_indices=np.arange(16) + 1,
    rois=np.array([1, 2, 3]),
    good_channel_threshold=1.5,
    session_ids_for_channel_selection=np.array([0, 3, 4, 5]),
    neural_data="mua",
    dataset="allMUA",
)


data_cfgs = {
    "monkeyF_lfp": cfg_F_lfp,
    "monkeyF_mua": cfg_F_mua,
    "monkeyN_lfp": cfg_N_lfp,
    "monkeyN_mua": cfg_N_mua,
}


for fname, data_cfg in data_cfgs.items():
    print(f"plotting for: {fname}")
    masks = get_channel_masks(data_cfg, root=".")
    good_channels = masks["good_channels"]
    array_indices = masks["array_ids"]
    if data_cfg["monkey"] == "monkeyN":
        good_channels[array_indices == 6] = False
        print("loaded masks for monkeyN: excluded all channels in array 6")
        fig = plot_data_on_anatomy(
            data_cfg["monkey"], good_channels, vmin=0, vmax=1, root=".", show_cbar=False
        )
        sess_str = "_".join(data_cfg["session_ids_for_channel_selection"])
        plt.savefig(
            f"included_channel_map_{fname}_threshold_{data_cfg['good_channel_threshold']}_{sess_str}.svg"
        )
