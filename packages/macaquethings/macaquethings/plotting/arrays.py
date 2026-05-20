import matplotlib.pyplot as plt
import numpy as np

roidict = {1: "V1", 2: "V4", 3: "IT"}


def plot_timeseries_per_array(time, data, array_ids, rois, sharey=False, sharex=False):
    """
    utility to plot THINGS timeseries data in separate subplots for each array

    time        - time vector for the x axis
    data        - data in the format channel x time
    array_ids   - array indices for each vector in data
    rois        - roi assignment for the array. determines color
    """
    fig, axes = plt.subplots(4, 4, figsize=(45, 30), sharex=sharex, sharey=sharey)
    flataxes = axes.flatten()

    roicolors = {
        1: (233 / 255, 51 / 255, 101 / 255),
        2: (102 / 255, 152 / 255, 246 / 255),
        3: (94 / 255, 203 / 255, 103 / 255),
    }

    for i, ax in enumerate(flataxes):
        array_data = data[array_ids == i + 1]  # account for matlab 1 indexing
        roi = np.unique(rois[array_ids == i + 1])
        assert (
            len(roi) <= 1
        ), f"one array cannot be in multiple rois. something went wrong. array: {i+1}. rois: {roi}"
        if len(roi) == 0:
            ax.axis("off")
            continue  # do not crash if there is no data for one array
        roi = roi[0]
        ax.plot(time, array_data.T, color=roicolors[roi])
        ax.set_title(f"array id: {i+1}. roi: {roidict[roi]}")
        ax.set_xlabel("time (ms)")
    return fig, axes
