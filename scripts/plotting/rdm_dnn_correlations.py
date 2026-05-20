#!/usr/bin/env python

import argparse
import glob
import os
import pickle
from os import path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from macaquethings.plotting.default_styles import *
from macaquethings.rdm_util import get_rdm_design_sort_indices
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.spatial.distance import cdist, squareform
from scipy.stats import rankdata

figure_style(font_size=6)  # set consistent plotting defaults for all figs


# ------------------------------------------ plot opt

PANEL_SIZE = THIRD_PANEL_SIZE
WITHCOLORBAR = False

XMIN = 0  # ms
XMAX = 250  # ms

YMIN = -0.05
YMAX = 0.4


# ------------------------------------------ DNN config

dnn_name = "resnet18"
model_rdm_name = "rdms-resnet18-metric_cosine-normalization_None-pca_1000.pkl"
model_rdm_folder = path.join("datasets", "TIMM", dnn_name)

# ------------------------------------------


def correlate_rdm_movie_with_models(rdm_timecourse, target_rdms, model_keys):
    print(f"  time course shape: {rdm_timecourse.shape}")
    # extract model rdms from dict
    models = np.array([target_rdms[key] for key in model_keys])
    print(f"  models shape: {models.shape}")
    return 1 - cdist(rdm_timecourse, models, metric="correlation")


def plot_model_rdm_grid(model_rdm_dict, model_keys, sort_idx, savedir):
    print(f"\nPlotting model RDMs grid ({len(model_keys)} layers)...")

    fig, axs = plt.subplots(5, 5, figsize=(FULL_WIDTH, FULL_WIDTH))
    axs_flt = axs.flatten()
    for i, key in enumerate(model_keys):
        model = model_rdm_dict[key]
        model = rankdata(model)
        model = squareform(model)
        model = model[sort_idx][:, sort_idx]
        ax = axs_flt[i]
        ax.imshow(model, rasterized=True)
        ax.set_title(key)
    for ax in axs_flt:
        ax.axis("off")
    savepath = path.join(savedir, "dnn_model_rdms.svg")
    plt.savefig(savepath, dpi=500)
    print(f"  Saved: {savepath}")
    plt.close(fig)


def process_rdm_file(
    rdmpath, model_rdm_data, model_keys, model_rdm_dict, dnn_corr_savedir
):
    rdm_file = path.basename(rdmpath)
    print(f"\n{'=' * 60}")
    print(f"Processing: {rdm_file}")
    print(f"Loading neural RDMs from: {rdmpath}")
    with open(rdmpath, "rb") as f:
        rdm_data = pickle.load(f)

    time = rdm_data["time"]
    rdms = rdm_data["rdms"]
    print(f"  Neural RDMs shape: {rdms.shape}  (timepoints x rdm_entries)")
    print(f"  Time range: {time[0]:.0f} – {time[-1]:.0f} ms  ({len(time)} bins)")
    print(f"  Labels: {rdm_data['data_cfg']['labels']}")

    # correlate
    print("\nCorrelating neural RDM timecourse with DNN layer RDMs...")
    rdm_corrs = correlate_rdm_movie_with_models(rdms, model_rdm_dict, model_keys)
    print(f"  Correlations shape: {rdm_corrs.shape}  (timepoints x layers)")

    colors = np.array(
        sns.color_palette("rocket", len(model_keys) + 1)[1:]
    )  # exclude first color, since it is too faint
    # sort colors by layer position
    layer_idx = np.argsort(model_rdm_data["node_indices"])
    colors = colors[layer_idx]

    # cmap for colorbar
    cmap = ListedColormap(colors)
    bounds = np.arange(len(colors) + 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig_rdm_corrs = plt.figure(figsize=(THIRD_WIDTH, THIRD_WIDTH / 2))

    for corrs, color, name, idx in zip(
        rdm_corrs.T, colors, model_keys, model_rdm_data["node_indices"]
    ):
        plt.plot(time, corrs, label=f"{idx}: {name}", color=color)
    plt.xlabel("time (ms)")
    plt.ylabel("pearson correlation")
    plt.xlim((XMIN, XMAX))
    plt.ylim((YMIN, YMAX))
    sns.despine(fig=fig_rdm_corrs)

    if WITHCOLORBAR:
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=plt.gca(),
        )
        cb.ax.minorticks_off()
        cb.set_ticks(
            np.arange(len(model_rdm_data["node_indices"])) + 0.5
        )  # center each tick
        cb.set_ticklabels(model_keys)
        cb.ax.tick_params(labelsize=5)

    savepath = path.join(dnn_corr_savedir, f"{rdm_file[:-4]}_dnn_corrs.svg")
    plt.savefig(savepath)
    print(f"  Saved: {savepath}")
    plt.close(fig_rdm_corrs)

    # if WITHCOLORBAR is set to False, plot colorbar in separate figure
    if not WITHCOLORBAR:
        fig, ax = plt.subplots(1, 1, figsize=PANEL_SIZE)
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8
        )
        cb.ax.minorticks_off()
        cb.set_ticks(
            np.arange(len(model_rdm_data["node_indices"])) + 0.5
        )  # center each tick
        cb.set_ticklabels(model_keys)
        cb.ax.tick_params(labelsize=5)
        ax.axis("off")
        savepath = path.join(dnn_corr_savedir, f"{rdm_file[:-4]}_dnn_corrs_cbar.svg")
        plt.savefig(savepath)
        print(f"  Saved: {savepath}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Correlate neural RDM timecourses with DNN layer RDMs."
    )
    parser.add_argument(
        "--input",
        help="Path to a single neural RDM .pkl file, or a folder containing multiple .pkl files.",
    )
    args = parser.parse_args()

    # resolve input files
    if path.isdir(args.input):
        rdm_files = sorted(glob.glob(path.join(args.input, "*.pkl")))
        rdm_folder_name = path.basename(args.input.rstrip("/\\"))
        print(f"Found {len(rdm_files)} PKL file(s) in: {args.input}")
        if not rdm_files:
            print("No .pkl files found — exiting.")
            return
    else:
        rdm_files = [args.input]
        rdm_folder_name = path.basename(path.dirname(args.input))

    dnn_corr_savedir = path.join("results", "rdm", rdm_folder_name, "svg", dnn_name)
    os.makedirs(dnn_corr_savedir, exist_ok=True)

    # load model RDMs once
    model_rdm_path = path.join(model_rdm_folder, model_rdm_name)
    print(f"\nLoading model RDMs from: {model_rdm_path}")
    with open(model_rdm_path, "rb") as f:
        model_rdm_data = pickle.load(f)

    model_keys = model_rdm_data["selected_nodes"]
    model_rdm_dict = model_rdm_data["rdms"]
    print(f"  DNN: {dnn_name}  |  {len(model_keys)} layers selected: {model_keys}")

    # load first file to get sort_idx for the model RDM grid (labels should be shared)
    with open(rdm_files[0], "rb") as f:
        first_rdm_data = pickle.load(f)
    sort_idx = get_rdm_design_sort_indices(
        ".", return_values=False, reduce_to_column=first_rdm_data["data_cfg"]["labels"]
    )

    # plot model RDM grid once per run
    plot_model_rdm_grid(model_rdm_dict, model_keys, sort_idx, dnn_corr_savedir)

    # process each neural RDM file
    for i, rdmpath in enumerate(rdm_files):
        print(f"\n[{i + 1}/{len(rdm_files)}]", end="")
        process_rdm_file(
            rdmpath, model_rdm_data, model_keys, model_rdm_dict, dnn_corr_savedir
        )

    print(f"\nDone. Processed {len(rdm_files)} file(s).")


if __name__ == "__main__":
    main()
