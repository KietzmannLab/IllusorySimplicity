"""Bar plot comparing full vs. constant-input RNN decoding accuracy per array.

Each array gets two bars (full model, constant-input control).
The five cross-validation fold accuracies are overlaid as scatter dots.

Usage
-----
python scripts/plotting/rnn_full_vs_const_barplot.py \
    --run-name monkeyF_rnn_single_array \
    --monkey monkeyF \
    --neural-data mua \
    --out results/rnn_decoding/monkeyF_rnn_single_array/full_vs_const_mua.svg
"""

import argparse
import glob
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
from macaquethings.plotting.default_styles import *

figure_style(font_size=6)

# ── colours ────────────────────────────────────────────────────────────────

COLOR_FULL = "#4C72B0"
COLOR_CONST = "#DD8452"
DOT_COLOR = "white"
DOT_SIZE = 2
BAR_WIDTH = 0.3
BAR_ALPHA = 1
PLOT_WIDTH = THIRD_WIDTH  # from default_styles
PLOT_HEIGHT = QUARTER_WIDTH


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-name",
        default="monkeyF_rnn_single_array",
        help="Subdirectory under results/rnn_decoding/",
    )
    p.add_argument(
        "--monkey",
        default="monkeyF",
        help="Monkey name prefix used in result filenames",
    )
    p.add_argument(
        "--results-root",
        default="results/rnn_decoding",
        help="Root directory for RNN decoding results",
    )
    p.add_argument(
        "--neural-data",
        default=None,
        choices=["lfp", "mua"],
        help="Filter to lfp or mua result files. If omitted, both are included.",
    )
    p.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Upper limit for the y-axis (e.g. 0.8). If omitted, matplotlib auto-scales.",
    )
    p.add_argument(
        "--min-groups",
        type=int,
        default=4,
        help="Minimum number of bar-group slots shown on x-axis (prevents bars becoming too wide). Default: 4.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output path (pdf/png/svg). Defaults to <results_dir>/full_vs_const[_<neural_data>].svg",
    )
    return p.parse_args()


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_array_key(fname):
    """Extract array indices from filename as a sorted tuple.

    Works for both single-array (arrays_3-) and multi-array
    (arrays_1_2_3_..._16-) filenames.
    """
    m = re.search(r"arrays_([\d_]+)-", fname)
    if m:
        return tuple(int(x) for x in m.group(1).split("_"))
    return None


def format_array_label(key):
    """Human-readable x-axis label for an array key tuple."""
    if len(key) == 1:
        return f"array {key[0]}"
    missing = sorted(set(range(key[0], key[-1] + 1)) - set(key))
    if not missing:
        return f"arrays {key[0]}–{key[-1]}"
    return f"arrays {key[0]}–{key[-1]} (excl. {', '.join(str(m) for m in missing)})"


def collect_results(results_dir, monkey, neural_data=None):
    """Return two dicts  {array_idx: accuracies_per_fold}  for full and const models."""
    pattern = os.path.join(results_dir, f"{monkey}-*.pkl")
    files = glob.glob(pattern)

    full = {}
    const = {}

    for fpath in files:
        fname = os.path.basename(fpath)
        arr_idx = extract_array_key(fname)
        if arr_idx is None:
            continue

        # filter by neural data type if requested
        if neural_data is not None and f"-{neural_data}-" not in fname:
            continue

        # distinguish full vs constant-input by avg_input flag in filename
        if "avg_input_False" in fname:
            target = full
        elif "avg_input_True" in fname:
            target = const
        else:
            continue

        data = load_pkl(fpath)
        accs = data["accuracies_per_fold"]  # shape (5,)
        target[arr_idx] = accs

    return full, const


def make_plot(
    full, const, monkey, out_path, neural_data=None, y_max=None, min_groups=4
):
    # arrays present in both conditions
    arrays = sorted(set(full.keys()) & set(const.keys()))
    if not arrays:
        raise RuntimeError(
            "No matching arrays found for both full and constant-input models."
        )

    x = np.arange(len(arrays))
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    for i, arr in enumerate(arrays):
        acc_full = full[arr]
        acc_const = const[arr]

        x_full = x[i] - BAR_WIDTH / 2
        x_const = x[i] + BAR_WIDTH / 2

        # bars
        ax.bar(
            x_full,
            acc_full.mean(),
            width=BAR_WIDTH,
            color=COLOR_FULL,
            alpha=BAR_ALPHA,
            edgecolor="k",
            label="full" if i == 0 else "_nolegend_",
        )
        ax.bar(
            x_const,
            acc_const.mean(),
            width=BAR_WIDTH,
            color=COLOR_CONST,
            alpha=BAR_ALPHA,
            edgecolor="k",
            label="constant input" if i == 0 else "_nolegend_",
        )

        # fold scatter dots
        jitter = (np.random.rand(len(acc_full)) - 0.5) * BAR_WIDTH * 0.4
        ax.scatter(
            x_full + jitter,
            acc_full,
            color="white",
            s=2,
            zorder=5,
            linewidth=0.5,
            edgecolor="k",
        )
        jitter = (np.random.rand(len(acc_const)) - 0.5) * BAR_WIDTH * 0.4
        ax.scatter(
            x_const + jitter,
            acc_const,
            color="white",
            s=2,
            zorder=5,
            linewidth=0.5,
            edgecolor="k",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([format_array_label(a) for a in arrays], rotation=45, ha="right")
    ax.set_ylabel("Decoding accuracy")
    nd_label = f" ({neural_data})" if neural_data else ""
    ax.set_title(f"{monkey}{nd_label} — RNN decoding: full vs. constant input")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # fix x-axis width so bars don't grow too wide when few conditions are shown
    n_slots = max(len(arrays), min_groups)
    center = (len(arrays) - 1) / 2
    ax.set_xlim(center - n_slots / 2, center + n_slots / 2)

    if y_max is not None:
        ax.set_ylim(top=y_max)

    # chance line
    # (we don't know n_classes here without loading data_cfg, so skip unless easy to infer)

    fig.tight_layout()
    fig.savefig(out_path, dpi=500)
    print(f"Saved plot to {out_path}")


def main():
    args = parse_args()
    results_dir = os.path.join(args.results_root, args.run_name)

    out_path = args.out
    neural_data = args.neural_data
    full, const = collect_results(results_dir, args.monkey, neural_data)
    print(f"Found {len(full)} full-model files, {len(const)} constant-input files")

    if not full:
        raise RuntimeError(f"No full-model pkl files found in {results_dir}")
    if not const:
        raise RuntimeError(f"No constant-input pkl files found in {results_dir}")

    if out_path is None:
        suffix = f"_{neural_data}" if neural_data else ""
        out_path = os.path.join(results_dir, f"full_vs_const{suffix}.svg")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    make_plot(
        full, const, args.monkey, out_path, neural_data, args.y_max, args.min_groups
    )


if __name__ == "__main__":
    main()
