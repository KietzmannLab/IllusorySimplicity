"""Plot decoding accuracy time-courses, one line per array, filtered by ROI.

Reads results produced by scripts/allMUA_decoding.py from results/decoding/{run_name}/.

Usage
-----
python scripts/plotting/decoding_accuracy_curves.py \
    --run-name default \
    --monkey monkeyF \
    --roi 3 \
    --array 9 10 11 12 13 \
    --neural-data mua
"""

import argparse
import glob
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np

SHADING_ALPHA = 0.15
LINE_WIDTH = 1.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-name",
        default="default",
        help="Subdirectory under results/decoding/",
    )
    p.add_argument(
        "--monkey",
        default=None,
        help="Filter by monkey name (e.g. monkeyF). If omitted, all monkeys included.",
    )
    p.add_argument(
        "--roi",
        type=int,
        nargs="+",
        default=None,
        help="ROI index or indices to match (e.g. --roi 3 or --roi 1 2 3). "
             "Matches files whose roi set is exactly this set (all-channel runs for that ROI).",
    )
    p.add_argument(
        "--array",
        type=int,
        nargs="+",
        default=None,
        help="Array indices to include from single-array runs (e.g. --array 9 10 11 12 13). "
             "Matches files where the array field is a single index in this list.",
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
        help="Upper limit for the y-axis.",
    )
    p.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Lower limit for the y-axis.",
    )
    p.add_argument(
        "--shade-folds",
        action="store_true",
        default=False,
        help="Shade ±1 SEM across CV folds around each line.",
    )
    p.add_argument(
        "--results-root",
        default="results/decoding",
        help="Root directory for decoding results.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output path. Defaults to <results_dir>/accuracy_curves[_roi{roi}][_{neural_data}].pdf",
    )
    return p.parse_args()


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_field(fname, field):
    """Extract the value string for a named field (e.g. 'rois' -> '3' or '1_3')."""
    m = re.search(rf"{field}_([\w]+)-", fname)
    return m.group(1) if m else None


def parse_int_list(s):
    """Parse underscore-separated integers, e.g. '1_2_3' -> (1, 2, 3)."""
    return tuple(int(x) for x in s.split("_"))


def collect_results(results_dir, monkey=None, rois=None, arrays=None, neural_data=None):
    """Return list of (label, array_key, data_dict) for files matching the filters.

    Filtering logic (OR when both flags are given):
      --roi:   include files whose roi set matches exactly (all-channel runs for that ROI)
      --array: include single-array files whose array index is in the given list
    If neither flag is given, all files are included.
    """
    pattern = os.path.join(results_dir, "*.pkl")
    files = sorted(glob.glob(pattern))

    results = []
    rois_set = set(rois) if rois is not None else None
    arrays_set = set(arrays) if arrays is not None else None

    for fpath in files:
        fname = os.path.basename(fpath)

        if monkey is not None and not fname.startswith(f"{monkey}-"):
            continue
        if neural_data is not None and f"-{neural_data}" not in fname:
            continue

        roi_str = extract_field(fname, "rois")
        arr_str = extract_field(fname, "arrays")
        if roi_str is None or arr_str is None:
            continue

        file_rois = parse_int_list(roi_str)
        array_key = parse_int_list(arr_str)

        matches_roi = rois_set is not None and set(file_rois) == rois_set
        matches_array = (
            arrays_set is not None
            and len(array_key) == 1
            and array_key[0] in arrays_set
        )

        if rois_set is None and arrays_set is None:
            include = True
        else:
            include = matches_roi or matches_array

        if not include:
            continue

        data = load_pkl(fpath)
        results.append((array_key, data))

    return results


def format_array_label(key):
    """Human-readable legend label for an array key tuple."""
    if len(key) == 1:
        return f"array {key[0]}"
    missing = sorted(set(range(key[0], key[-1] + 1)) - set(key))
    if not missing:
        return f"arrays {key[0]}–{key[-1]}"
    return f"arrays {key[0]}–{key[-1]} (excl. {', '.join(str(m) for m in missing)})"


def make_plot(entries, roi, arrays, monkey, neural_data, out_path, y_min, y_max, shade_folds):
    fig, ax = plt.subplots(figsize=(8, 4))

    # sort entries by array key so legend is ordered
    entries = sorted(entries, key=lambda x: x[0])
    colors = plt.colormaps["tab20"](np.linspace(0, 1, max(len(entries), 1)))

    for (array_key, data), color in zip(entries, colors):
        times = data["times"]
        acc = data["accuracies"]  # (n_times,)
        label = format_array_label(array_key)

        ax.plot(times, acc, color=color, linewidth=LINE_WIDTH, label=label)

        if shade_folds:
            fold_acc = data["accuracies_per_fold"]  # (n_times, n_folds)
            sem = fold_acc.std(axis=1) / np.sqrt(fold_acc.shape[1])
            ax.fill_between(
                times, acc - sem, acc + sem, color=color, alpha=SHADING_ALPHA
            )

    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Decoding accuracy")

    title_parts = []
    if monkey:
        title_parts.append(monkey)
    if roi is not None:
        roi_str = "_".join(str(r) for r in sorted(roi))
        title_parts.append(f"ROI {roi_str}")
    if arrays is not None:
        title_parts.append(f"arrays {', '.join(str(a) for a in sorted(arrays))}")
    if neural_data:
        title_parts.append(neural_data)
    ax.set_title(" — ".join(title_parts) if title_parts else "Decoding accuracy")

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    args = parse_args()
    results_dir = os.path.join(args.results_root, args.run_name)

    entries = collect_results(results_dir, args.monkey, args.roi, args.array, args.neural_data)
    print(f"Found {len(entries)} matching result files")
    if not entries:
        raise RuntimeError(f"No matching pkl files found in {results_dir}")

    if args.out is None:
        suffix = ""
        if args.roi is not None:
            suffix += "_roi" + "_".join(str(r) for r in sorted(args.roi))
        if args.array is not None:
            suffix += "_arrays" + "_".join(str(a) for a in sorted(args.array))
        suffix += f"_{args.neural_data}" if args.neural_data else ""
        out_path = os.path.join(results_dir, f"accuracy_curves{suffix}.pdf")
    else:
        out_path = args.out

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    make_plot(
        entries,
        args.roi,
        args.array,
        args.monkey,
        args.neural_data,
        out_path,
        args.y_min,
        args.y_max,
        args.shade_folds,
    )


if __name__ == "__main__":
    main()
