from os import path

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import rankdata


def get_rdm_design_sort_indices(
    root=".", reduce_to_column="category", return_values=False
):
    stim_info = pd.read_csv(path.join(root, "datasets", "stimulus_information.csv"))
    stim_info_sorted = stim_info.sort_values(
        [
            "animate",
            "body_parts",
            "human",
            "mammal",
            "non_mammal",
            #
            "inanimate",
            "natural",
            "food",
            "fruit",
            "vegetable",
            "other_food",
            #
            "plants",
            "other_natural",
            #
            "artificial",
            "artificial_small",
            "tools",
            "artificial_small_other",
            #
            "artificial_large",
            "furniture",
            "vehicles",
            "outside_large",
            #
            "cat_id",
        ],
        ascending=False,
    )

    # get only the column we are interested in
    stim_info_select = stim_info_sorted[reduce_to_column]
    stim_info_select = stim_info_select.drop_duplicates()
    # get full dataframe for these rows
    indices = stim_info_select.index.values
    stim_info_select_allcols = stim_info.iloc[indices]
    sort_idx = rankdata(stim_info_select.index.values).astype(int) - 1
    if not return_values:
        return sort_idx
    else:
        return sort_idx, stim_info_select.values, stim_info_select_allcols


def create_model(df, column):
    vals = df[column].values
    model = vals[:, None] @ vals[None, :]
    model = (-model) + 1
    # make sure diagonal is zero
    model -= np.diag(np.diag(model))
    return squareform(model)
