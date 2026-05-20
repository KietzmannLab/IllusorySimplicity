import numpy as np
from macaquethings.rdm import get_rdm_design_sort_indices


def plot_model_rdm(ax, root="."):
    sort_idx, _, df = get_rdm_design_sort_indices(
        root, return_values=True, reduce_to_column="category"
    )
    df_model = df.drop(["filenames", "im_id", "cat_id", "category"], axis=1)
    design_mat = df_model.to_numpy()
    all_cats = df_model.columns.values

    # for each category level, create a model rdm
    models = []
    for cat in all_cats:
        x = df_model[cat].values[:, None]
        model = x @ x.T
        models.append(model)

    models = np.array(models)
    all_models = models.sum(axis=0)

    all_models = all_models - np.max(all_models)

    ax.imshow(all_models, cmap="cividis")

    # Move x-axis to top
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Define label tiers
    x_label_tiers = [
        ("natural", "artificial"),
        ("animate", "inanimate"),
    ]

    y_label_tiers = [
        (
            "body_parts",
            "human_face",
            "mammal",
            "non_mammal",
            "fruit",
            "vegetable",
            "other_food",
            "tools",
            "artificial_small_other",
            "vehicles",
            "furniture",
            "outside_large",
        ),
        (
            "human",
            "animal",
            "food",
            "plants",
            "other_natural",
            "artificial_small",
            "artificial_large",
        ),
    ]

    # --- X-axis labels on top ---
    x_y_offset_start = 1.01  # now above the axis (1.0 is top)
    x_y_step = 0.15  # spacing between tiers

    for tier_idx, tier in enumerate(x_label_tiers):
        y_offset = x_y_offset_start + tier_idx * x_y_step
        for label in tier:
            idx = np.where(df_model[label].values > 0)[0][0]
            ax.text(
                idx,
                y_offset,
                label,
                ha="left",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                fontsize=10 + (3 * tier_idx),
                rotation=90,
            )

    # --- Y-axis labels on left ---
    y_x_offset_start = -0.05
    y_x_step = -0.25

    for tier_idx, tier in enumerate(y_label_tiers):
        x_offset = y_x_offset_start + tier_idx * y_x_step
        for label in tier:
            idx = np.where(df_model[label].values > 0)[0][0]
            ax.text(
                x_offset,
                idx,
                label,
                ha="right",
                va="top",
                transform=ax.get_yaxis_transform(),
                fontsize=10 + (3 * tier_idx),
            )

    # Hide default ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)
