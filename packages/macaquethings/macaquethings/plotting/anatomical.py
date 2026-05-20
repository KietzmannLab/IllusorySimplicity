import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from skimage.measure import label, regionprops

from macaquethings.plotting.colors import roicolors


def prepare_layout_F(root="."):
    layout_fpath = f"{root}/datasets/F_array.png"
    layout = plt.imread(layout_fpath)
    layout = layout.mean(axis=-1)
    layout = layout < 0.5  # binarize

    # label connected regions
    layout_labels = label(layout)

    # get centroids
    regions = regionprops(layout_labels)
    centroids = np.array([r.centroid for r in regions])

    # label map
    idx_to_array = {
        0: 15,
        1: 16,
        2: 14,
        3: 12,
        4: 6,
        5: 4,
        6: 2,
        7: 9,
        8: 10,
        9: 11,
        10: 13,
        11: 8,
        12: 7,
        13: 5,
        14: 3,
        15: 1,
    }

    idx_to_roi = {
        4: 1,
        5: 1,
        6: 1,
        11: 1,
        12: 1,
        13: 1,
        14: 1,
        15: 1,
        0: 2,
        1: 2,
        2: 2,
        3: 3,
        7: 3,
        8: 3,
        9: 3,
        10: 3,
    }

    rotations = [45, 45, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # convert label maps
    roi_to_idx = {
        roi: [idx for idx, r in idx_to_roi.items() if r == roi]
        for roi in set(idx_to_roi.values())
    }
    array_to_idx = {array: idx for idx, array in idx_to_array.items()}

    # convert labels to color
    layout_colors = np.ones((layout_labels.shape[0], layout_labels.shape[1], 3))
    for idx in idx_to_roi.keys():
        areamask = layout_labels == idx + 1  # 0 is no region
        roi = idx_to_roi[idx]
        layout_colors[areamask] = roicolors[roi]

    return {
        "centroids": centroids,
        "array_to_idx": array_to_idx,
        "roi_to_idx": roi_to_idx,
        "layout_labels": layout_labels,
        "layout_colors": layout_colors,
        "rotations": rotations,
    }


def prepare_layout_N(root="."):
    layout_fpath = f"{root}/datasets/N_array.png"
    layout = plt.imread(layout_fpath)
    layout = layout.mean(axis=-1)
    layout = layout < 0.5  # binarize

    # label connected regions
    layout_labels = label(layout)

    # get centroids
    regions = regionprops(layout_labels)
    centroids = np.array([r.centroid for r in regions])

    # label map
    idx_to_array = {
        0: 11,
        1: 12,
        2: 16,
        3: 15,
        4: 13,
        5: 9,
        6: 10,
        7: 8,
        8: 1,
        9: 7,
        10: 14,
        11: 3,
        12: 5,
        13: 2,
        14: 4,
    }

    idx_to_roi = {
        7: 1,
        8: 1,
        9: 1,
        11: 1,
        12: 1,
        13: 1,
        14: 1,
        0: 2,
        1: 2,
        5: 2,
        6: 2,
        2: 3,
        3: 3,
        4: 3,
        10: 3,
    }

    rotations = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # convert label maps
    roi_to_idx = {
        roi: [idx for idx, r in idx_to_roi.items() if r == roi]
        for roi in set(idx_to_roi.values())
    }
    array_to_idx = {array: idx for idx, array in idx_to_array.items()}

    # convert labels to color
    layout_colors = np.ones((layout_labels.shape[0], layout_labels.shape[1], 3))
    for idx in idx_to_roi.keys():
        areamask = layout_labels == idx + 1  # 0 is no region
        roi = idx_to_roi[idx]
        layout_colors[areamask] = roicolors[roi]

    return {
        "centroids": centroids,
        "array_to_idx": array_to_idx,
        "roi_to_idx": roi_to_idx,
        "layout_labels": layout_labels,
        "layout_colors": layout_colors,
        "rotations": rotations,
    }


def arraypatch(
    ax,
    xcenter,
    ycenter,
    data=np.arange(64).reshape(8, 8),
    cmap="magma",
    vmin=None,
    vmax=None,
    width=150,
    height=150,
    rotation=0,
    nrows=8,
    ncols=8,
    with_border=True,
):
    if with_border:
        linewidth = 0.1
    else:
        linewidth = 0
    # all rectangles together add up to the width and height
    element_h = height / nrows
    element_w = width / ncols
    xs = np.linspace(0, width, ncols)
    ys = np.linspace(0, height, nrows)
    xs = xs - width / 2
    ys = ys - height / 2

    # add offset
    xs += xcenter
    ys += ycenter

    xgrid, ygrid = np.meshgrid(xs, ys)

    # flip grids left to right
    xgrid = np.flip(xgrid, axis=1)
    ygrid = np.flip(ygrid, axis=1)

    # Compute rotation matrix
    theta = np.radians(rotation)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Apply rotation to grid points
    points = np.column_stack((xgrid.flatten(), ygrid.flatten()))
    center = np.array([xcenter, ycenter])
    points = points - center
    points = np.dot(points, R.T)
    points = points + center
    xgrid = points[:, 0].reshape(xgrid.shape)
    ygrid = points[:, 1].reshape(ygrid.shape)

    # Convert data to color values
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    data_normed = (data - vmin) / (vmax - vmin)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(data_normed)
    # Set the colors of the rectangles
    for i in range(nrows):
        for j in range(ncols):
            rect = Rectangle(
                (xgrid[i, j] - element_w / 2, ygrid[i, j] - element_h / 2),
                width=element_w,
                height=element_h,
                angle=rotation,
                facecolor=colors[i, j],
                edgecolor="k",
                linewidth=linewidth,
            )
            ax.add_patch(rect)


def plot_data_on_anatomy(
    monkey,
    data,
    match_cmaps=True,
    show_cbar=True,
    vmin=None,
    vmax=None,
    cmap_name="magma",
    root=".",
    fig_kwargs={},
    ax=None,
    return_cbar=False,
    background_alpha=1,
):
    if monkey == "monkeyF":
        layout_info = prepare_layout_F(root=root)
    elif monkey == "monkeyN":
        layout_info = prepare_layout_N(root=root)
    else:
        raise NotImplementedError("no plotting for this monkey")

    assert data.shape[0] == 1024, "expected 1024 channels of data"
    data_per_array = np.split(data, 16)

    if match_cmaps:
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
    else:
        vmin = None
        vmax = None

    # if no axis is specified, create a new plot and respect fig_kwargs
    if ax == None:
        fig = plt.figure(**fig_kwargs)
        ax = plt.gca()

    background = np.ones((592, 1920, 3))
    ax.imshow(background, alpha=background_alpha)
    # otherwise ax will be the axis passed as a parameter
    ax.set_aspect("equal")

    for i, array in enumerate(np.arange(16) + 1):
        if array not in layout_info["array_to_idx"].keys():
            continue  # monkeyN is missing an array, skip it
        idx = layout_info["array_to_idx"][array]
        arrayloc = layout_info["centroids"][idx]
        rotation = layout_info["rotations"][idx]
        y, x = arrayloc
        arraydata = data_per_array[i].reshape(8, 8)
        arraypatch(
            ax,
            x,
            y,
            data=arraydata,
            rotation=rotation,
            height=140,
            width=140,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap_name,
        )
        ax.axis("off")

    cbar = None
    if show_cbar:
        if not match_cmaps:
            raise ValueError("Cannot show colorbar when match_cmaps=False")
        else:
            cmap = plt.get_cmap(cmap_name)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.3
            )

    if return_cbar and not cbar == None:
        return ax.get_figure(), cbar
    # this will always work, also if we plot into an existing axis
    return ax.get_figure()


def plot_square_data_on_anatomy(
    monkey,
    data,
    match_cmaps=True,
    show_cbar=True,
    vmin=None,
    vmax=None,
    cmap_name="magma",
    scale_factor=5,
):
    if monkey == "monkeyF":
        layout_info = prepare_layout_F()
    elif monkey == "monkeyN":
        layout_info = prepare_layout_N()
    else:
        raise NotImplementedError("no plotting for this monkey")

    data_per_array = np.split(data, 16)
    elements_per_array = len(data_per_array[0])
    nrowcol = int(np.sqrt(elements_per_array))

    if match_cmaps:
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
    else:
        vmin = None
        vmax = None

    fig = plt.figure(figsize=(15, 15))
    background = np.ones((592 * scale_factor, 1920 * scale_factor, 3))
    plt.imshow(background)
    ax = plt.gca()
    plt.gca().set_aspect("equal")
    for i, array in enumerate(np.arange(16) + 1):
        if array not in layout_info["array_to_idx"].keys():
            continue  # monkeyN is missing an array, skip it
        idx = layout_info["array_to_idx"][array]
        arrayloc = layout_info["centroids"][idx]
        rotation = layout_info["rotations"][idx]
        y, x = arrayloc
        arraydata = np.fliplr([i].reshape(nrowcol, nrowcol))
        print("data for array has shape", arraydata.shape)
        arraypatch(
            ax,
            x * scale_factor,
            y * scale_factor,
            data=arraydata,
            rotation=rotation,
            height=140 * scale_factor,
            width=140 * scale_factor,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap_name,
            nrows=nrowcol,
            ncols=nrowcol,
            with_border=False,
        )
        plt.axis("off")

    if show_cbar:
        if not match_cmaps:
            raise ValueError("Cannot show colorbar when match_cmaps=False")
        else:
            cmap = plt.get_cmap(cmap_name)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.3)

    return fig


def animate_square_data(
    monkey,
    data_timeseries,
    times_ms=None,
    outpath="animation.mp4",
    fps=10,
    cmap_name="magma",
    scale_factor=3,
):
    assert data_timeseries.ndim == 2
    T, N = data_timeseries.shape

    # Prepare figure
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()

    # Precompute vmin/vmax for consistent colormap across frames
    vmin = np.nanmin(data_timeseries)
    vmax = np.nanmax(data_timeseries)

    # Background once
    background = np.ones((592 * scale_factor, 1920 * scale_factor, 3))
    ax.imshow(background)
    ax.set_aspect("equal")
    ax.axis("off")
    title = fig.suptitle("", fontsize=20)

    layout_info = prepare_layout_F() if monkey == "monkeyF" else prepare_layout_N()
    centroids = layout_info["centroids"]
    rotations = layout_info["rotations"]
    array_to_idx = layout_info["array_to_idx"]

    nrowcol = int(np.sqrt(N / 16))
    element_patches = []

    # Set up empty rectangles once and store references
    for i, array in enumerate(np.arange(1, 17)):
        if array not in array_to_idx:
            element_patches.append(None)
            continue
        idx = array_to_idx[array]
        y, x = centroids[idx]
        x *= scale_factor
        y *= scale_factor
        rotation = rotations[idx]
        w = h = 140 * scale_factor
        e_w = w / nrowcol
        e_h = h / nrowcol
        xs = np.linspace(0, w, nrowcol, endpoint=False) + e_w / 2 - w / 2 + x
        ys = np.linspace(0, h, nrowcol, endpoint=False) + e_h / 2 - h / 2 + y
        xs, ys = np.meshgrid(xs, ys)
        # xs = np.flip(xs, axis=1)

        theta = np.radians(0)  # no rotation for animation
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        points = np.column_stack((xs.flatten(), ys.flatten()))
        center = np.array([x, y])
        points = np.dot(points - center, R.T) + center
        xs_rot = points[:, 0].reshape(nrowcol, nrowcol)
        ys_rot = points[:, 1].reshape(nrowcol, nrowcol)

        cmap = plt.get_cmap(cmap_name)
        patch_grid = []
        for row in range(nrowcol):
            for col in range(nrowcol):
                rect = Rectangle(
                    (xs_rot[row, col] - e_w / 2, ys_rot[row, col] - e_h / 2),
                    e_w,
                    e_h,
                    linewidth=0,
                    edgecolor="none",
                    facecolor=(1, 1, 1),
                )
                ax.add_patch(rect)
                patch_grid.append(rect)
        element_patches.append(patch_grid)

    def update(frame_idx):
        print(f"Rendering frame {frame_idx + 1}/{T}")
        frame_data = np.split(data_timeseries[frame_idx], 16)
        for i, array_data_flat in enumerate(frame_data):
            if element_patches[i] is None:
                continue
            normed = (array_data_flat - vmin) / (vmax - vmin)
            normed = np.clip(normed, 0, 1)
            colors = cmap(normed)
            for patch, color in zip(element_patches[i], colors):
                patch.set_facecolor(color)

        # Update title
        if times_ms is not None:
            time_label = f"t = {int(times_ms[frame_idx])} ms"
        else:
            time_label = f"frame {frame_idx + 1}/{T}"
        title.set_text(time_label)

        return sum([p for p in element_patches if p is not None], [])

    ani = animation.FuncAnimation(fig, update, frames=T, blit=True)
    ani.save(outpath, fps=fps, dpi=150)
    print(f"Saved animation to {outpath}")


# ------------------------------- testing

if __name__ == "__main__":
    T = 3
    test_timeseries = np.random.randn(T, 16 * 100**2)
    animate_square_data(
        "monkeyN",
        test_timeseries,
        [10, 20, 30],
        outpath="animated_rdm.mp4",
        fps=1,
        scale_factor=3,
    )
