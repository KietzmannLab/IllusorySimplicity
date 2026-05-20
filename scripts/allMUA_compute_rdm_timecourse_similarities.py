# --------------------- IMPORTS
import os
from os import path
import pickle
from glob import glob
import numpy as np
import cupy as cp 
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --------------------- LOAD DATA

rdm_folder = path.join('results', 'rdm', 'monkeyF_lfp_filenames_arrays_and_rois_10ms')
rdm_pkls = [f for f in os.listdir(rdm_folder) if f.endswith('.pkl')]

# rdm_pkls = [f for f in rdm_pkls if 'arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16' in f]

rdm_data_per_run = []

for rdmfile in rdm_pkls:
    fpath = path.join(rdm_folder, rdmfile)
    with open(fpath, 'rb') as f:
        rdm_data = pickle.load(f)
        rdm_time = rdm_data['rdm_cfg']['bin_centers'][:,0]
        rdms = rdm_data['rdms']
        rdm_data_per_run.append({
            'time': rdm_time,
            'rdm': rdms
        })

all_rdms = np.concatenate([data['rdm'] for data in rdm_data_per_run])
all_times = np.concatenate([data['time'] for data in rdm_data_per_run])
nper_run = [len(data['time']) for data in rdm_data_per_run]

rois = [fname.split('-')[3] for fname in rdm_pkls]
arrays = [fname.split('-')[4] for fname in rdm_pkls]

rois_repeat = []
arrays_repeat = []

for r, a, n in zip(rois, arrays, nper_run):
    rois_repeat += [r] * n
    arrays_repeat += [a] * n

all_labels = []
for r,a in zip(rois_repeat, arrays_repeat):
    if r == 'rois_1_2_3':
        all_labels.append(a)
    else:
        all_labels.append(r)

# exclude baseline
all_rdms = all_rdms[all_times >= 0]
rois_repeat = np.array(rois_repeat)[all_times >= 0]
arrays_repeat = np.array(arrays_repeat)[all_times >= 0]
all_labels = np.array(all_labels)[all_times >= 0]
all_times = all_times[all_times >= 0]


def correlate_rdms(rdms, gpu=True):
    '''
        rdms is a matrix where each row is an rdm in flat representation
        computes the correlation between all pairs of rdms
    '''
    if not gpu:
        rdms = rdms.copy()  # no overwriting input data
    else:
        # move to gpu
        rdms = cp.asarray(rdms)

    # mean center each rdm
    rdms -= rdms.mean(axis=1, keepdims=True)

    # divide by standard deviation
    rdms /= rdms.std(axis=1, keepdims=True)

    # outer product
    corrs = (rdms @ rdms.T) / rdms.shape[1]

    if gpu:
        # move to cpu
        corrs = cp.asnumpy(corrs)

    # result has shape [n_rdms x n_rdms]
    return corrs


info = {
    'time': all_times,
    'label': all_labels
}

df = pd.DataFrame.from_dict(info)
mds = MDS(n_components=2, dissimilarity='precomputed')
mds_results = []
unique_times = np.sort(df['time'].unique())
xys = []
xy = None
for ut in unique_times:
    df_t = df[df['time'] == ut]
    df_t = df_t.sort_values(by=['label'])
    print(df_t)
    rdms_t = all_rdms[df_t.index.values]
    rdms_t -= rdms_t.mean(axis=0, keepdims=True)
    dists = 1 - correlate_rdms(rdms_t)
    xy = mds.fit_transform(dists, init=xy)

    res = {
        'x': xy[:, 0],
        'y': xy[:, 1],
        'label': df_t['label'].values
    }

    res = pd.DataFrame.from_dict(res)
    res['time'] = ut
    xys.append(res)

xy_df = pd.concat(xys)



from matplotlib.animation import FuncAnimation

# Sort by time (optional if already sorted)
xy_df = xy_df.sort_values("time")

# Unique time points
times = sorted(xy_df["time"].unique())

# Assign a color to each label
labels = xy_df["label"].unique()
colors = {label: color for label, color in zip(labels, plt.cm.magma(np.linspace(0, 1, len(labels))))}

# Precompute axis limits to stay fixed
x_min, x_max = xy_df["x"].min(), xy_df["x"].max()
y_min, y_max = xy_df["y"].min(), xy_df["y"].max()

fig, ax = plt.subplots()
scatter = ax.scatter([], [])
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Time: 0")

def update(frame_index):
    t = times[frame_index]
    data = xy_df[xy_df["time"] == t]

    # Update scatter points
    scatter.set_offsets(data[["x", "y"]])
    scatter.set_color([colors[l] for l in data["label"]])

    ax.set_title(f"Time: {t}")
    return scatter,

ani = FuncAnimation(fig, update, frames=len(times), interval=200, blit=True)
ani.save('rdms_over_time.gif')
