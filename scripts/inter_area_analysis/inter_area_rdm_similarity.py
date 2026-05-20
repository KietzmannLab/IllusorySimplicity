import numpy as np
import pickle as pkl
from scipy.spatial.distance import cdist
import sys
from tqdm import tqdm
from os import path
import os

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'

# --------- select RDMs

# if we want a shared cmap, set these for all plots, else None
vmin = .5
vmax = 1.

analysis_dir = "results/inter_area/monkeyF_v4_target_array_10_stride_2_ridgecv_lfp_threefold/analysis"
sub_dirs = [d for d in os.listdir(analysis_dir) if d.startswith('subspace_t')]


ts = [
    int(sd.split('-')[1].split('_')[0]) for sd in sub_dirs
]

ds = [
    int(sd.split('-')[2].split('_')[0]) for sd in sub_dirs
]

mark_times = [
    (t - d, t) for t,d in zip(ts, ds)
]


def compute_and_plot_inter_area_rdm_corr(rdmfile, source_key, target_key, vmin=None, vmax=None, highlight_indices=[]):
    # --- load data
    with open(rdmfile, 'rb') as f:
        rdm_data = pkl.load(f)

    # --- extract data
    time = rdm_data['time']
    rdms_source = rdm_data[source_key]
    rdms_target = rdm_data[target_key]

    dists_per_fold = np.empty((rdms_source.shape[0], rdms_source.shape[1], rdms_source.shape[1]))

    for i in tqdm(range(rdms_source.shape[0]), file=sys.stdout):
        rdm_dists = cdist(rdms_source[i], rdms_target[i], metric='correlation')
        dists_per_fold[i] = rdm_dists

    avg_inter_dists = np.mean(dists_per_fold, axis=0)

    # make sure that the chosen color range does not cause clipping
    if not vmin is None:
        assert avg_inter_dists.min() >= vmin, f'clipping lower bound of cmap. Lowest value: {avg_inter_dists.min()}, vmin: {vmin}'
    if not vmin is None:
        assert avg_inter_dists.max() <= vmax, f'clipping upper bound of cmap. Largest value: {avg_inter_dists.max()}, vmax: {vmax}'

    # get normed markers
    t_norm = (time - time[0]) / time[-1]  # time in 0 - 1
    source_indices_norm = [
        (idx[0] - time[0]) / time[-1] for idx in mark_times
    ]

    source_indices_pixel = [
        idx * len(time) for idx in source_indices_norm
    ]

    target_indices_norm = [
        (idx[1] - time[0]) / time[-1] for idx in mark_times
    ]

    target_indices_pixel = [
        idx * len(time) for idx in target_indices_norm
    ]
    
    scatter_colors = [
        'gray' if not i in highlight_indices else 'white' for i in range(len(mark_times))
    ]

    fig_rdm_inter = plt.figure()
    plt.imshow(avg_inter_dists.T, cmap='magma', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='correlation distance')
    plt.scatter(
        source_indices_pixel, 
        target_indices_pixel, 
        c=scatter_colors,
        edgecolor='k',
        s=10
    )
    plt.plot(
        [0, len(time)], 
        [0, len(time)], 
        color='white',
        linewidth=1
    )
    plt.xlabel('source time (ms)')
    plt.ylabel('target time (ms)')
    plt.xticks(np.arange(len(time))[::50], time[::50])
    plt.yticks(np.arange(len(time))[::50], time[::50])
    plt.xlim((0, len(time)))
    plt.ylim((0, len(time)))

    return fig_rdm_inter, dists_per_fold, time


# ------------------- full RDM time courses

rdmfile = f"{analysis_dir}/{sub_dirs[0]}/rdms.pkl"  # take the first folder since 'ground truth' RDMs are the same in all of them
fig_rdm_inter, dists_full, time = compute_and_plot_inter_area_rdm_corr(rdmfile, 'rdms_source', 'rdms_target', vmin, vmax)

fig_rdm_inter.savefig(
    path.join(analysis_dir, 'rdm_source_target_corr_dist.png'),
    dpi=200)

fig_rdm_inter.savefig(
    path.join(analysis_dir, 'rdm_source_target_corr_dist.svg'))

# set up results dict
res_data = {
    'time': time,
    'dists_full': dists_full,
    'subspace_dirs': sub_dirs,
    'model_times_source_target': mark_times,
    'dists_readout_sub': [],
    'dists_pred': [],
    'info': "all distance matrices have shapes [fold x source_time x target_time]"
}


# ----------------- subspace specific RDM time courses



for i in range(len(sub_dirs)):
    rdmfile = f"{analysis_dir}/{sub_dirs[i]}/rdms.pkl"  # take the first folder since 'ground truth' RDMs are the same in all of them
    fig_rdm_source_sub, dists_readout, time = compute_and_plot_inter_area_rdm_corr(rdmfile, 'rdms_source_readout_subspace', 'rdms_target', vmin, vmax, highlight_indices=[i])
    fig_rdm_source_sub.savefig(path.join(analysis_dir, sub_dirs[i], 'rdm_source_target_corr_dist_source_readout.png'), dpi=200)
    fig_rdm_source_sub.savefig(path.join(analysis_dir, sub_dirs[i], 'rdm_source_target_corr_dist_source_readout.svg'))
    res_data['dists_readout_sub'].append(dists_readout)

    fig_rdm_pred, dists_pred, time = compute_and_plot_inter_area_rdm_corr(rdmfile, 'rdms_target_prediction', 'rdms_target', vmin, vmax, highlight_indices=[i])
    fig_rdm_pred.savefig(path.join(analysis_dir, sub_dirs[i], 'rdm_source_target_corr_dist_prediction.png'), dpi=200)
    fig_rdm_pred.savefig(path.join(analysis_dir, sub_dirs[i], 'rdm_source_target_corr_dist_prediction.svg'))
    res_data['dists_pred'].append(dists_pred)


# store results
save_path = path.join(analysis_dir, 'rdm_inter_correlations.pkl')
print('saving results to:', save_path)
with open(save_path, 'wb') as f:
    pkl.dump(res_data, f)
print('done.')


# ---------------------------------------------------------------
'''
For each subspace, select all three distances at the model (t, d) coordinate
'''

time = res_data['time']

times_source = []
times_target = []
dists_full = []
dists_read = []
dists_pred = []


for i in range(len(res_data['subspace_dirs'])):
    ts = res_data['model_times_source_target'][i]

    # find closest match in time vector (may not be exact match depending on strides chosen)
    closest_idx_source_t = np.argmin(np.abs(time - ts[0]))
    print(f'desired model time (source): {ts[0]}ms. Best match: {time[closest_idx_source_t]}ms')

    closest_idx_target_t = np.argmin(np.abs(time - ts[1]))
    print(f'desired model time (target): {ts[1]}ms. Best match: {time[closest_idx_target_t]}ms')

    # now select three distances:
    # 1. full 'true' data
    # 2. target data to source data in readout subspace
    # 3. target data to prediction from source data

    dist_sel_full = res_data['dists_full'][:, closest_idx_source_t, closest_idx_target_t]
    dist_sel_read =  res_data['dists_readout_sub'][i][:, closest_idx_source_t, closest_idx_target_t]
    dist_sel_pred =  res_data['dists_pred'][i][:, closest_idx_source_t, closest_idx_target_t]

    # store
    times_source.append(time[closest_idx_source_t])
    times_target.append(time[closest_idx_target_t])
    dists_full.append(dist_sel_full)
    dists_read.append(dist_sel_read)
    dists_pred.append(dist_sel_pred)

dists_full = np.array(dists_full)
dists_read = np.array(dists_read)
dists_pred = np.array(dists_pred)

plt.figure()
plt.plot(times_target, dists_full.mean(1), marker='s', label='full signal')
plt.plot(times_target, dists_read.mean(1), marker='o', label='readout signal')
plt.plot(times_target, dists_pred.mean(1), marker='x', label='predicted signal')
plt.xlabel('time in target region (ms)')
plt.ylabel('rdm dissimilarity (correlation distance)')
plt.legend()

plt.savefig(
    path.join(analysis_dir, 'subspace_rdm_corr_scatter.png'), 
    dpi=200
)
plt.savefig(
    path.join(analysis_dir, 'subspace_rdm_corr_scatter.svg'), 
)


# add data to res dict and re-save
subspace_dist_results = {
    'times_source': times_source,
    'times_target': times_target,
    'dists_full': dists_full,
    'dists_read': dists_read,
    'dists_pred': dists_pred
}

res_data['subspace_dist'] = subspace_dist_results

print('adding results to:', save_path)
with open(save_path, 'wb') as f:
    pkl.dump(res_data, f)
print('done.')


