import numpy as np
import os
from os import path
from pymatreader import read_mat
import h5py as h5

cfg = {"monkey": "monkeyF"}

# --- load data

# datadir has the raw mat files
datadir = path.join("datasets", cfg["monkey"])

# list mat files
mat_files = [f for f in os.listdir(datadir) if f.endswith(".mat")]
things_lfp_mua_files = [f for f in mat_files if "longtrial" in f]
things_lfp_mua_files = np.sort(things_lfp_mua_files)

mua_shapes = []
lfp_shapes = []

for i, fname in enumerate(things_lfp_mua_files):
    fhandle = h5.File(path.join(datadir, fname), "r")
    muashape = fhandle["ALLMUA"].shape
    lfpshape = fhandle["ALLLFP"].shape
    mua_shapes.append(muashape)
    lfp_shapes.append(lfpshape)

mua_shapes = np.array(mua_shapes)
lfp_shapes = np.array(lfp_shapes)

mua_shape = np.array([mua_shapes[0][0], mua_shapes[:, 1].sum(), mua_shapes[2, 2]])
lfp_shape = np.array([lfp_shapes[0][0], lfp_shapes[:, 1].sum(), lfp_shapes[2, 2]])

print(mua_shape, lfp_shape)

# read the data
# - pre-allocate arrays for data

mua = np.empty(mua_shape)
lfp = np.empty(lfp_shape)

mua.fill(np.nan)
lfp.fill(np.nan)

# h5py reads data in different order from read_mat
# reorder to correct shapes:
mua = np.swapaxes(mua, 0, 2)
lfp = np.swapaxes(lfp, 0, 2)


train_idx_mask = np.empty(mua.shape[1])
test_idx_mask = np.empty(mua.shape[1])

allmat = np.empty((mua.shape[1], 7))

n_trials_loaded = 0

# load data per session
for fname in things_lfp_mua_files:
    print(f"loading: {fname} ...")
    data = read_mat(path.join(datadir, fname))

    time = data["tb"]
    sess_mua = data["ALLMUA"]
    sess_lfp = data["ALLLFP"]

    sess_allmat = data["ALLMAT"]
    # store
    n_sess_trials = sess_mua.shape[1]
    mua[:, n_trials_loaded : n_trials_loaded + n_sess_trials] = sess_mua
    lfp[:, n_trials_loaded : n_trials_loaded + n_sess_trials] = sess_lfp
    allmat[n_trials_loaded : n_trials_loaded + n_sess_trials] = sess_allmat

    n_trials_loaded += n_sess_trials

assert not np.isnan(mua).any()
assert not np.isnan(lfp).any()


sess_idx = allmat[:, -1]
# convert from matlab indexing and ensure int
sess_idx = sess_idx.astype(int) - 1

train_idx_mask = allmat[:, 1] > 0
test_idx_mask = allmat[:, 2] > 0

# together these two masks should sum to the number of mua trials
assert (np.sum(train_idx_mask) + np.sum(test_idx_mask)) == mua.shape[
    1
], "train and test indices don't sum to total number of trials"

# stimulus indices are used to get the stim info from the stimulus mat
train_im_num = allmat[train_idx_mask, 1]
test_im_num = allmat[test_idx_mask, 2]

# should be integer type, convert
train_im_num = train_im_num.astype(int)
test_im_num = test_im_num.astype(int)

# get the mua data for train and test
train_mua = mua[:, train_idx_mask]
test_mua = mua[:, test_idx_mask]

# get the lfp data for train and test
train_lfp = lfp[:, train_idx_mask]
test_lfp = lfp[:, test_idx_mask]

train_sess_idx = sess_idx[train_idx_mask]
test_sess_idx = sess_idx[test_idx_mask]

print("MUA:")
print("train", train_mua.shape)
print("test", test_mua.shape)
print()
print("LFP:")
print("train", train_lfp.shape)
print("test", test_lfp.shape)

array_id = np.repeat(np.arange(1, 17), 64)

if cfg["monkey"] == "monkeyN":
    v1_arrays = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # note that 6 is reported broken
    v4_arrays = np.array([9, 10, 11, 12])
    it_arrays = np.array([13, 14, 15, 16])
elif cfg["monkey"] == "monkeyF":
    v1_arrays = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    v4_arrays = np.array([14, 15, 16])
    it_arrays = np.array([9, 10, 11, 12, 13])
else:
    raise NotImplementedError("this monkey does not exist in the dataset")

rois = np.zeros_like(array_id)
rois[np.isin(array_id, v1_arrays)] = 1
rois[np.isin(array_id, v4_arrays)] = 2
rois[np.isin(array_id, it_arrays)] = 3

print("saving train data ...")
# create h5 file with data
h5path = path.join(datadir, "allMUA_22k_lfp_mua_train.h5")
f = h5.File(h5path, "w")
f.create_dataset("mua", data=train_mua)
f.create_dataset("lfp", data=train_lfp)
f.create_dataset("rois", data=rois.astype(int))
f.create_dataset("array_ids", data=array_id.astype(int))
f.create_dataset("im_number", data=train_im_num)
f.create_dataset("time", data=time)
f.create_dataset("sess_idx", data=train_sess_idx)
f.close()

print("saving test data ...")
# create h5 file with data
h5path = path.join(datadir, "allMUA_22k_lfp_mua_test.h5")
f = h5.File(h5path, "w")
f.create_dataset("mua", data=test_mua)
f.create_dataset("lfp", data=test_lfp)
f.create_dataset("rois", data=rois.astype(int))
f.create_dataset("array_ids", data=array_id.astype(int))
f.create_dataset("im_number", data=test_im_num)
f.create_dataset("time", data=time)
f.create_dataset("sess_idx", data=test_sess_idx)
f.close()

print("done.")
