import numpy as np
from os import path
from pymatreader import read_mat
import h5py as h5

cfg = {"monkey": "monkeyF"}

# --- load data

# datadir has the raw mat files
datadir = path.join("datasets", cfg["monkey"])
map_mat = read_mat(path.join("datasets", "1024chns_mapping_20220105.mat"))
mapping = map_mat["mapping"]
mapping -= 1  # matlab

data = read_mat(path.join(datadir, "THINGS_MUA_trials.mat"))
time = data["tb"][:]
mua = data["ALLMUA"]
mua = mua[mapping]

allmat = data["ALLMAT"]
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

train_sess_idx = sess_idx[train_idx_mask]
test_sess_idx = sess_idx[test_idx_mask]

print("train", train_mua.shape)
print("test", test_mua.shape)


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

# create h5 file with data
h5path = path.join(datadir, "allMUA_22k_train.h5")
f = h5.File(h5path, "w")
f.create_dataset("mua", data=train_mua)
f.create_dataset("rois", data=rois.astype(int))
f.create_dataset("array_ids", data=array_id.astype(int))
f.create_dataset("im_number", data=train_im_num)
f.create_dataset("time", data=time)
f.create_dataset("sess_idx", data=train_sess_idx)
f.close()

# create h5 file with data
h5path = path.join(datadir, "allMUA_22k_test.h5")
f = h5.File(h5path, "w")
f.create_dataset("mua", data=test_mua)
f.create_dataset("rois", data=rois.astype(int))
f.create_dataset("array_ids", data=array_id.astype(int))
f.create_dataset("im_number", data=test_im_num)
f.create_dataset("time", data=time)
f.create_dataset("sess_idx", data=test_sess_idx)
f.close()
