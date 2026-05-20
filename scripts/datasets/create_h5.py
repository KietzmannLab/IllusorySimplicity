import os
from os import path

import h5py as h5
import numpy as np
from pymatreader import read_mat

cfg = {"monkey": "monkeyN"}

# --- load data
# datadir has the raw mat files
datadir = path.join("datasets", cfg["monkey"])

# list all files in the datadir
# get all mat files containing MUA_trials. These are raw MUAs for different sessions
files = os.listdir(datadir)
mua_files = [
    f
    for f in files
    if f.endswith(".mat") and "MUA_LFP_trials" in f and "Drift-hpf01" in f
]

mua_files.sort()  # sort dates

print(">>> available sessions:")
for f in mua_files:
    print(f)

# get session dates
dates = [f.split("_")[1] for f in mua_files]

data_per_sess = dict()
print(">>> load data")
for date, fname in zip(dates, mua_files):
    print(f"loading session {date}")
    try:
        data = read_mat(path.join(datadir, fname))
        data_per_sess[date] = data
    except Exception as e:
        print("failed read. Error:", e)

dates = list(data_per_sess.keys())

print("data keys:", data_per_sess[dates[0]].keys())
time = data_per_sess[dates[0]]["tb"]
sess_idx = np.concatenate(
    [
        np.ones(d["ALLMAT"].shape[0], dtype=int) * i
        for i, d in enumerate(data_per_sess.values())
    ]
)
stim_id = np.concatenate(
    [d["ALLMAT"][:, 0].astype(int) for d in data_per_sess.values()]
)

mua = np.concatenate([d["ALLMUA"] for d in data_per_sess.values()], axis=1)
lfp = np.concatenate([d["ALLLFP"] for d in data_per_sess.values()], axis=1)


# ------------------------------------------------------------------------
# add additional info
array_id = np.repeat(np.arange(1, 17), 64)

if cfg["monkey"].startswith("monkeyN"):
    v1_arrays = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # note that 6 is reported broken
    v4_arrays = np.array([9, 10, 11, 12])
    it_arrays = np.array([13, 14, 15, 16])
elif cfg["monkey"].startswith("monkeyF"):
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
h5path = path.join(datadir, "allMUA.h5")
f = h5.File(h5path, "w")
f.create_dataset("mua", data=mua)
f.create_dataset("lfp", data=lfp)
f.create_dataset("date", data=np.array([int(d) for d in dates]))
f.create_dataset("rois", data=rois.astype(int))
f.create_dataset("array_ids", data=array_id.astype(int))
f.create_dataset("im_number", data=stim_id.astype(int))
f.create_dataset("time", data=time)
f.create_dataset("sess_idx", data=sess_idx)
f.close()

print(f"Dataset has been saved to {h5path}.")
