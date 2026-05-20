import numpy as np
import os
from os import path
from pymatreader import read_mat
import h5py as h5


cfg = {"monkey": "monkeyN"}


# --- load data
datadir = path.join("datasets", cfg["monkey"])

# list all files
files = os.listdir(datadir)
mua_files = [f for f in files if "normMUA" in f if f.endswith(".mat")]
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
rois = data_per_sess[dates[0]]["rois"]
snr = np.array([d["SNR"] for d in data_per_sess.values()])
time = data_per_sess[dates[0]]["tb"]
sess_idx = np.concatenate(
    [
        np.ones(d["ALLMAT"].shape[0], dtype=int) * i
        for i, d in enumerate(data_per_sess.values())
    ]
)

mua = np.concatenate([d["normMUA"] for d in data_per_sess.values()], axis=1)
allmat = np.concatenate([d["ALLMAT"] for d in data_per_sess.values()], axis=0)

# save as h5
h5path = path.join(datadir, "normMUA.h5")
f = h5.File(h5path, "w")
f.create_dataset("mua", data=mua)
f.create_dataset("date", data=np.array([int(d) for d in dates]))
f.create_dataset("rois", data=rois.astype(int))
f.create_dataset("im_number", data=allmat[:, 0].astype(int))
f.create_dataset("isi_duration", data=allmat[:, 1])
f.create_dataset("pos_in_block", data=allmat[:, 2].astype(int))
f.create_dataset("time", data=time)
f.create_dataset("snr", data=snr)
f.create_dataset("sess_idx", data=sess_idx)
f.close()
