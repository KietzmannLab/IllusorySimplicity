from os import path

import h5py as h5
import numpy as np

dset_path = path.join("datasets", "monkeyN", "allMUA.h5")
fhandle = h5.File(dset_path, "r+")

sess_idx = fhandle["sess_idx"][:]

# the last session for monkeyN was recorded twice.
# The second attempt has the full 3000 trials. Increment the sess_idx by 1 to
# ensure we can access the recording attempts separately
sess_idx[-3000:] = 5

# save back to the h5 file
fhandle["sess_idx"][...] = sess_idx

fhandle.close()

# verify
fhandle = h5.File(dset_path, "r")
print(np.unique(fhandle["sess_idx"][:], return_counts=True))
fhandle.close()
