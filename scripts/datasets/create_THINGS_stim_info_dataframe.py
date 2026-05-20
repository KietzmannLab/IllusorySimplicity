import pandas as pd
import numpy as np
from os import path
from pymatreader import read_mat

stimulus_mat = read_mat(path.join("datasets", "things_imgs.mat"))
train_stiminfo = stimulus_mat["train_imgs"]
test_stiminfo = stimulus_mat["test_imgs"]


train_category = np.array(train_stiminfo["class"])
test_category = np.array(test_stiminfo["class"])
train_paths = train_stiminfo["things_path"]
test_paths = test_stiminfo["things_path"]

train_filenames = np.array([path.basename(x) for x in train_paths])
test_filenames = np.array([path.basename(x) for x in test_paths])

train_im_id = np.arange(len(train_filenames)) + 1
test_im_id = np.arange(len(test_filenames)) + 1

train_df = pd.DataFrame.from_dict(
    {"im_id": train_im_id, "filenames": train_filenames, "category": train_category}
)

test_df = pd.DataFrame.from_dict(
    {"im_id": test_im_id, "filenames": test_filenames, "category": test_category}
)

train_df.to_csv(path.join("datasets", "stimulus_information_22k_train.csv"))
test_df.to_csv(path.join("datasets", "stimulus_information_22k_test.csv"))
