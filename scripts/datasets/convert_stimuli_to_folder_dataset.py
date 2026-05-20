import os
from os import path
import numpy as np
from PIL import Image
from tqdm import tqdm

stim_root = path.join("datasets", "stimuli")

# find all stimuli
stims = np.array([x for x in os.listdir(stim_root) if x.endswith(".bmp")])


# get classes
classes = np.array(["_".join(x.split("_")[:-1]) for x in stims])
unique_classes = np.unique(classes)

# make target folder
target_folder = path.join("datasets", "stimuli_folder_dataset")
os.makedirs(target_folder, exist_ok=False)

for uc in tqdm(unique_classes):
    class_folder = path.join(target_folder, uc)
    os.makedirs(class_folder, exist_ok=False)

    # find all images for class
    mask = classes == uc
    class_files = stims[mask]

    # load and save as jpg
    for f in class_files:
        fpath = path.join(stim_root, f)
        ftarget_path = path.join(class_folder, f[:-4] + ".jpg")
        img = Image.open(fpath)
        img.save(ftarget_path, "jpeg")
