import torchvision
import torch
import os
import numpy as np
import pickle as pkl
from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets.folder import DatasetFolder

dataset_save_folder = os.path.join("datasets", "embeddings")
dataset_name = "alexnet_imagenet1k_v1.pkl"
dataset_savepath = os.path.join(dataset_save_folder, dataset_name)

os.makedirs(dataset_save_folder, exist_ok=False)  # dont overwrite dataset if it exists

model = torchvision.models.alexnet(weights="IMAGENET1K_V1")
model.eval()  # no randomness
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

layers = {
    "conv1": model.features[0],
    "conv2": model.features[3],
    "conv3": model.features[6],
    "conv4": model.features[8],
    "conv5": model.features[10],
    "fc1": model.classifier[1],
    "fc2": model.classifier[4],
}

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.Lambda(lambda x: x.float() / 255.0),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ]
)


# overwrite getitem to return the sample image path
class DatasetFolderWithFilename(DatasetFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, os.path.split(path)[-1]


dset = DatasetFolderWithFilename(
    os.path.join("datasets", "stimuli_folder_dataset"),
    loader=torchvision.io.decode_image,
    extensions="jpg",
    transform=transform,
)

# attach hooks
activations = {k: [] for k in layers.keys()}


def activation_hook(target_name, target_list, module, input, output):
    target_list.append(output)


for name, module in layers.items():
    # register the activation_hook
    module.register_forward_hook(partial(activation_hook, name, activations[name]))


loader = DataLoader(dset, batch_size=100, shuffle=False)

cats = []
fnames = []

with torch.no_grad():
    for batch in tqdm(loader):
        x, y, fname = batch
        _ = model(x)
        cats.append(y)
        fnames.append(fname)

# concatenate into np arrays
cats = np.concatenate(cats, axis=0)
fnames = np.concatenate(fnames, axis=0)
activations = {k: np.concatenate(v, axis=0) for k, v in activations.items()}

for k, v in activations.items():
    print(f"{k}: {v.shape}")

print(f"category: {cats.shape}")
print(f"file name: {fnames.shape}")

results = {"embeddings": activations, "categoty": cats, "filenames": fnames}

with open(dataset_savepath, "wb") as f:
    pkl.dump(results, f)


# testing
from ephyslib.rdm import compute_robust_rdm
from macaquethings.rdm_util import get_rdm_design_sort_indices
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

layer = "conv5"
x = activations[layer]

if len(x.shape) > 2:
    x = x.mean(axis=(2, 3))  # avg over space for conv layers

y = cats
labels = np.unique(y)
x -= x.mean(axis=0)  # mean center
rdm = compute_robust_rdm(x, y, labels, navg=8, nsamples=100, metric="correlation")

sort_idx, info_class, df = get_rdm_design_sort_indices(return_values=True)

plt.figure()
plt.imshow(squareform(rdm)[sort_idx][:, sort_idx], vmin=0, vmax=2, cmap="magma")
plt.colorbar()
plt.show()
