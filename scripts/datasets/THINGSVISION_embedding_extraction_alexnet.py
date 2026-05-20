import os
import pickle as pkl
from glob import glob
from os import path

import numpy as np
import torchvision
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from sklearn.decomposition import PCA
from thingsvision import get_extractor
from torch.utils.data import DataLoader
from tqdm import tqdm

# Default options
root = "."
layer_names = [
    "features.0",
    "features.3",
    "features.6",
    "features.8",
    "features.10",
    "classifier.1",
    "classifier.4",
]
model_name = "alexnet"
pca = 1000
normalization = "None"
metric = "cosine"

extractor = get_extractor(
    model_name=model_name,
    source="torchvision",
    device="cpu",
    pretrained=True,
)
transform = extractor.get_transformations()

savedir = path.join("datasets", "TIMM", model_name)


def compute_embedding_rdm(xs, metric, normalization, pca):
    if len(xs.shape) > 2:
        xs = np.reshape(xs, (xs.shape[0], -1))  # flatten feature dims

    if pca:
        pca_obj = PCA(n_components=pca)
        xs = pca_obj.fit_transform(xs)

    if normalization == "centering":
        xs -= xs.mean(axis=0)
    elif normalization == "zscore":
        xs = zscore(xs, axis=0)
    elif normalization == "None":
        pass
    else:
        raise NotImplementedError(f"unknown normalization method {normalization}")

    dists = pdist(xs, metric=metric)
    return dists


imfolder = path.join(
    root, "datasets", "stimuli_folder_dataset", "stimuli_folder_dataset"
)

imnames = glob(f"{imfolder}/**/*.jpg")
imnames = [fp.split("/")[-1] for fp in imnames]

dataset = torchvision.datasets.ImageFolder(
    imfolder,
    transform=extractor.get_transformations(),
    target_transform=None,
    is_valid_file=None,
)
batches = DataLoader(dataset=dataset, batch_size=1000)


embeddings = dict()

for module_name in layer_names:
    features = extractor.extract_features(
        batches=batches, module_name=module_name, flatten_acts=False
    )
    embeddings[module_name] = features

# Save embeddings and metadata
embeddings_output_path = os.path.join(
    savedir,
    f"embeddings-{model_name}.pkl",
)

parameters = {
    "model_name": model_name,
    "selected_nodes": layer_names,
    "layer_indices": np.arange(len(layer_names)),
    "pca": pca,
    "normalization": normalization,
    "metric": metric,
}

embeddings_out = dict(
    parameters=parameters,
    selected_nodes=layer_names,
    node_indices=np.arange(len(layer_names)),
    stimulus_names=imnames,
    embeddings=embeddings,
)

with open(embeddings_output_path, "wb") as f:
    pkl.dump(embeddings_out, f)

print(f"Saved embeddings and metadata to {embeddings_output_path}")


# compute rdms
rdms = {
    k: compute_embedding_rdm(v, metric, normalization, pca)
    for k, v in tqdm(embeddings.items())
}

for k, v in rdms.items():
    print(f"{k}: {v.shape}")

# Save RDMs and metadata
os.makedirs(savedir, exist_ok=True)
output_path = os.path.join(
    savedir,
    f"rdms-{model_name}-metric_{metric}-normalization_{normalization}-pca_{pca}.pkl",
)

out = dict(
    parameters=parameters,
    selected_nodes=layer_names,
    node_indices=np.arange(len(layer_names)),
    rdms=rdms,
)

with open(output_path, "wb") as f:
    pkl.dump(out, f)


print(f"Saved RDMs and metadata to {output_path}")
