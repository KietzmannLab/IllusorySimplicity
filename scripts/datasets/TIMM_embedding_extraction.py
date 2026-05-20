import argparse
import os
import pickle as pkl
from os import path

import imageio
import numpy as np
import timm
import torch
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from sklearn.decomposition import PCA
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from tqdm import tqdm

# Default options
root = "."
layer_names = ["conv", "fc"]
model_name = "resnet18"
pca = 1000
normalization = "None"
metric = "cosine"

parser = argparse.ArgumentParser(
    description="Extract TIMM model embeddings and compute RDMs."
)
parser.add_argument("--root", type=str, default=root, help="Root directory")
parser.add_argument(
    "--layer_names",
    type=str,
    nargs="+",
    default=layer_names,
    help="Layer keywords to extract",
)
parser.add_argument(
    "--model_name", type=str, default=model_name, help="TIMM model name"
)
parser.add_argument("--pca", type=int, default=pca, help="Number of PCA components")
parser.add_argument(
    "--normalization",
    type=str,
    choices=["None", "centering", "zscore"],
    default=normalization,
    help="Normalization method",
)
parser.add_argument(
    "--metric", type=str, default=metric, help="Distance metric for RDM"
)

args = parser.parse_args()
root = args.root
layer_names = args.layer_names
model_name = args.model_name
pca = args.pca
normalization = args.normalization
metric = args.metric

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


def load_mini_THINGS_as_tensors(imfolder):
    imnames = np.sort(os.listdir(imfolder))
    images = list()
    for fname in imnames:
        images.append(imageio.v2.imread(path.join(imfolder, fname)))
    images = torch.from_numpy(np.array(images).astype(np.float32) / 255.0)
    images = torch.moveaxis(images, -1, 1)
    return images, imnames


def get_nodes_for_keywords(nodelist, keywords):
    selected = []
    indices = []  # save indices into original list
    for idx, node in enumerate(nodelist):
        for kw in keywords:
            if kw in node:
                selected.append(node)
                indices.append(idx)
                break  # exit after first match, avoid adding duplicates
    return selected, indices


# load images
imfolder = path.join(root, "datasets", "stimuli")
images, imnames = load_mini_THINGS_as_tensors(imfolder)

# load model and find layers that match selection criteria
model = timm.create_model(model_name, pretrained=True, exportable=True)
_, eval_nodes = get_graph_node_names(model)
selected_nodes, indices = get_nodes_for_keywords(eval_nodes, layer_names)

# get data transform
data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_cfg)

# get feature extractor for selected layers
feature_extractor = create_feature_extractor(model, return_nodes=selected_nodes)
feature_extractor.eval()

# get embeddings
with torch.no_grad():
    activations = feature_extractor(transform(images))
    for k, v in activations.items():
        print(f"{k}: {v.shape}")

# Save embeddings and metadata
embeddings_output_path = os.path.join(
    savedir,
    f"embeddings-{model_name}.pkl",
)

parameters = {
    "model_name": model_name,
    "selected_nodes": selected_nodes,
    "layer_indices": indices,
    "pca": pca,
    "normalization": normalization,
    "metric": metric,
}

embeddings_out = dict(
    parameters=parameters,
    selected_nodes=selected_nodes,
    node_indices=indices,
    stimulus_names=imnames,
    embeddings={k: v.detach().cpu().numpy() for k, v in activations.items()},
)

with open(embeddings_output_path, "wb") as f:
    pkl.dump(embeddings_out, f)

print(f"Saved embeddings and metadata to {embeddings_output_path}")


# compute rdms
rdms = {
    k: compute_embedding_rdm(v.detach().numpy(), metric, normalization, pca)
    for k, v in tqdm(activations.items())
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
    selected_nodes=selected_nodes,
    node_indices=indices,
    rdms=rdms,
)

with open(output_path, "wb") as f:
    pkl.dump(out, f)


print(f"Saved RDMs and metadata to {output_path}")
