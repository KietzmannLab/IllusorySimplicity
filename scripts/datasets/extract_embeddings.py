import os
from os import path
import numpy as np
import imageio
import matplotlib.pyplot as plt
from functools import partial
import pickle

import torch
import torchvision


from macaquethings import rdm_util
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, zscore
from sklearn.decomposition import PCA

sns.set_theme(style="white")

root = "."
embedding_savedir = path.join(root, "datasets", "DNN_embeddings")
imfolder = path.join(root, "datasets", "stimuli")
metric = "cosine"
normalization = "centering"
pca = 1000  # False or input to n_components for PCA. Can be integer or percent variance explained
rank_transform = True


def load_mini_THINGS_as_tensors(imfolder):
    imnames = np.sort(os.listdir(imfolder))
    images = list()
    for fname in imnames:
        images.append(imageio.v2.imread(path.join(imfolder, fname)))
    images = torch.from_numpy(np.array(images).astype(np.float32) / 255.0)
    images = torch.moveaxis(images, -1, 1)
    return images, imnames


def setup_vgg16():
    model = torchvision.models.vgg16(weights="IMAGENET1K_V1")
    model.eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ]
    )

    # layers to extract
    layers = {
        "conv1": model.features[0],
        "conv2": model.features[2],
        "conv3": model.features[5],
        "conv4": model.features[7],
        "conv5": model.features[10],
        "conv6": model.features[12],
        "conv7": model.features[14],
        "conv8": model.features[17],
        "conv9": model.features[19],
        "conv10": model.features[21],
        "conv11": model.features[24],
        "conv12": model.features[26],
        "conv13": model.features[28],
        "fc1": model.classifier[0],
        "fc2": model.classifier[3],
    }

    return model, transform, layers


def setup_alexnet():
    model = torchvision.models.alexnet(weights="IMAGENET1K_V1")
    model.eval()  # no randomness
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ]
    )

    # layers to extract
    layers = {
        "conv1": model.features[0],
        "conv2": model.features[3],
        "conv3": model.features[6],
        "conv4": model.features[8],
        "conv5": model.features[10],
        "fc1": model.classifier[1],
        "fc2": model.classifier[4],
    }

    return model, transform, layers


def setup_resnet50():
    model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    model.eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ]
    )

    layers = {
        "block_1": model.layer1,
        "block_2": model.layer2,
        "block_3": model.layer3,
        "block_4": model.layer4,
    }

    return model, transform, layers


def activation_hook(target_name, target_list, module, layer_input, output):
    target_list.append(
        output.clone()
    )  # clone to make sure inplace operations do not cause trouble


def extract_activations(model, transform, layers, images):
    # preprocess images with model transforms
    ims_transform = transform(images)

    # attach hooks
    activations = {k: [] for k in layers.keys()}

    # register hook for each chosen layer
    for name, module in layers.items():
        # register the activation_hook
        module.register_forward_hook(partial(activation_hook, name, activations[name]))

    # pass stimuli through model
    with torch.no_grad():
        logits = model(ims_transform)

    # collect images as tensor
    activations = {k: torch.concatenate(activations[k]) for k, v in activations.items()}
    activations["logits"] = logits

    return activations


images, imnames = load_mini_THINGS_as_tensors(imfolder)

print()
print("************** Alexnet")
print()
alexnet, alexnet_transforms, alexnet_layers = setup_alexnet()
alexnet_embeddings = extract_activations(
    alexnet, alexnet_transforms, alexnet_layers, images
)

for k, v in alexnet_embeddings.items():
    print(f"{k}: {v.shape}")

print()
print("************** ResNet50")
print()

resnet, resnet_transforms, resnet_layers = setup_resnet50()
resnet_embeddings = extract_activations(
    resnet, resnet_transforms, resnet_layers, images
)

for k, v in resnet_embeddings.items():
    print(f"{k}: {v.shape}")


print()
print("************** VGG16")
print()

vgg, vgg_transforms, vgg_layers = setup_vgg16()
vgg_embeddings = extract_activations(vgg, vgg_transforms, vgg_layers, images)

for k, v in vgg_embeddings.items():
    print(f"{k}: {v.shape}")


# -------------------------------------------------------------------------------------------- RDM


sort_idx = rdm_util.get_rdm_design_sort_indices(
    root=root, reduce_to_column="filenames", return_values=False
)


def compute_embedding_rdm(embeddings, layer, metric, normalization, pca):
    xs = embeddings[layer]
    if len(xs.shape) > 2:
        print(f"received embeddings with shape {xs.shape}. Flattening feature dims ...")
        xs = np.reshape(xs, (xs.shape[0], -1))  # flatten feature dims
        print(f"new shape: {xs.shape}")

    if pca:
        print(f"Applying PCA to embeddings, n_components = {pca} ...")
        pca_obj = PCA(n_components=pca, svd_solver="randomized")
        xs = pca_obj.fit_transform(xs)
        print(f"number of components kept: {pca_obj.n_components_}.")

    if normalization == "centering":
        print("centering embeddings ...")
        xs = xs - xs.mean(axis=0)
    elif normalization == "zscore":
        print("z-scoring embeddings ...")
        xs = zscore(xs, axis=0)
    elif normalization == "None":
        print("no normalization applied.")
    else:
        raise NotImplementedError(f"unknown normalization method {normalization}")

    print(f"computing distances for layer {layer}, with metric {metric} ...")
    dists = pdist(xs, metric=metric)
    print("done.")
    return dists


rdms_alexnet = {
    l: compute_embedding_rdm(alexnet_embeddings, l, metric, normalization, pca)
    for l in alexnet_embeddings.keys()
}
rdms_resnet = {
    l: compute_embedding_rdm(resnet_embeddings, l, metric, normalization, pca)
    for l in resnet_embeddings.keys()
}

rdms_vgg = {
    l: compute_embedding_rdm(vgg_embeddings, l, metric, normalization, pca)
    for l in vgg_embeddings.keys()
}

# set up savedirs and save embeddings
alexnet_savedir = path.join(embedding_savedir, "alexnet")
resnet_savedir = path.join(embedding_savedir, "resnet50")
vgg_savedir = path.join(embedding_savedir, "vgg16")

os.makedirs(alexnet_savedir, exist_ok=True)
os.makedirs(resnet_savedir, exist_ok=True)
os.makedirs(vgg_savedir, exist_ok=True)

alexnet_embedding_dict = {"image_names": imnames, "embeddings": alexnet_embeddings}
resnet_embedding_dict = {"image_names": imnames, "embeddings": resnet_embeddings}
vgg_embedding_dict = {"image_names": imnames, "embeddings": vgg_embeddings}

# pickle embeddings
with open(path.join(alexnet_savedir, "embeddings.pkl"), "wb") as f:
    pickle.dump(alexnet_embedding_dict, f)

with open(path.join(resnet_savedir, "embeddings.pkl"), "wb") as f:
    pickle.dump(resnet_embedding_dict, f)

with open(path.join(vgg_savedir, "embeddings.pkl"), "wb") as f:
    pickle.dump(vgg_embedding_dict, f)


# pickle rdms
with open(
    path.join(alexnet_savedir, f"rdms_{metric}_{normalization}_{pca}.pkl"), "wb"
) as f:
    pickle.dump(rdms_alexnet, f)

with open(
    path.join(resnet_savedir, f"rdms_{metric}_{normalization}_{pca}.pkl"), "wb"
) as f:
    pickle.dump(rdms_resnet, f)

with open(
    path.join(vgg_savedir, f"rdms_{metric}_{normalization}_{pca}.pkl"), "wb"
) as f:
    pickle.dump(rdms_vgg, f)


for layername, dists in rdms_alexnet.items():
    fig = plt.figure()
    if rank_transform:
        dists = rankdata(dists)
    plt.imshow(squareform(dists)[sort_idx][:, sort_idx])
    plt.title(
        f"AlexNet - {layername}, {normalization}, {metric}, ranks {rank_transform}"
    )
    if not rank_transform:
        plt.colorbar()
    fig.savefig(
        path.join(
            alexnet_savedir,
            f"rdm_alexnet_{layername}_{normalization}_{metric}_{pca}_{rank_transform}.svg",
        )
    )

for layername, dists in rdms_resnet.items():
    fig = plt.figure()
    if rank_transform:
        dists = rankdata(dists)
    plt.imshow(squareform(dists)[sort_idx][:, sort_idx])
    plt.title(
        f"ResNet - {layername}, {normalization}, {metric}, ranks {rank_transform}"
    )
    if not rank_transform:
        plt.colorbar()
    fig.savefig(
        path.join(
            resnet_savedir,
            f"rdm_resnet_{layername}_{normalization}_{metric}_{pca}_{rank_transform}.svg",
        )
    )

for layername, dists in rdms_vgg.items():
    fig = plt.figure()
    if rank_transform:
        dists = rankdata(dists)
    plt.imshow(squareform(dists)[sort_idx][:, sort_idx])
    plt.title(f"VGG - {layername}, {normalization}, {metric}, ranks {rank_transform}")
    if not rank_transform:
        plt.colorbar()
    fig.savefig(
        path.join(
            vgg_savedir,
            f"rdm_vgg_{layername}_{normalization}_{metric}_{pca}_{rank_transform}.svg",
        )
    )

plt.show()
