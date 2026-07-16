import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from component.dataset import CsvVolumeDataset, OneClassDataset
from component.vae import VAE
from utils.splits import split_train_val_test


def load_vae(model_dir, device, checkpoint_name):
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / checkpoint_name
    hyperparameter_path = model_dir / "vae_hyperparameter.json"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not hyperparameter_path.is_file():
        raise FileNotFoundError(f"Missing hyperparameter file: {hyperparameter_path}")

    with hyperparameter_path.open("r") as file:
        hyperparameters = json.load(file)

    vae = VAE(
        input_shape=(1, 128, 128, 128),
        featuremap_size=hyperparameters["vae_featuremap_size"],
        base_channel=hyperparameters["vae_base_channel"],
        flatten_latent_dim=None,
        with_residual=hyperparameters.get("vae_use_residual", True),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("vae_state_dict", checkpoint)
    vae.load_state_dict(state_dict)
    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


def sample_subset(dataset, sample_count, rng, name):
    if len(dataset) < sample_count:
        raise ValueError(f"{name} has only {len(dataset)} samples; cannot draw {sample_count}")
    indices = rng.choice(len(dataset), size=sample_count, replace=False).tolist()
    return Subset(dataset, indices), indices


def encode_dataset(vae, dataset, device, batch_size, num_workers, name):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    features = []
    labels = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Encoding {name}", unit="batch"):
            x = x.to(device, non_blocking=True)
            mu, _ = vae.encode(x)
            features.append(mu.flatten(start_dim=1).cpu().numpy())
            labels.extend(y.detach().cpu().view(-1).tolist())
    if not features:
        raise ValueError(f"No samples found while encoding {name}")
    return np.concatenate(features, axis=0), np.asarray(labels)


def reduce_to_2d(features, method, random_state, pca_components, tsne_perplexity, umap_neighbors, umap_min_dist):
    if len(features) < 2:
        raise ValueError("At least two samples are needed for dimensionality reduction")

    max_pca_components = min(pca_components, features.shape[0] - 1, features.shape[1])
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(features), "PCA"

    if max_pca_components < 2:
        pre_reduced = features
    else:
        pre_reduced = PCA(n_components=max_pca_components, random_state=random_state).fit_transform(features)

    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("UMAP requested, but umap-learn is not installed") from exc
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(umap_neighbors, len(features) - 1),
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=random_state,
        )
        return reducer.fit_transform(pre_reduced), f"PCA({max_pca_components}) + UMAP"

    perplexity = min(tsne_perplexity, max(1, len(features) - 1))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return reducer.fit_transform(pre_reduced), f"PCA({max_pca_components}) + t-SNE"


def plot_embedding(embedding, sources, labels, output_path, title):
    source_markers = {
        "train": "o",
        "test": "^",
        "synthetic": "s",
    }
    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab10")
    label_colors = {label: cmap(idx % 10) for idx, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for source in source_markers:
        for label in unique_labels:
            mask = (sources == source) & (labels == label)
            if not np.any(mask):
                continue
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                marker=source_markers[source],
                color=label_colors[label],
                edgecolors="black",
                linewidths=0.4,
                s=60,
                alpha=0.85,
            )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

    shape_handles = [
        Line2D([0], [0], marker=marker, color="black", linestyle="", label=source, markersize=8)
        for source, marker in source_markers.items()
    ]
    color_handles = [
        Line2D([0], [0], marker="o", color=color, linestyle="", label=f"class {label}", markersize=8)
        for label, color in label_colors.items()
    ]
    shape_legend = ax.legend(handles=shape_handles, title="source", loc="upper right")
    ax.add_artist(shape_legend)
    ax.legend(handles=color_handles, title="class", loc="lower right")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_embedding_csv(path, embedding, sources, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source", "label", "x", "y"])
        for source, label, point in zip(sources, labels, embedding):
            writer.writerow([source, int(label), float(point[0]), float(point[1])])


def main():
    parser = argparse.ArgumentParser(
        description="Sample train/test/synthetic volumes, encode with VAE, reduce latents to 2D, and plot them."
    )
    parser.add_argument("--image_dir", type=str, required=True, help="folder containing labeled real NIfTI volumes")
    parser.add_argument("--labels_csv", type=str, required=True, help="CSV containing filenames and labels")
    parser.add_argument("--synthetic_data_dir", type=str, required=True, help="folder containing synthetic NIfTI volumes")
    parser.add_argument("--model_dir", type=str, required=True, help="trained VAE folder")
    parser.add_argument("--checkpoint_name", type=str, default="best.pth", help="VAE checkpoint inside model_dir")
    parser.add_argument("--filename_column", type=str, default="filename", help="CSV filename column")
    parser.add_argument("--label_column", type=str, default="label", help="CSV label column")
    parser.add_argument("--val_split", type=float, default=0.1, help="validation split ratio, matching train_vae.py")
    parser.add_argument("--test_split", type=float, default=0.1, help="test split ratio, matching train_vae.py")
    parser.add_argument("--random_state", type=int, default=42, help="seed for split, sampling, and reduction")
    parser.add_argument("--sample_count", type=int, default=50, help="samples drawn from each of train/test/synthetic")
    parser.add_argument("--synthetic_label", type=int, default=-1, help="class label assigned to synthetic samples")
    parser.add_argument("--batch_size", type=int, default=4, help="encoding batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--method", choices=("umap", "tsne", "pca"), default="umap",
                        help="2D reduction method; UMAP is recommended")
    parser.add_argument("--pca_components", type=int, default=50,
                        help="PCA dimensions before UMAP/t-SNE")
    parser.add_argument("--umap_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--output_path", type=str, default="vae_latent_2d.png", help="output plot path")
    parser.add_argument("--embedding_csv", type=str, default="", help="optional path to save 2D coordinates")
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Encoding with", "GPU" if device.type == "cuda" else "CPU")

    vae = load_vae(args.model_dir, device, args.checkpoint_name)
    real_dataset = CsvVolumeDataset(
        args.image_dir,
        args.labels_csv,
        filename_column=args.filename_column,
        label_column=args.label_column,
    )
    synthetic_dataset = OneClassDataset(args.synthetic_data_dir)
    train_dataset, _, test_dataset = split_train_val_test(
        real_dataset,
        args.val_split,
        args.test_split,
        args.random_state,
    )

    train_subset, _ = sample_subset(train_dataset, args.sample_count, rng, "train split")
    test_subset, _ = sample_subset(test_dataset, args.sample_count, rng, "test split")
    synthetic_subset, _ = sample_subset(synthetic_dataset, args.sample_count, rng, "synthetic folder")

    train_features, train_labels = encode_dataset(vae, train_subset, device, args.batch_size, args.num_workers, "train")
    test_features, test_labels = encode_dataset(vae, test_subset, device, args.batch_size, args.num_workers, "test")
    synthetic_features, _ = encode_dataset(
        vae,
        synthetic_subset,
        device,
        args.batch_size,
        args.num_workers,
        "synthetic",
    )
    synthetic_labels = np.full(len(synthetic_features), args.synthetic_label)

    features = np.concatenate([train_features, test_features, synthetic_features], axis=0)
    labels = np.concatenate([train_labels, test_labels, synthetic_labels], axis=0).astype(int)
    sources = np.asarray(
        ["train"] * len(train_features)
        + ["test"] * len(test_features)
        + ["synthetic"] * len(synthetic_features)
    )

    embedding, method_name = reduce_to_2d(
        features,
        args.method,
        args.random_state,
        args.pca_components,
        args.tsne_perplexity,
        args.umap_neighbors,
        args.umap_min_dist,
    )
    title = f"VAE latent 2D embedding ({method_name})"
    plot_embedding(embedding, sources, labels, args.output_path, title)
    print(f"Saved plot to {args.output_path}")

    if args.embedding_csv:
        write_embedding_csv(args.embedding_csv, embedding, sources, labels)
        print(f"Saved embedding CSV to {args.embedding_csv}")


if __name__ == "__main__":
    main()
