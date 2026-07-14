import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
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
    with torch.no_grad():
        for x, _ in tqdm(loader, desc=f"Encoding {name}", unit="batch"):
            x = x.to(device, non_blocking=True)
            mu, _ = vae.encode(x)
            features.append(mu.flatten(start_dim=1).cpu())
    if not features:
        raise ValueError(f"No samples found while encoding {name}")
    return torch.cat(features, dim=0)


def nearest_neighbor_distances(query_features, reference_features, chunk_size, distance_device):
    if distance_device == "auto":
        distance_device = "cuda" if torch.cuda.is_available() else "cpu"
    if distance_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--distance_device cuda was requested, but CUDA is not available")
    device = torch.device("cuda" if distance_device == "cuda" else "cpu")
    reference_features = reference_features.to(device)
    distances = []

    for start in tqdm(range(0, len(query_features), chunk_size), desc="Nearest neighbours", unit="chunk"):
        query_chunk = query_features[start:start + chunk_size].to(device)
        chunk_distances = torch.cdist(query_chunk, reference_features, p=2).min(dim=1).values
        distances.append(chunk_distances.cpu())

    return torch.cat(distances, dim=0).numpy()


def summarize_distances(name, distances):
    return {
        "comparison": name,
        "count": int(distances.size),
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "min": float(np.min(distances)),
        "p05": float(np.percentile(distances, 5)),
        "p25": float(np.percentile(distances, 25)),
        "median": float(np.median(distances)),
        "p75": float(np.percentile(distances, 75)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
    }


def print_summary(summary):
    print(
        f"{summary['comparison']}: n={summary['count']} | "
        f"mean={summary['mean']:.6f} | std={summary['std']:.6f} | "
        f"median={summary['median']:.6f} | p05={summary['p05']:.6f} | "
        f"p95={summary['p95']:.6f} | min={summary['min']:.6f} | max={summary['max']:.6f}"
    )


def write_summary_csv(path, summaries):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def write_distances_csv(path, distance_groups):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["comparison", "sample_index", "nearest_train_distance"])
        for name, distances in distance_groups:
            for idx, distance in enumerate(distances):
                writer.writerow([name, idx, float(distance)])


def main():
    parser = argparse.ArgumentParser(
        description="Measure nearest-neighbour distances in VAE latent space for real and synthetic 3D NIfTI data."
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
    parser.add_argument("--random_state", type=int, default=42, help="split seed, matching train_vae.py")
    parser.add_argument("--batch_size", type=int, default=4, help="encoding batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--distance_chunk_size", type=int, default=32, help="query chunk size for torch.cdist")
    parser.add_argument("--distance_device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="device used for nearest-neighbour distance calculation")
    parser.add_argument("--summary_csv", type=str, default="", help="optional path to save summary statistics")
    parser.add_argument("--distances_csv", type=str, default="", help="optional path to save per-sample distances")
    args = parser.parse_args()

    encode_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Encoding with", "GPU" if encode_device.type == "cuda" else "CPU")

    vae = load_vae(args.model_dir, encode_device, args.checkpoint_name)
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

    print(f"Real split sizes: train={len(train_dataset)}, test={len(test_dataset)}")
    print(f"Synthetic size: {len(synthetic_dataset)}")

    train_features = encode_dataset(vae, train_dataset, encode_device, args.batch_size, args.num_workers, "train")
    test_features = encode_dataset(vae, test_dataset, encode_device, args.batch_size, args.num_workers, "test")
    synthetic_features = encode_dataset(
        vae,
        synthetic_dataset,
        encode_device,
        args.batch_size,
        args.num_workers,
        "synthetic",
    )

    print("Calculating test -> train nearest-neighbour distances")
    test_to_train = nearest_neighbor_distances(
        test_features,
        train_features,
        args.distance_chunk_size,
        args.distance_device,
    )
    print("Calculating synthetic -> train nearest-neighbour distances")
    synthetic_to_train = nearest_neighbor_distances(
        synthetic_features,
        train_features,
        args.distance_chunk_size,
        args.distance_device,
    )

    summaries = [
        summarize_distances("test_to_train", test_to_train),
        summarize_distances("synthetic_to_train", synthetic_to_train),
    ]
    for summary in summaries:
        print_summary(summary)

    if args.summary_csv:
        write_summary_csv(args.summary_csv, summaries)
        print(f"Saved summary CSV to {args.summary_csv}")
    if args.distances_csv:
        write_distances_csv(
            args.distances_csv,
            [("test_to_train", test_to_train), ("synthetic_to_train", synthetic_to_train)],
        )
        print(f"Saved per-sample distances CSV to {args.distances_csv}")


if __name__ == "__main__":
    main()

