import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from component.vae import VAE
from utils.volumes import normalize_minmax


def load_vae_from_best(model_dir, device):
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / "best.pth"
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


def load_volume(path):
    image = nib.load(str(path))
    volume = normalize_minmax(image.get_fdata())
    if volume.shape != (128, 128, 128):
        raise ValueError(f"Expected a 128x128x128 volume, got {volume.shape} from {path}")
    return volume.astype(np.float32)


def reconstruct(vae, volume, device):
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstruction, _, _ = vae(tensor)
    return reconstruction.squeeze(0).squeeze(0).cpu().numpy()


def get_slices(volume, axial_slice, coronal_slice, sagittal_slice):
    return [
        volume[:, :, axial_slice].T,
        volume[:, coronal_slice, :].T,
        volume[sagittal_slice, :, :].T,
    ]


def plot_reconstructions(original, reconstructions, model_names, output_path,
                         axial_slice, coronal_slice, sagittal_slice):
    columns = ["Original"] + model_names
    row_names = [
        f"Axial z={axial_slice}",
        f"Coronal y={coronal_slice}",
        f"Sagittal x={sagittal_slice}",
    ]
    volumes = [original] + reconstructions

    fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(24, 9))
    for col_idx, (column, volume) in enumerate(zip(columns, volumes)):
        for row_idx, image_slice in enumerate(get_slices(volume, axial_slice, coronal_slice, sagittal_slice)):
            ax = axes[row_idx, col_idx]
            ax.imshow(image_slice, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(column, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(row_names[row_idx], fontsize=10)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare one NIfTI volume against reconstructions from seven VAE best.pth checkpoints."
    )
    parser.add_argument("--nii_path", type=str, required=True, help="input 128x128x128 .nii or .nii.gz file")
    parser.add_argument("--model_dirs", type=str, nargs=7, required=True,
                        help="seven model folders, each containing best.pth and vae_hyperparameter.json")
    parser.add_argument("--output_path", type=str, default="vae_reconstruction_comparison.png",
                        help="path for the output plot")
    parser.add_argument("--axial_slice", type=int, default=64, help="axial slice index")
    parser.add_argument("--coronal_slice", type=int, default=64, help="coronal slice index")
    parser.add_argument("--sagittal_slice", type=int, default=64, help="sagittal slice index")
    args = parser.parse_args()

    for name, index in {
        "axial_slice": args.axial_slice,
        "coronal_slice": args.coronal_slice,
        "sagittal_slice": args.sagittal_slice,
    }.items():
        if index < 0 or index >= 128:
            raise ValueError(f"{name} must be between 0 and 127, got {index}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", "GPU" if device.type == "cuda" else "CPU")

    original = load_volume(args.nii_path)
    reconstructions = []
    model_names = []
    for model_dir in args.model_dirs:
        model_dir_path = Path(model_dir)
        print(f"Reconstructing with {model_dir_path}")
        vae = load_vae_from_best(model_dir_path, device)
        reconstructions.append(reconstruct(vae, original, device))
        model_names.append(model_dir_path.name)

    plot_reconstructions(
        original,
        reconstructions,
        model_names,
        args.output_path,
        args.axial_slice,
        args.coronal_slice,
        args.sagittal_slice,
    )
    print(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()

