import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from component.dataset import CsvVolumeDataset
from train_clf_3d import build_transform


def build_gamma_transform(log_gamma):
    import torchio as tio

    return tio.RandomGamma(log_gamma=log_gamma)


def tensor_to_volume(tensor):
    return tensor.squeeze(0).detach().cpu().numpy()


def select_sample(dataset, sample_index=None, filename=None):
    if filename:
        target = Path(filename).name
        for index, (path, _label) in enumerate(dataset.samples):
            if Path(path).name == target:
                return index
        raise ValueError(f"Could not find filename in CSV samples: {filename}")
    return sample_index


def plane_slices(volume, axial_slice, coronal_slice, sagittal_slice):
    return [
        volume[:, :, axial_slice].T,
        volume[:, coronal_slice, :].T,
        volume[sagittal_slice, :, :].T,
    ]


def plot_views(volumes, titles, output_path, axial_slice, coronal_slice, sagittal_slice):
    row_names = [
        f"Axial z={axial_slice}",
        f"Coronal y={coronal_slice}",
        f"Sagittal x={sagittal_slice}",
    ]

    fig, axes = plt.subplots(nrows=3, ncols=len(volumes), figsize=(3 * len(volumes), 8))
    if len(volumes) == 1:
        axes = axes[:, None]

    for col_idx, (volume, title) in enumerate(zip(volumes, titles)):
        for row_idx, image_slice in enumerate(plane_slices(volume, axial_slice, coronal_slice, sagittal_slice)):
            ax = axes[row_idx, col_idx]
            ax.imshow(image_slice, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(row_names[row_idx], fontsize=10)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D classifier augmentations on axial, coronal, and sagittal slices."
    )
    parser.add_argument("--train_dir", type=str, required=True, help="folder containing labeled NIfTI volumes")
    parser.add_argument("--label_csv", type=str, required=True, help="CSV with filename and label columns")
    parser.add_argument("--filename_column", type=str, default="filename", help="CSV filename column")
    parser.add_argument("--label_column", type=str, default="label", help="CSV label column")
    parser.add_argument("--sample_index", type=int, default=0, help="sample index to visualize")
    parser.add_argument("--filename", type=str, default="", help="optional filename to visualize instead of index")
    parser.add_argument("--num_augments", type=int, default=4, help="number of random augmented copies to show")
    parser.add_argument("--output_path", type=str, default="clf_augmentation_preview.png", help="output PNG path")
    parser.add_argument("--axial_slice", type=int, default=64, help="axial slice index")
    parser.add_argument("--coronal_slice", type=int, default=64, help="coronal slice index")
    parser.add_argument("--sagittal_slice", type=int, default=64, help="sagittal slice index")
    parser.add_argument("--gamma_only", action="store_true", help="visualize only RandomGamma, without flip/affine")
    parser.add_argument("--log_gamma_min", type=float, default=-0.1, help="minimum log_gamma for gamma-only mode")
    parser.add_argument("--log_gamma_max", type=float, default=0.1, help="maximum log_gamma for gamma-only mode")
    parser.add_argument("--seed", type=int, default=None, help="optional torch seed for repeatable augmentation")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    for name, index in {
        "axial_slice": args.axial_slice,
        "coronal_slice": args.coronal_slice,
        "sagittal_slice": args.sagittal_slice,
    }.items():
        if index < 0 or index >= 128:
            raise ValueError(f"{name} must be between 0 and 127, got {index}")

    base_dataset = CsvVolumeDataset(
        args.train_dir,
        args.label_csv,
        filename_column=args.filename_column,
        label_column=args.label_column,
    )
    sample_index = select_sample(base_dataset, args.sample_index, args.filename)
    original_tensor, label = base_dataset[sample_index]

    transform = (
        build_gamma_transform((args.log_gamma_min, args.log_gamma_max))
        if args.gamma_only
        else build_transform(True)
    )

    volumes = [tensor_to_volume(original_tensor)]
    titles = [f"Original\nlabel={int(label.item())}"]
    for idx in range(args.num_augments):
        augmented_tensor = transform(original_tensor.clone())
        volumes.append(tensor_to_volume(augmented_tensor))
        titles.append(f"Aug {idx + 1}")

    plot_views(
        volumes,
        titles,
        args.output_path,
        args.axial_slice,
        args.coronal_slice,
        args.sagittal_slice,
    )
    sample_path, _ = base_dataset.samples[sample_index]
    print(f"Sample: {sample_path}")
    print(f"Saved augmentation preview to {args.output_path}")


if __name__ == "__main__":
    main()
