from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import gridspec


NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti(path: Path) -> bool:
    return path.is_file() and any(str(path).lower().endswith(suffix) for suffix in NIFTI_SUFFIXES)


def list_nifti_files(folder_path: str | Path) -> list[Path]:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    files = sorted(path for path in folder.iterdir() if is_nifti(path))
    if not files:
        raise ValueError(f"No .nii or .nii.gz files found in {folder}")
    return files


def load_volume(path: Path) -> np.ndarray:
    volume = nib.load(str(path)).get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {volume.shape}: {path}")
    return np.asarray(volume, dtype=np.float32)


def middle_slices(volume: np.ndarray) -> dict[str, np.ndarray]:
    """Return anatomical middle slices for a volume indexed as X, Y, Z.

    axial:    XY plane at middle Z
    coronal:  XZ plane at middle Y
    sagittal: YZ plane at middle X
    """
    x_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    z_mid = volume.shape[2] // 2
    return {
        "axial XY": volume[:, :, z_mid],
        "coronal XZ": volume[:, y_mid, :],
        "sagittal YZ": volume[x_mid, :, :],
    }


def robust_limits(volume: np.ndarray) -> tuple[float, float]:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [1, 99])
    if vmax <= vmin:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def show_random_volumes_grid(
    folder_path: str | Path,
    output_path: str | Path = "random_generation.png",
    seed: int | None = None,
    num_volumes: int = 16,
    grid_rows: int = 4,
    grid_cols: int = 4,
    dpi: int = 200,
) -> list[Path]:
    """Plot random NIfTI volumes as a 4x4 grid of axial/coronal/sagittal slices.

    Each selected volume occupies one outer grid cell. Inside that cell:
      top-left     = axial XY middle slice
      top-right    = coronal XZ middle slice
      bottom-left  = sagittal YZ middle slice
      bottom-right = blank

    Returns the selected file paths, which is useful for reproducibility logs/tests.
    """
    if grid_rows * grid_cols != num_volumes:
        raise ValueError("grid_rows * grid_cols must equal num_volumes")

    nifti_files = list_nifti_files(folder_path)
    if len(nifti_files) < num_volumes:
        raise ValueError(f"Need at least {num_volumes} NIfTI files, found {len(nifti_files)} in {folder_path}")

    rng = random.Random(seed)
    selected_files = rng.sample(nifti_files, num_volumes)

    fig = plt.figure(figsize=(grid_cols * 3.2, grid_rows * 3.2), constrained_layout=False)
    outer = gridspec.GridSpec(
        grid_rows,
        grid_cols,
        figure=fig,
        wspace=0.08,
        hspace=0.12,
        top=0.98,
        bottom=0.02,
        left=0.02,
        right=0.98,
    )

    for index, volume_path in enumerate(selected_files):
        row = index // grid_cols
        col = index % grid_cols
        inner = outer[row, col].subgridspec(2, 2, wspace=0.01, hspace=0.01)

        volume = load_volume(volume_path)
        vmin, vmax = robust_limits(volume)
        slices = middle_slices(volume)

        panels = [
            (0, 0, "axial XY", slices["axial XY"]),
            (0, 1, "coronal XZ", slices["coronal XZ"]),
            (1, 0, "sagittal YZ", slices["sagittal YZ"]),
        ]

        for panel_row, panel_col, title, image in panels:
            ax = fig.add_subplot(inner[panel_row, panel_col])
            ax.imshow(image.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=6, pad=1)
            ax.axis("off")

        blank_ax = fig.add_subplot(inner[1, 1])
        blank_ax.axis("off")
        blank_ax.text(
            0.5,
            0.5,
            volume_path.name,
            ha="center",
            va="center",
            fontsize=5,
            wrap=True,
            transform=blank_ax.transAxes,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)
    return selected_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show random middle slices from 3D NIfTI volumes.")
    parser.add_argument("folder_path", type=Path, help="Folder containing 3D .nii or .nii.gz volumes")
    parser.add_argument("--output", type=Path, default=Path("random_generation.png"), help="Output PNG path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible volume selection")
    parser.add_argument("--num_volumes", type=int, default=16, help="Number of volumes to select; default is 16")
    parser.add_argument("--dpi", type=int, default=200, help="Saved figure DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_files = show_random_volumes_grid(
        args.folder_path,
        output_path=args.output,
        seed=args.seed,
        num_volumes=args.num_volumes,
        grid_rows=4,
        grid_cols=4,
        dpi=args.dpi,
    )
    print(f"Saved slice grid to {args.output}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print("Selected volumes:")
    for path in selected_files:
        print(path)


if __name__ == "__main__":
    main()