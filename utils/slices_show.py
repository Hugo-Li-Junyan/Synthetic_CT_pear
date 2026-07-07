from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def foreground_mask_from_min_background(volume: np.ndarray) -> np.ndarray:
    """Segment foreground using the project rule: background equals image minimum."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros(volume.shape, dtype=bool)
    background_value = float(np.min(finite))
    return np.isfinite(volume) & (volume > background_value)


def downsample_volume_and_mask(volume: np.ndarray, mask: np.ndarray, max_size: int = 96) -> tuple[np.ndarray, np.ndarray, int]:
    step = max(1, int(np.ceil(max(volume.shape) / max_size)))
    return volume[::step, ::step, ::step], mask[::step, ::step, ::step], step


def set_3d_axes_clean(ax, shape: tuple[int, int, int]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[1])
    ax.set_zlim(0, shape[2])
    ax.set_box_aspect(shape)
    ax.view_init(elev=18, azim=-55)
    ax.set_facecolor("black")
    ax.grid(False)


def add_cut_face(ax, volume: np.ndarray, mask: np.ndarray, x_index: int, step: int) -> None:
    """Draw the exposed middle cut face as a grayscale texture."""
    if x_index < 0 or x_index >= volume.shape[0]:
        return

    cut_mask = mask[x_index, :, :]
    if not np.any(cut_mask):
        return

    cut_values = volume[x_index, :, :]
    vmin, vmax = robust_limits(cut_values[cut_mask])
    normalized = np.clip((cut_values - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)

    colors = plt.cm.gray(normalized)
    colors[..., 3] = np.where(cut_mask, 0.98, 0.0)

    y = np.arange(volume.shape[1]) * step
    z = np.arange(volume.shape[2]) * step
    yy, zz = np.meshgrid(y, z, indexing="ij")
    xx = np.full_like(yy, fill_value=x_index * step, dtype=np.float32)

    ax.plot_surface(
        xx,
        yy,
        zz,
        rstride=1,
        cstride=1,
        facecolors=colors,
        shade=False,
        antialiased=False,
        linewidth=0,
    )


def add_half_cut_3d_panel(ax, volume: np.ndarray) -> None:
    """Render min-background segmented foreground, cut in half, in soft gray."""
    full_mask = foreground_mask_from_min_background(volume)
    small_volume, small_mask, step = downsample_volume_and_mask(volume, full_mask)

    x_mid = small_mask.shape[0] // 2
    half_mask = small_mask.copy()
    half_mask[:x_mid, :, :] = False

    world_shape = tuple(size * step for size in small_mask.shape)
    set_3d_axes_clean(ax, world_shape)
    if not np.any(half_mask):
        return

    add_cut_face(ax, small_volume, small_mask, x_mid, step)

    try:
        from skimage import measure

        padded = np.pad(half_mask.astype(np.float32), 1, mode="constant", constant_values=0)
        verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=(step, step, step))
        verts -= step

        mesh = Poly3DCollection(verts[faces], linewidths=0.0, alpha=0.62)
        mesh.set_facecolor((0.78, 0.78, 0.74, 1.0))
        mesh.set_edgecolor("none")
        ax.add_collection3d(mesh)
    except Exception:
        points = np.argwhere(half_mask)
        if points.size == 0:
            return
        if len(points) > 6000:
            rng = np.random.default_rng(0)
            points = points[rng.choice(len(points), size=6000, replace=False)]
        points = points * step
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.18, c=[(0.78, 0.78, 0.74)], alpha=0.38)


def show_random_volumes_grid(
    folder_path: str | Path,
    output_path: str | Path = "random_generation.png",
    seed: int | None = None,
    num_volumes: int = 16,
    grid_rows: int = 4,
    grid_cols: int = 4,
    dpi: int = 200,
) -> list[Path]:
    """Plot random NIfTI volumes as a 4x4 grid of middle slices plus a 3D cutaway.

    Each selected volume occupies one outer grid cell. Inside that cell:
      top-left     = axial XY middle slice
      top-right    = coronal XZ middle slice
      bottom-left  = sagittal YZ middle slice
      bottom-right = gray 3D half-cut foreground rendering

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
            (0, 0, slices["axial XY"]),
            (0, 1, slices["coronal XZ"]),
            (1, 0, slices["sagittal YZ"]),
        ]

        for panel_row, panel_col, image in panels:
            ax = fig.add_subplot(inner[panel_row, panel_col])
            ax.imshow(image.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            ax.axis("off")

        ax_3d = fig.add_subplot(inner[1, 1], projection="3d")
        add_half_cut_3d_panel(ax_3d, volume)

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