from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import gridspec
from scipy import ndimage as ndi


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
    """Segment foreground using the rule: background equals image minimum."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros(volume.shape, dtype=bool)
    background_value = float(np.min(finite))
    return np.isfinite(volume) & (volume > background_value)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Remove noisy foreground islands and keep only the main object."""
    if not np.any(mask):
        return mask

    labels, count = ndi.label(mask)
    if count <= 1:
        return mask

    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest_label = int(np.argmax(component_sizes))
    return labels == largest_label


def clean_foreground_mask(volume: np.ndarray) -> np.ndarray:
    mask = foreground_mask_from_min_background(volume)
    mask = ndi.binary_fill_holes(mask)
    mask = largest_connected_component(mask)
    return mask.astype(bool)


def downsample_volume_and_mask(volume: np.ndarray, mask: np.ndarray, max_size: int = 96) -> tuple[np.ndarray, np.ndarray, int]:
    step = max(1, int(np.ceil(max(volume.shape) / max_size)))
    return volume[::step, ::step, ::step], mask[::step, ::step, ::step], step


def render_cutaway_with_pyvista(volume: np.ndarray, image_size: int = 480) -> np.ndarray:
    """Render the earlier smooth open-mesh cutaway using PyVista/VTK."""
    import pyvista as pv

    full_mask = clean_foreground_mask(volume)
    _, small_mask, step = downsample_volume_and_mask(volume, full_mask)
    if not np.any(small_mask):
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

    pv.global_theme.window_size = [image_size, image_size]
    pv.global_theme.background = "black"
    pv.global_theme.smooth_shading = True

    smooth_mask = ndi.gaussian_filter(small_mask.astype(np.float32), sigma=1.0)

    grid = pv.ImageData()
    grid.dimensions = smooth_mask.shape
    grid.spacing = (step, step, step)
    grid.point_data["foreground"] = smooth_mask.ravel(order="F")
    surface = grid.contour([0.5], scalars="foreground")
    if surface.n_points == 0 or surface.n_cells == 0:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

    try:
        surface = surface.smooth_taubin(n_iter=45, pass_band=0.06)
    except Exception:
        surface = surface.smooth(n_iter=35, relaxation_factor=0.08)

    x_cut = 0.5 * step * (small_mask.shape[0] - 1)
    cell_centers = surface.cell_centers().points
    keep_cells = np.flatnonzero(cell_centers[:, 0] >= x_cut)
    open_surface = surface.extract_cells(keep_cells).extract_surface(algorithm="dataset_surface")
    if open_surface.n_points == 0 or open_surface.n_cells == 0:
        open_surface = surface

    bounds = np.array(open_surface.bounds, dtype=np.float32)
    center = (
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    )
    max_extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

    plotter = pv.Plotter(off_screen=True, window_size=(image_size, image_size), border=False)
    plotter.set_background("black")
    plotter.add_mesh(
        open_surface,
        color=(0.74, 0.74, 0.70),
        opacity=0.92,
        smooth_shading=True,
        specular=0.18,
        roughness=0.70,
    )

    # View from a 45-degree diagonal in the X-Y plane while keeping the 50% cut.
    diagonal_offset = 2.2 * max_extent / np.sqrt(2.0)
    plotter.camera_position = [
        (center[0] - diagonal_offset, center[1] - diagonal_offset, center[2] + 0.08 * max_extent),
        center,
        (0, 0, 1),
    ]
    plotter.camera.zoom(1.18)
    plotter.enable_parallel_projection()

    try:
        image = plotter.screenshot(return_img=True)
    finally:
        plotter.close()

    return np.asarray(image)[..., :3]


def render_cutaway_fallback(volume: np.ndarray, image_size: int = 480) -> np.ndarray:
    """Fallback when PyVista is unavailable: show only the half-cut foreground projection."""
    mask = clean_foreground_mask(volume)
    x_mid = mask.shape[0] // 2
    half_mask = mask.copy()
    half_mask[:x_mid, :, :] = False
    projection = np.max(half_mask, axis=0).astype(np.float32)
    rgb = (plt.cm.gray(0.68 * projection.T)[..., :3] * 255).astype(np.uint8)
    return rgb


def render_cutaway_image(volume: np.ndarray, image_size: int = 480) -> np.ndarray:
    try:
        return render_cutaway_with_pyvista(volume, image_size=image_size)
    except Exception as exc:
        print(f"Warning: PyVista rendering failed; using fallback half-cut projection. Reason: {exc}")
        return render_cutaway_fallback(volume, image_size=image_size)


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
      bottom-right = PyVista-rendered gray 3D half-cut foreground surface

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

        ax_3d = fig.add_subplot(inner[1, 1])
        ax_3d.imshow(render_cutaway_image(volume), origin="upper")
        ax_3d.axis("off")

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