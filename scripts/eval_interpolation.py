r"""Evaluate interpolation_line.py outputs against five target NIfTI folders.

For every .nii/.nii.gz filename shared by the healthy and defective folders,
this script runs interpolation_line.line_interpolate(..., num_steps=5).  It then
compares generated 0.nii..4.nii against files with the same name in five target
folders and writes MAE + 3D SSIM results to a CSV file.

Example:
    python scripts/eval_interpolation.py ^
        --model_dir path\to\model ^
        --healthy_path path\to\healthy ^
        --defective_path path\to\defective ^
        --target_paths path\to\month0 path\to\month1 path\to\month2 path\to\month3 path\to\month4 ^
        --output_csv interpolation_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interpolation_line import line_interpolate  # noqa: E402


NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti(path: Path) -> bool:
    return path.is_file() and any(str(path).lower().endswith(suffix) for suffix in NIFTI_SUFFIXES)


def nifti_key(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return path.stem


def safe_output_name(path: Path) -> str:
    return nifti_key(path).replace(" ", "_")


def list_nifti_by_name(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    files = {}
    for path in sorted(folder.iterdir()):
        if is_nifti(path):
            files[path.name] = path
    return files


def load_volume(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)


def normalize_minmax(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    min_value = float(np.min(volume))
    max_value = float(np.max(volume))
    value_range = max_value - min_value
    if value_range < eps:
        return np.zeros_like(volume, dtype=np.float32)
    return (volume - min_value) / value_range


def mae_3d(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def ssim_3d(
    left: np.ndarray,
    right: np.ndarray,
    data_range: float,
    sigma: float = 1.5,
    truncate: float = 3.5,
    eps: float = 1e-12,
) -> float:
    """Compute a volumetric SSIM score using a Gaussian local window."""
    left = left.astype(np.float32, copy=False)
    right = right.astype(np.float32, copy=False)

    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch for SSIM: {left.shape} vs {right.shape}")

    data_range = max(float(data_range), eps)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_left = gaussian_filter(left, sigma=sigma, truncate=truncate, mode="reflect")
    mu_right = gaussian_filter(right, sigma=sigma, truncate=truncate, mode="reflect")

    mu_left_sq = mu_left * mu_left
    mu_right_sq = mu_right * mu_right
    mu_left_right = mu_left * mu_right

    sigma_left_sq = gaussian_filter(left * left, sigma=sigma, truncate=truncate, mode="reflect") - mu_left_sq
    sigma_right_sq = gaussian_filter(right * right, sigma=sigma, truncate=truncate, mode="reflect") - mu_right_sq
    sigma_left_right = gaussian_filter(left * right, sigma=sigma, truncate=truncate, mode="reflect") - mu_left_right

    numerator = (2.0 * mu_left_right + c1) * (2.0 * sigma_left_right + c2)
    denominator = (mu_left_sq + mu_right_sq + c1) * (sigma_left_sq + sigma_right_sq + c2)
    score_map = numerator / np.maximum(denominator, eps)
    return float(np.mean(score_map))


def prepare_for_metrics(
    generated: np.ndarray,
    target: np.ndarray,
    normalize_metrics: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    if generated.shape != target.shape:
        raise ValueError(f"Shape mismatch: generated {generated.shape}, target {target.shape}")

    if normalize_metrics:
        generated = normalize_minmax(generated)
        target = normalize_minmax(target)
        return generated, target, 1.0

    min_value = min(float(np.min(generated)), float(np.min(target)))
    max_value = max(float(np.max(generated)), float(np.max(target)))
    return generated, target, max_value - min_value


def evaluate_one_pair(
    *,
    filename: str,
    healthy_file: Path,
    defective_file: Path,
    target_folders: list[Path],
    model_dir: Path,
    interpolation_root: Path,
    num_steps: int,
    normalize_metrics: bool,
    diffusion: bool,
    show_latent: bool,
    interpolation: str,
    ssim_sigma: float,
    ssim_truncate: float,
    overwrite: bool,
) -> list[dict[str, object]]:
    pair_output_dir = interpolation_root / safe_output_name(healthy_file)
    expected_outputs = [pair_output_dir / f"{step}.nii" for step in range(num_steps)]

    if overwrite or not all(path.exists() for path in expected_outputs):
        line_interpolate(
            str(model_dir),
            str(pair_output_dir),
            str(healthy_file),
            str(defective_file),
            num_steps=num_steps,
            show_latent=show_latent,
            diffusion=diffusion,
            interpolation=interpolation,
        )

    rows = []
    for step, target_folder in enumerate(target_folders):
        generated_file = pair_output_dir / f"{step}.nii"
        target_file = target_folder / filename
        if not generated_file.exists():
            raise FileNotFoundError(f"Expected interpolation output was not created: {generated_file}")
        if not target_file.exists():
            raise FileNotFoundError(f"Missing target file for step {step}: {target_file}")

        generated = load_volume(generated_file)
        target = load_volume(target_file)
        generated_for_metric, target_for_metric, data_range = prepare_for_metrics(
            generated,
            target,
            normalize_metrics=normalize_metrics,
        )

        rows.append(
            {
                "file": filename,
                "step": step,
                "healthy_file": str(healthy_file),
                "defective_file": str(defective_file),
                "target_folder": str(target_folder),
                "target_file": str(target_file),
                "interpolation_file": str(generated_file),
                "mae": mae_3d(generated_for_metric, target_for_metric),
                "ssim_3d": ssim_3d(
                    generated_for_metric,
                    target_for_metric,
                    data_range=data_range,
                    sigma=ssim_sigma,
                    truncate=ssim_truncate,
                ),
                "metrics_normalized_minmax": normalize_metrics,
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "step",
        "healthy_file",
        "defective_file",
        "target_folder",
        "target_file",
        "interpolation_file",
        "mae",
        "ssim_3d",
        "metrics_normalized_minmax",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run interpolation_line.py over matching healthy/defective NIfTI pairs, "
            "then evaluate 0.nii..4.nii against five target folders."
        )
    )
    parser.add_argument("--model_dir", required=True, type=Path, help="Model directory passed to interpolation_line.py")
    parser.add_argument("--healthy_path", required=True, type=Path, help="Folder containing healthy/source .nii files")
    parser.add_argument("--defective_path", required=True, type=Path, help="Folder containing defective/end .nii files")
    parser.add_argument(
        "--target_paths",
        required=True,
        nargs=5,
        type=Path,
        metavar=("STEP0", "STEP1", "STEP2", "STEP3", "STEP4"),
        help="Five folders whose same-named files are compared with generated 0.nii..4.nii",
    )
    parser.add_argument("--output_csv", required=True, type=Path, help="CSV file for all MAE/SSIM results")
    parser.add_argument(
        "--interpolation_root",
        type=Path,
        default=Path("eval_interpolation_outputs"),
        help="Folder where per-file interpolation outputs are saved",
    )
    parser.add_argument("--num_steps", type=int, default=5, help="Number of interpolation steps; must match target_paths")
    parser.add_argument("--diffusion", action="store_true", help="Pass diffusion=True to interpolation_line.line_interpolate")
    parser.add_argument("--show_latent", action="store_true", help="Evaluate latent outputs instead of decoded images")
    parser.add_argument("--interpolation", choices=("slerp", "linear"), default="slerp", help="Latent interpolation method")
    parser.add_argument("--ssim_sigma", type=float, default=1.5, help="Gaussian sigma for 3D SSIM")
    parser.add_argument("--ssim_truncate", type=float, default=3.5, help="Gaussian truncate radius for 3D SSIM")
    parser.add_argument(
        "--raw_metrics",
        action="store_true",
        help="Use raw voxel values for MAE/SSIM instead of per-volume min-max normalized values",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate interpolation outputs even if 0.nii..4.nii already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_steps != len(args.target_paths):
        raise ValueError(f"--num_steps ({args.num_steps}) must equal the number of --target_paths ({len(args.target_paths)})")

    healthy_files = list_nifti_by_name(args.healthy_path)
    defective_files = list_nifti_by_name(args.defective_path)
    shared_names = sorted(set(healthy_files) & set(defective_files))
    if not shared_names:
        raise ValueError(f"No shared .nii/.nii.gz filenames found in {args.healthy_path} and {args.defective_path}")

    missing_from_defective = sorted(set(healthy_files) - set(defective_files))
    missing_from_healthy = sorted(set(defective_files) - set(healthy_files))
    if missing_from_defective:
        print(f"Warning: {len(missing_from_defective)} healthy files have no defective match; skipping them.")
    if missing_from_healthy:
        print(f"Warning: {len(missing_from_healthy)} defective files have no healthy match; skipping them.")

    target_folders = [Path(path) for path in args.target_paths]
    for folder in target_folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"Target folder does not exist: {folder}")

    all_rows = []
    for index, filename in enumerate(shared_names, start=1):
        print(f"[{index}/{len(shared_names)}] Running interpolation and metrics for {filename}")
        all_rows.extend(
            evaluate_one_pair(
                filename=filename,
                healthy_file=healthy_files[filename],
                defective_file=defective_files[filename],
                target_folders=target_folders,
                model_dir=args.model_dir,
                interpolation_root=args.interpolation_root,
                num_steps=args.num_steps,
                normalize_metrics=not args.raw_metrics,
                diffusion=args.diffusion,
                show_latent=args.show_latent,
                interpolation=args.interpolation,
                ssim_sigma=args.ssim_sigma,
                ssim_truncate=args.ssim_truncate,
                overwrite=args.overwrite,
            )
        )

    write_csv(args.output_csv, all_rows)
    print(f"Saved {len(all_rows)} metric rows to {args.output_csv}")


if __name__ == "__main__":
    main()
