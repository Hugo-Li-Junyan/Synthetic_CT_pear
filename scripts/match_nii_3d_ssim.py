"""Find the highest 3D SSIM match for each real NIfTI volume."""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def nii_files(folder):
    if not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")
    files = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))
        ),
        key=lambda path: path.name.lower(),
    )
    if not files:
        raise ValueError(f"No .nii or .nii.gz files found in: {folder}")
    return files


def load_volume(path):
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}: {path}")
    return np.asarray(image.dataobj, dtype=np.float32)


def volume_stats(volume, sigma, truncate):
    mean = gaussian_filter(volume, sigma=sigma, truncate=truncate, mode="reflect")
    variance = gaussian_filter(
        volume * volume, sigma=sigma, truncate=truncate, mode="reflect"
    ) - mean * mean
    return volume, mean, np.maximum(variance, 0.0)


def ssim_3d(left, right, data_range, sigma, truncate):
    x, mean_x, variance_x = left
    y, mean_y, variance_y = right
    if x.shape != y.shape:
        raise ValueError(f"Volume shape mismatch: {x.shape} versus {y.shape}")

    covariance = gaussian_filter(
        x * y, sigma=sigma, truncate=truncate, mode="reflect"
    ) - mean_x * mean_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
    denominator = (
        mean_x * mean_x + mean_y * mean_y + c1
    ) * (variance_x + variance_y + c2)
    score = numerator / np.maximum(denominator, np.finfo(np.float32).eps)

    border = int(truncate * sigma + 0.5)
    if min(score.shape) > 2 * border:
        score = score[border:-border, border:-border, border:-border]
    return float(np.mean(score, dtype=np.float64))


def find_global_range(paths):
    minimum, maximum = np.inf, -np.inf
    for path in paths:
        volume = load_volume(path)
        minimum = min(minimum, float(np.min(volume)))
        maximum = max(maximum, float(np.max(volume)))
    value = maximum - minimum
    if value <= 0:
        raise ValueError("All input volumes are constant; SSIM is undefined.")
    return value


def run(real_dir, interpolation_dir, output_csv, data_range, sigma, truncate):
    real_paths = nii_files(real_dir)
    interpolation_paths = nii_files(interpolation_dir)
    all_paths = real_paths + interpolation_paths
    if data_range is None:
        data_range = find_global_range(all_paths)
    if data_range <= 0:
        raise ValueError("--data-range must be greater than zero.")

    interpolation = []
    for path in interpolation_paths:
        interpolation.append((path, volume_stats(load_volume(path), sigma, truncate)))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("real_file", "best_interpolation_file", "ssim_3d")
        )
        writer.writeheader()
        for index, real_path in enumerate(real_paths, start=1):
            real = volume_stats(load_volume(real_path), sigma, truncate)
            best_path, best_score = None, -np.inf
            for candidate_path, candidate in interpolation:
                score = ssim_3d(real, candidate, data_range, sigma, truncate)
                if score > best_score:
                    best_path, best_score = candidate_path, score
            writer.writerow(
                {
                    "real_file": real_path.name,
                    "best_interpolation_file": best_path.name,
                    "ssim_3d": f"{best_score:.8f}",
                }
            )
            print(
                f"[{index}/{len(real_paths)}] {real_path.name} -> "
                f"{best_path.name}, SSIM={best_score:.8f}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare every 3D NIfTI file in 'real' with every file in "
            "'interpolation' and report the highest-SSIM match."
        )
    )
    parser.add_argument("real", type=Path, help="Folder containing real volumes.")
    parser.add_argument(
        "interpolation", type=Path, help="Folder containing interpolation volumes."
    )
    parser.add_argument("-o", "--output-csv", type=Path, default=Path("ssim_matches.csv"))
    parser.add_argument(
        "--data-range",
        type=float,
        default=None,
        help="SSIM intensity range; default is the global range of all input files.",
    )
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--truncate", type=float, default=3.5)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run(
            args.real,
            args.interpolation,
            args.output_csv,
            args.data_range,
            args.sigma,
            args.truncate,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
