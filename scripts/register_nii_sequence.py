"""Register an ordered sequence of 3D NIfTI images with SimpleITK.

Examples
--------
python scripts/register_nii_sequence.py scan_01.nii scan_02.nii scan_03.nii -o registered
python scripts/register_nii_sequence.py --input-dir scans --reference first --transform affine -o registered
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover - depends on the user's environment
    raise SystemExit(
        "SimpleITK is required. Install it with `pip install SimpleITK`."
    ) from exc


def nifti_stem(path: Path) -> str:
    """Return a filename without either the .nii or .nii.gz suffix."""
    name = path.name
    return name[:-7] if name.lower().endswith(".nii.gz") else path.stem


def discover_images(files: Sequence[Path], input_dir: Path | None) -> list[Path]:
    if files and input_dir is not None:
        raise ValueError("Give either image paths or --input-dir, not both.")
    if input_dir is not None:
        images = sorted(
            (p for p in input_dir.iterdir() if p.name.lower().endswith((".nii", ".nii.gz"))),
            key=lambda p: p.name.lower(),
        )
    else:
        images = list(files)
    if len(images) < 2:
        raise ValueError("At least two NIfTI images are required.")
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input image(s): " + ", ".join(missing))
    return images


def initial_transform(fixed: sitk.Image, moving: sitk.Image, kind: str) -> sitk.Transform:
    if kind == "rigid":
        transform = sitk.Euler3DTransform()
    else:
        transform = sitk.AffineTransform(3)
    return sitk.CenteredTransformInitializer(
        fixed,
        moving,
        transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )


def register(fixed: sitk.Image, moving: sitk.Image, kind: str) -> sitk.Transform:
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.20, seed=42)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=300,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=15,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial_transform(fixed, moving, kind), inPlace=False)
    return method.Execute(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
    )


def run(
    images: Sequence[Path],
    output_dir: Path,
    reference: str,
    transform_kind: str,
    interpolation: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    first = sitk.ReadImage(str(images[0]))
    fixed = first
    first_output = output_dir / f"{nifti_stem(images[0])}_registered.nii"
    sitk.WriteImage(first, str(first_output), useCompression=False)

    interpolators = {"linear": sitk.sitkLinear, "nearest": sitk.sitkNearestNeighbor}
    for index, path in enumerate(images[1:], start=1):
        moving = sitk.ReadImage(str(path))
        transform = register(fixed, moving, transform_kind)
        # All outputs use the first image's grid, making the resulting sequence stackable.
        registered = sitk.Resample(
            moving,
            first,
            transform,
            interpolators[interpolation],
            0.0,
            moving.GetPixelID(),
        )
        stem = nifti_stem(path)
        image_output = output_dir / f"{stem}_registered.nii"
        transform_output = output_dir / f"{stem}_to_{nifti_stem(images[0] if reference == 'first' else images[index - 1])}.tfm"
        sitk.WriteImage(registered, str(image_output), useCompression=False)
        sitk.WriteTransform(transform, str(transform_output))
        print(f"[{index + 1}/{len(images)}] {path.name} -> {image_output.name}")
        if reference == "previous":
            fixed = registered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register a sequence of 3D .nii/.nii.gz images into the first image's grid."
    )
    parser.add_argument("images", nargs="*", type=Path, help="Input images in sequence order.")
    parser.add_argument("--input-dir", type=Path, help="Directory of images (sorted by filename).")
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference", choices=("first", "previous"), default="first",
        help="Register each image to the first image or the preceding aligned image (default: first).",
    )
    parser.add_argument(
        "--transform", choices=("rigid", "affine"), default="rigid",
        help="Transformation model (default: rigid).",
    )
    parser.add_argument(
        "--interpolation", choices=("linear", "nearest"), default="linear",
        help="Use nearest for label maps; linear for intensity images (default: linear).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        images = discover_images(args.images, args.input_dir)
        run(images, args.output_dir, args.reference, args.transform, args.interpolation)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()

