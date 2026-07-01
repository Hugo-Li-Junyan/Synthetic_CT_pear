"""Register corresponding 3D NIfTI files across multiple folders."""

import argparse
from pathlib import Path
import SimpleITK as sitk


def nifti_names(folder):
    if not folder.is_dir():
        raise ValueError(f"Input folder does not exist: {folder}")
    return {
        p.name for p in folder.iterdir()
        if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))
    }


def validate_inputs(folders):
    if len(folders) < 2:
        raise ValueError("Provide at least two input folders.")
    expected = nifti_names(folders[0])
    if not expected:
        raise ValueError(f"No NIfTI files found in {folders[0]}")
    for folder in folders[1:]:
        found = nifti_names(folder)
        missing, extra = sorted(expected - found), sorted(found - expected)
        if missing or extra:
            raise ValueError(
                f"{folder}: missing={missing or 'none'}, extra={extra or 'none'}"
            )
    return sorted(expected, key=str.lower)


def registration_image_and_mask(image):
    """Normalize intensities and isolate foreground for metric evaluation."""
    image = sitk.Cast(image, sitk.sitkFloat32)
    normalized = sitk.RescaleIntensity(image, 0.0, 1.0)
    mask = sitk.OtsuThreshold(normalized, 0, 1, 128)
    mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2])
    return normalized, sitk.Cast(mask, sitk.sitkUInt8)


def register(fixed, moving, model):
    fixed_registration, fixed_mask = registration_image_and_mask(fixed)
    moving_registration, moving_mask = registration_image_and_mask(moving)

    initial = sitk.Euler3DTransform() if model == "rigid" else sitk.AffineTransform(3)
    initial = sitk.CenteredTransformInitializer(
        fixed_registration,
        moving_registration,
        initial,
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    method = sitk.ImageRegistrationMethod()
    # Mutual information remains stable when corresponding scans differ in contrast.
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    method.SetMetricFixedMask(fixed_mask)
    method.SetMetricMovingMask(moving_mask)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.10, seed=42)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-3,
        numberOfIterations=150,
        relaxationFactor=0.6,
        gradientMagnitudeTolerance=1e-6,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)

    transform = method.Execute(fixed_registration, moving_registration)
    print(
        f"    metric={method.GetMetricValue():.6g}, "
        f"iterations={method.GetOptimizerIteration()}, "
        f"stop={method.GetOptimizerStopConditionDescription()}"
    )
    return transform

def output_name(name):
    return name[:-7] + ".nii" if name.lower().endswith(".nii.gz") else name


def run(folders, output_dir, model, interpolation):
    filenames = validate_inputs(folders)
    output_folders = [output_dir / folder.name for folder in folders]
    if len(set(output_folders)) != len(output_folders):
        raise ValueError("Input folders must have unique names.")
    for folder in output_folders:
        folder.mkdir(parents=True, exist_ok=True)
    transform_dir = output_dir / "transforms"
    transform_dir.mkdir(parents=True, exist_ok=True)
    interpolator = (
        sitk.sitkLinear if interpolation == "linear"
        else sitk.sitkNearestNeighbor
    )
    total, done = len(filenames) * (len(folders) - 1), 0

    for filename in filenames:
        fixed = sitk.ReadImage(str(folders[0] / filename))
        name = output_name(filename)
        sitk.WriteImage(
            fixed, str(output_folders[0] / name), useCompression=False
        )
        for index, moving_folder in enumerate(folders[1:], start=1):
            moving = sitk.ReadImage(str(moving_folder / filename))
            transform = register(fixed, moving, model)
            minimum_filter = sitk.MinimumMaximumImageFilter()
            minimum_filter.Execute(moving)
            fill_value = float(minimum_filter.GetMinimum())
            registered = sitk.Resample(
                moving,
                fixed,
                transform,
                interpolator,
                fill_value,
                moving.GetPixelID(),
            )
            destination = output_folders[index] / name
            sitk.WriteImage(registered, str(destination), useCompression=False)
            transform_path = transform_dir / (
                f"{moving_folder.name}__{Path(name).stem}.tfm"
            )
            sitk.WriteTransform(transform, str(transform_path))
            done += 1
            print(f"[{done}/{total}] {moving_folder.name}/{filename} -> {destination}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Register matching NIfTI files across folders. Files in the first "
            "folder are the fixed references."
        )
    )
    parser.add_argument(
        "input_dirs", nargs="+", type=Path,
        help="Input folders; the first folder is the reference.",
    )
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    parser.add_argument(
        "--transform", choices=("rigid", "affine"), default="rigid"
    )
    parser.add_argument(
        "--interpolation", choices=("linear", "nearest"), default="linear"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run(args.input_dirs, args.output_dir, args.transform, args.interpolation)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()





