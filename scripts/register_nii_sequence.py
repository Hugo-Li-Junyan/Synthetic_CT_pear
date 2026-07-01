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


def register(fixed, moving, model):
    initial = sitk.Euler3DTransform() if model == "rigid" else sitk.AffineTransform(3)
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, initial,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.20, seed=42)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=300,
        convergenceMinimumValue=1e-6, convergenceWindowSize=15,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    return method.Execute(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
    )


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
            registered = sitk.Resample(
                moving, fixed, transform, interpolator, 0.0, moving.GetPixelID()
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
