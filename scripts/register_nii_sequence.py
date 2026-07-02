"""Register corresponding 3D NIfTI files across multiple folders."""

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
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


def foreground_mask(image, threshold=None):
    """Create a binary foreground mask from the image."""
    image = sitk.Cast(image, sitk.sitkFloat32)
    limits = sitk.MinimumMaximumImageFilter()
    limits.Execute(image)
    minimum = float(limits.GetMinimum())
    maximum = float(limits.GetMaximum())
    lower = (
        float(np.nextafter(np.float32(minimum), np.float32(np.inf)))
        if threshold is None
        else float(threshold)
    )
    mask = sitk.BinaryThreshold(
        image,
        lowerThreshold=lower,
        upperThreshold=maximum,
        insideValue=1,
        outsideValue=0,
    )
    return sitk.Cast(mask, sitk.sitkUInt8)


def shape_distance(mask):
    """Return a physical signed-distance field whose zero level is the boundary."""
    return sitk.SignedMaurerDistanceMap(
        mask,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )


def mask_surface_points(mask, image, max_points=2500):
    """Extract a deterministic physical-space point cloud from the boundary."""
    array = sitk.GetArrayViewFromImage(mask) > 0
    surface = array & ~binary_erosion(array)
    zyx = np.argwhere(surface)
    if len(zyx) < 4:
        raise ValueError("Foreground boundary is empty or too small.")
    if len(zyx) > max_points:
        stride = int(np.ceil(len(zyx) / max_points))
        zyx = zyx[::stride]
    xyz = zyx[:, ::-1].astype(float)
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    return (
        (xyz * np.asarray(image.GetSpacing())) @ direction.T
        + np.asarray(image.GetOrigin())
    )


def point_pca(points):
    centroid = points.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov((points - centroid).T))
    return centroid, eigenvectors[:, ::-1]


def rigid_icp(source, target, initial_rotation, initial_translation):
    """Trimmed point-to-point ICP mapping source points onto target points."""
    tree = cKDTree(target)
    rotation = initial_rotation.copy()
    translation = initial_translation.copy()

    for _ in range(40):
        transformed = source @ rotation.T + translation
        distances, indices = tree.query(transformed)
        keep = distances <= np.quantile(distances, 0.85)
        selected = transformed[keep]
        matched = target[indices[keep]]
        source_center = selected.mean(axis=0)
        target_center = matched.mean(axis=0)
        u, _, vt = np.linalg.svd(
            (selected - source_center).T @ (matched - target_center)
        )
        update_rotation = vt.T @ u.T
        if np.linalg.det(update_rotation) < 0:
            vt[-1] *= -1
            update_rotation = vt.T @ u.T
        update_translation = target_center - update_rotation @ source_center
        rotation = update_rotation @ rotation
        translation = update_rotation @ translation + update_translation
        if (
            np.linalg.norm(update_rotation - np.eye(3)) < 1e-7
            and np.linalg.norm(update_translation) < 1e-4
        ):
            break

    return rotation, translation


def pca_rigid_from_masks(fixed, moving, fixed_mask, moving_mask):
    """Rigidly align foreground surfaces and select by full-mask Dice."""
    fixed_points = mask_surface_points(fixed_mask, fixed)
    moving_points = mask_surface_points(moving_mask, moving)
    fixed_center, fixed_axes = point_pca(fixed_points)
    moving_center, moving_axes = point_pca(moving_points)
    fixed_array = sitk.GetArrayViewFromImage(fixed_mask) > 0
    best_dice, best_transform = -1.0, None

    for signs in itertools.product((-1, 1), repeat=3):
        initial_rotation = fixed_axes @ np.diag(signs) @ moving_axes.T
        if np.linalg.det(initial_rotation) < 0:
            continue
        initial_translation = (
            fixed_center - initial_rotation @ moving_center
        )
        # ICP maps moving points to fixed points.
        rotation, translation = rigid_icp(
            moving_points,
            fixed_points,
            initial_rotation,
            initial_translation,
        )
        # SimpleITK resampling needs the inverse: fixed output -> moving input.
        inverse_rotation = rotation.T
        inverse_translation = -inverse_rotation @ translation
        candidate = sitk.Euler3DTransform()
        candidate.SetMatrix(tuple(inverse_rotation.ravel()))
        candidate.SetTranslation(tuple(inverse_translation))
        aligned = sitk.Resample(
            moving_mask,
            fixed_mask,
            candidate,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        aligned_array = sitk.GetArrayViewFromImage(aligned) > 0
        denominator = fixed_array.sum() + aligned_array.sum()
        dice = (
            2.0 * np.logical_and(fixed_array, aligned_array).sum() / denominator
            if denominator
            else 0.0
        )
        if dice > best_dice:
            best_dice, best_transform = float(dice), candidate

    print(f"    rigid surface-ICP foreground Dice={best_dice:.6f}")
    return best_transform

def translation_from_masks(fixed, moving, fixed_mask, moving_mask):
    """Find the integer translation that maximizes binary foreground overlap."""
    fixed_array = sitk.GetArrayFromImage(fixed_mask).astype(np.float32)
    moving_array = sitk.GetArrayFromImage(moving_mask).astype(np.float32)
    correlation = fftconvolve(
        fixed_array,
        moving_array[::-1, ::-1, ::-1],
        mode="full",
    )
    peak = np.asarray(np.unravel_index(np.argmax(correlation), correlation.shape))
    shift_zyx = peak - (np.asarray(moving_array.shape) - 1)
    shift_xyz = shift_zyx[::-1].astype(float)

    direction = np.asarray(fixed.GetDirection()).reshape(3, 3)
    physical_shift = direction @ (shift_xyz * np.asarray(fixed.GetSpacing()))
    transform = sitk.TranslationTransform(3)
    # Resample transforms fixed output points into moving input points.
    transform.SetOffset(tuple((-physical_shift).tolist()))
    print(f"    foreground translation z,y,x={shift_zyx.tolist()} voxels")
    return transform

def register(fixed, moving, model, threshold):
    fixed_mask = foreground_mask(fixed, threshold)
    moving_mask = foreground_mask(moving, threshold)
    if model == "translation":
        return translation_from_masks(
            fixed, moving, fixed_mask, moving_mask
        )
    if model == "rigid":
        return pca_rigid_from_masks(
            fixed, moving, fixed_mask, moving_mask
        )

    fixed_shape = shape_distance(fixed_mask)
    moving_shape = shape_distance(moving_mask)

    initial = sitk.Euler3DTransform() if model == "rigid" else sitk.AffineTransform(3)
    initial = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_mask, sitk.sitkFloat32),
        sitk.Cast(moving_mask, sitk.sitkFloat32),
        initial,
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    method = sitk.ImageRegistrationMethod()
    # Distance-map mean squares optimizes foreground boundary agreement only.
    method.SetMetricAsMeanSquares()
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.15, seed=42)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-3,
        numberOfIterations=200,
        relaxationFactor=0.6,
        gradientMagnitudeTolerance=1e-6,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)

    transform = method.Execute(fixed_shape, moving_shape)
    print(
        f"    shape_metric={method.GetMetricValue():.6g}, "
        f"iterations={method.GetOptimizerIteration()}, "
        f"stop={method.GetOptimizerStopConditionDescription()}"
    )
    return transform

def output_name(name):
    return name[:-7] + ".nii" if name.lower().endswith(".nii.gz") else name


def run(folders, output_dir, model, interpolation, foreground_threshold):
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
            transform = register(fixed, moving, model, foreground_threshold)
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
        "--transform",
        choices=("translation", "rigid", "affine"),
        default="rigid",
        help="Default rigid mode aligns foreground boundary points with ICP."
    )
    parser.add_argument(
        "--interpolation", choices=("linear", "nearest"), default="linear"
    )
    parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=None,
        help="Foreground is intensity >= this value; default uses values above the image minimum.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run(
            args.input_dirs,
            args.output_dir,
            args.transform,
            args.interpolation,
            args.foreground_threshold,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()












