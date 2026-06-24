from pathlib import Path

import nibabel as nib
import numpy as np
import torch


NIFTI_SUFFIXES = (".nii", ".nii.gz")


def list_nifti_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and any(str(path).lower().endswith(suffix) for suffix in NIFTI_SUFFIXES)
    )


def normalize_minmax(volume, eps=1e-8):
    volume = np.asarray(volume, dtype=np.float32)
    min_value = np.min(volume)
    value_range = np.max(volume) - min_value
    if value_range < eps:
        return np.zeros_like(volume, dtype=np.float32)
    return (volume - min_value) / value_range


def load_nifti(path, normalize=True):
    volume = nib.load(str(path)).get_fdata()
    if normalize:
        volume = normalize_minmax(volume)
    return volume


def volume_to_tensor(volume):
    volume = np.ascontiguousarray(volume, dtype=np.float32)
    return torch.tensor(volume, dtype=torch.float32).unsqueeze(0)




def to_uint16(volume):
    return (normalize_minmax(volume) * 65535).astype(np.uint16)


def save_nifti(volume, path, affine=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(volume, affine), str(path))


