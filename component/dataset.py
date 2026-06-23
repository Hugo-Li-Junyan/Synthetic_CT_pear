import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils.volumes import list_nifti_files, load_nifti, volume_to_tensor


def _load_volume_tensor(path, transform=None):
    img = volume_to_tensor(load_nifti(path))
    if transform:
        #img = img.unsqueeze(0)
        img = transform(img)
        #img = img.squeeze().unsqueeze(0)
    return img


class CsvVolumeDataset(Dataset):
    def __init__(self, image_dir, csv_path, filename_column="filename", label_column="label",
                 transform=None, label_dtype=torch.long, allowed_labels=None):
        """
        Args:
            image_dir (str): Folder used to resolve relative filenames from the CSV.
            csv_path (str): CSV with one image filename/path and one integer label per row.
            filename_column (str): Column containing `.nii` or `.nii.gz` filenames.
            label_column (str): Column containing integer class labels.
            transform (callable, optional): Optional transform to apply to each volume.
        """
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image folder does not exist: {self.image_dir}")
        self.transform = transform
        self.label_dtype = label_dtype
        self.allowed_labels = set(allowed_labels) if allowed_labels is not None else None
        self.samples = self._read_samples(csv_path, filename_column, label_column)
        if not self.samples:
            raise ValueError(f"No labeled samples found in CSV: {csv_path}")

    def _read_samples(self, csv_path, filename_column, label_column):
        samples = []
        with open(csv_path, newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header row: {csv_path}")
            if filename_column not in reader.fieldnames:
                raise ValueError(f"CSV is missing filename column: {filename_column}")
            if label_column not in reader.fieldnames:
                raise ValueError(f"CSV is missing label column: {label_column}")

            for row in reader:
                filename = row[filename_column].strip()
                if not filename:
                    continue
                path = Path(filename)
                if not path.is_absolute():
                    path = self.image_dir / path
                if not path.exists():
                    continue
                label = float(row[label_column])
                if self.allowed_labels is not None and label not in self.allowed_labels:
                    raise ValueError(f"Label must be one of {sorted(self.allowed_labels)} for {filename}; got {label}")
                if self.label_dtype == torch.long:
                    label = int(label)
                samples.append((path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return _load_volume_tensor(path, self.transform), torch.tensor(label, dtype=self.label_dtype)


class OneClassDataset(Dataset):
    def __init__(self, folder, transform=None):
        """
        Args:
            folder (str): Path to images from one unlabeled class/folder.
            transform (callable, optional): Optional transform to apply to each volume.
        """
        self.images = list_nifti_files(folder)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return _load_volume_tensor(self.images[idx], self.transform), 0
