import nibabel as nib  # For NIfTI files, if you're using .nii/.nii.gz format
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch

# Define the custom dataset
class TwoClassDataset(Dataset):
    def __init__(self, class1_dir, class2_dir, transform=None):
        """
        Args:
            class1_dir (string): Path to the folder containing images from Class 1.
            class2_dir (string): Path to the folder containing images from Class 2.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.class1_images = [os.path.join(class1_dir, f) for f in os.listdir(class1_dir)]
        self.class2_images = [os.path.join(class2_dir, f) for f in os.listdir(class2_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.class1_images) + len(self.class2_images)

    def __getitem__(self, idx):
        # Determine if the image is from Class 1 or Class 2
        if idx < len(self.class1_images):
            img_path = self.class1_images[idx]
            label = 0  # Class 1
        else:
            img_path = self.class2_images[idx - len(self.class1_images)]
            label = 1  # Class 2

        # Load the image (Assuming 3D medical image in NIfTI format)
        img = nib.load(img_path).get_fdata()
        # Convert to PyTorch tensor
        # Normalize the image if necessary
        img = (img-np.min(img)) / (np.max(img)-np.min(img))  # Normalize between 0 and 1
        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)
        if self.transform:
            img = img.unsqueeze(0)
            img = self.transform(img)
            img = img.squeeze().unsqueeze(0)
        return img, label

# Define the custom dataset
class OneClassDataset(Dataset):
    def __init__(self, folder, transform=None):
        """
        Args:
            class1_dir (string): Path to the folder containing images from Class 1.
            class2_dir (string): Path to the folder containing images from Class 2.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = [os.path.join(folder, f) for f in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Determine if the image is from Class 1 or Class 2

        img_path = self.images[idx]


        # Load the image (Assuming 3D medical image in NIfTI format)
        img = nib.load(img_path).get_fdata()
        # Convert to PyTorch tensor
        # Normalize the image if necessary
        img = (img-np.min(img)) / (np.max(img)-np.min(img))  # Normalize between 0 and 1
        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)
        if self.transform:
            img = img.unsqueeze(0)
            img = self.transform(img)
            img = img.squeeze().unsqueeze(0)
        return img, 0