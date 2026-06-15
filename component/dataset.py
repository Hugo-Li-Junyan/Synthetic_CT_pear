from torch.utils.data import Dataset

from utils.volumes import list_nifti_files, load_nifti, volume_to_tensor


def _load_volume_tensor(path, transform=None):
    img = volume_to_tensor(load_nifti(path))
    if transform:
        img = img.unsqueeze(0)
        img = transform(img)
        img = img.squeeze().unsqueeze(0)
    return img


class TwoClassDataset(Dataset):
    def __init__(self, class1_dir, class2_dir, transform=None):
        """
        Args:
            class1_dir (str): Path to images from class 0.
            class2_dir (str): Path to images from class 1.
            transform (callable, optional): Optional transform to apply to each volume.
        """
        self.class1_images = list_nifti_files(class1_dir)
        self.class2_images = list_nifti_files(class2_dir)
        self.transform = transform

    def __len__(self):
        return len(self.class1_images) + len(self.class2_images)

    def __getitem__(self, idx):
        if idx < len(self.class1_images):
            img_path = self.class1_images[idx]
            label = 0
        else:
            img_path = self.class2_images[idx - len(self.class1_images)]
            label = 1

        return _load_volume_tensor(img_path, self.transform), label


class OneClassDataset(Dataset):
    def __init__(self, folder, transform=None):
        """
        Args:
            folder (str): Path to images from one class.
            transform (callable, optional): Optional transform to apply to each volume.
        """
        self.images = list_nifti_files(folder)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return _load_volume_tensor(self.images[idx], self.transform), 0
