from component.dataset import MedicalImageDataset
import torchio as tio
from torch.utils.data import DataLoader
from utils.visualization import plot_volume
import torch


if __name__ == '__main__':
    transform = tio.Compose([
        tio.RandomFlip(axes=(0,1)),
        tio.RandomAffine(
            scales=(0.9, 1.1),
            degrees=(0,0,0,0,-30,30),
            isotropic=True
        )
    ])
    class1_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\healthy"
    class2_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\defective"
    dataset = MedicalImageDataset(class1_dir, class2_dir, transform=transform)
    batch_size = 8
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for data in test_loader:
            x, label = data
            for i in range(batch_size):
                plot_volume(x.squeeze().cpu().numpy()[i,:,:,:])
            break
