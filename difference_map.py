import torch
import os
import numpy as np
import nibabel as nib
from utils.load_models import load_vae
import matplotlib.pyplot as plt
from component import MedicalImageDataset
from torch.utils.data import DataLoader, random_split


def main(model_dir, healthy_dir, defective_dir, max_size=6, val_split=0.1):
    # load VAE
    vae, random_state = load_vae(model_dir, 'cpu', with_rand_state=True)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    dataset = MedicalImageDataset(healthy_dir, defective_dir)
    generator = torch.Generator().manual_seed(random_state)
    _, val_dataset = random_split(dataset, [1 - val_split, val_split], generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    fig, axes = plt.subplots(nrows=max_size, ncols=6, figsize=(12, 15))
    num_healthy=0
    num_defective=0
    idx = 0
    with torch.no_grad():
        for data in val_loader:
            x_0, label = data
            if label == 0:
                num_healthy += 1
                if num_healthy > max_size / 2:
                    continue
            if label == 1:
                num_defective += 1
                if num_defective > max_size / 2:
                    continue
            idx += 1
            x_1, _, _ = vae(x_0)
            img_0 = x_0.squeeze().cpu().numpy()
            img_1 = x_1.squeeze().cpu().numpy()

            ax = axes[idx-1, 0]
            ax.imshow(img_1[:,64,:].T, cmap='gray', origin='lower')
            ax.axis('off')

            ax = axes[idx - 1, 1]
            ax.imshow(img_1[:,64,:].T-img_0[:,64,:].T,cmap='Spectral_r', vmin=-1, vmax=1, origin='lower')
            ax.axis('off')

            ax = axes[idx - 1, 2]
            ax.imshow(img_1[64,:, :].T, cmap='gray', origin='lower')
            ax.axis('off')

            ax = axes[idx - 1, 3]
            ax.imshow(img_1[64,:, :].T - img_0[64,:, :].T,cmap='Spectral_r', vmin=-1, vmax=1, origin='lower')
            ax.axis('off')

            ax = axes[idx - 1, 4]
            ax.imshow(img_1[:, :, 64].T, cmap='gray', origin='lower')
            ax.axis('off')

            ax = axes[idx - 1, 5]
            im = ax.imshow(img_1[:, :, 64].T - img_0[:, :, 64].T, cmap='Spectral_r', vmin=-1, vmax=1, origin='lower')
            ax.axis('off')

            if idx >= max_size:
                cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # [left, bottom, width, height]
                fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

                plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for colorbar
                plt.show()
                break


if __name__ == "__main__":
    model_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model\20250626-021325"
    healthy_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\healthy"
    defective_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\defective"

    main(model_dir, healthy_dir, defective_dir)