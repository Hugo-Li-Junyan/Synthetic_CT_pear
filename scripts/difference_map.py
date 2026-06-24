import torch
import os
import numpy as np
import nibabel as nib
from utils.load_models import load_vae
import matplotlib.pyplot as plt
from component import CsvVolumeDataset
from torch.utils.data import DataLoader
from utils.splits import split_train_val


def main(model_dir, image_dir, labels_csv, filename_column="filename", label_column="cavity", max_size=6, val_split=0.1):
    # load VAE
    vae, random_state = load_vae(model_dir, 'cpu', with_rand_state=True)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    dataset = CsvVolumeDataset(
        image_dir,
        labels_csv,
        filename_column=filename_column,
        label_column=label_column,
    )
    _, val_dataset = split_train_val(dataset, val_split, random_state)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

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

            ax = axes[idx - 1, 0]
            ax.imshow(img_1[:, :, 64].T, cmap='gray', origin='lower')
            ax.axis('off')

            ax = axes[idx - 1, 1]
            im = ax.imshow(img_1[:, :, 64].T - img_0[:, :, 64].T, cmap='Spectral_r', vmin=-1, vmax=1, origin='lower')
            ax.axis('off')

            ax = axes[idx-1, 2]
            ax.imshow(img_1[:,64,:].T, cmap='gray', origin='lower')
            ax.axis('off')

            ax = axes[idx - 1, 3]
            ax.imshow(img_1[:,64,:].T-img_0[:,64,:].T,cmap='Spectral_r', vmin=-1, vmax=1, origin='lower')
            ax.axis('off')

            ax = axes[idx - 1, 4]
            ax.imshow(img_1[64,:, :].T, cmap='gray', origin='lower')
            ax.axis('off')

            ax = axes[idx - 1, 5]
            ax.imshow(img_1[64,:, :].T - img_0[64,:, :].T,cmap='Spectral_r', vmin=-1, vmax=1, origin='lower')
            ax.axis('off')



            if idx >= max_size:
                cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # [left, bottom, width, height]
                fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

                plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for colorbar
                plt.savefig("difference_map.png", bbox_inches='tight', pad_inches=0)
                break


if __name__ == "__main__":
    model_dir = r"D:\Hugo\synthetic_paper\synthetic_model\beta1e-3nogan"
    image_dir = r"D:\Hugo\conference_feb2025\volumes"
    labels_csv = r"D:\Hugo\conference_feb2025\labels.csv"

    main(model_dir, image_dir, labels_csv)
