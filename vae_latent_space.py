import os.path
import nibabel as nib
import numpy as np
import warnings
import torch
from component import MedicalImageDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
from utils.visualization import plot_volume
from utils.load_models import load_vae, load_diffuser


def vae_latent(vae, dataset, sample_size, device):
    test_loader = DataLoader(dataset, batch_size=sample_size, shuffle=True)
    vae.eval()
    labels = []
    with torch.no_grad():
        for data in test_loader:
            x, label = data
            x = x.to(device)
            mu, logvar = vae.encode(x)
            z = vae.reparameterize(mu,logvar)
            labels.append(label)
            break
        return z, labels


def visualize_tsne(z, labels, pca_components=50):
    # Convert PyTorch tensor to NumPy
    z_np = z.reshape(z.shape[0], -1).cpu().detach().numpy()

    if pca_components and pca_components < z_np.shape[1]:
        z_np = PCA(n_components=pca_components).fit_transform(z_np)

    # Apply t-SNE (reducing 128D -> 2D)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(z_np)

    # Plot the 2D representation
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.7, c=labels, cmap='viridis')
    # Create legend manually
    label_names = {0: "Healthy", 1: "Defective"}
    colors = [plt.cm.viridis(i / 1) for i in label_names.keys()]  # Normalized colormap for 0 and 1
    for label, color in zip(label_names.keys(), colors):
        plt.scatter([], [], c=[color], alpha=0.7, label=label_names[label])

    plt.legend(title="Labels")
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.title("t-SNE Visualization of Feature Representations")
    plt.show()


def visualize_umap(z, labels, pca_components=50):
    # Convert PyTorch tensor to NumPy
    z_np = z.reshape(z.shape[0], -1).cpu().detach().numpy()

    # Optional: Reduce dimensionality with PCA first
    if pca_components and pca_components < z_np.shape[1]:
        z_np = PCA(n_components=pca_components).fit_transform(z_np)

    # Apply UMAP (reducing 128D -> 2D)
    reducer = umap.UMAP(n_components=2, random_state=42)
    z_2d = reducer.fit_transform(z_np)

    # Plot the 2D representation
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.7, c=labels, cmap='viridis')
    # Create legend manually
    label_names = {0: "Healthy", 1: "Defective"}
    colors = [plt.cm.viridis(i / 1) for i in label_names.keys()]  # Normalized colormap for 0 and 1
    for label, color in zip(label_names.keys(), colors):
        plt.scatter([], [], c=[color], alpha=0.7, label=label_names[label])

    plt.legend(title="Labels")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.title("UMAP Visualization of Feature Representations")
    plt.show()


def main(model_dir, healthy_dir, defective_dir, method='tsne', sample_size=64, pca_components=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', 'GPU' if torch.cuda.is_available() else 'CPU')
    dataset = MedicalImageDataset(healthy_dir, defective_dir)
    # load VAE
    vae = load_vae(model_dir, device)
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()
    if method == 'dissusion':
        diffuser = load_diffuser(model_dir, device)
        diffuser.eval()
        for param in diffuser.parameters():
            param.requires_grad = False
        z = torch.randn(1, 1, 32, 32, 32, device=device, requires_grad=False)  # Sample from latent space
        z_arr = diffuser.denoise(z, steps=100, save_steps=20)
        for step, z_denoised in enumerate(z_arr):
            volume = z_denoised[0, :, :, :, :].squeeze().cpu().numpy()
            #plot_volume(volume)
            volume = ((volume - np.min(volume)) / (np.max(volume) - np.min(volume)) * 255).astype(np.uint8)
            img = nib.Nifti1Image(volume, np.eye(4))
            nib.save(img, os.path.join(model_dir, f'{step}.nii'))

    else:
        z, labels = vae_latent(vae, dataset, sample_size, device)
        if method == 'tsne':
            visualize_tsne(z, labels, pca_components=pca_components)
        elif method == 'umap':
            visualize_umap(z, labels, pca_components=pca_components)
        elif method == 'volume':
            if sample_size != 1:
                warnings.warn('only visualize 1 sample')
            print(f'label is {labels[0]} (0 for healthy, 1 for defective)')
            volume = z[0, :, :, :, :].squeeze().cpu().numpy()
            plot_volume(volume)
            volume = ((volume - np.min(volume)) / (np.max(volume) - np.min(volume)) * 255).astype(np.uint8)
            img = nib.Nifti1Image(volume, np.eye(4))
            nib.save(img, os.path.join(model_dir, 'latent.nii'))
        else:
            raise ValueError('only tsne and umap are supported')


if __name__ == "__main__":
    # load vae model
    model_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model\20250626-021325"
    healthy_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\healthy"
    defective_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\defective"
    main(model_dir, healthy_dir, defective_dir, method='volume', sample_size=1, pca_components=5)
