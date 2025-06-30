import torch
from component import VAE
from component import MedicalImageDataset
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA


def vae_latent(vae, dataset):
    batch_size = 128
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vae.eval()
    labels = []
    with torch.no_grad():
        for data in test_loader:
            x, label = data
            mu, logvar = vae.encode(x)
            z = vae.reparameterize(mu,logvar).reshape(batch_size, -1)
            labels.append(label)
            break
        return z, labels


def visualize_tsne(z, labels, pca_components=50):
    # Convert PyTorch tensor to NumPy
    z_np = z.cpu().detach().numpy()

    if pca_components and pca_components < z_np.shape[1]:
        z_np = PCA(n_components=pca_components).fit_transform(z_np)

    # Apply t-SNE (reducing 128D -> 2D)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(z_np)

    # Plot the 2D representation
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.7, c=labels)
    plt.colorbar(scatter, label="Labels")
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.title("t-SNE Visualization of Feature Representations")
    plt.show()


def visualize_umap(z, labels, pca_components=50):
    # Convert PyTorch tensor to NumPy
    z_np = z.cpu().detach().numpy()

    # Optional: Reduce dimensionality with PCA first
    if pca_components and pca_components < z_np.shape[1]:
        z_np = PCA(n_components=pca_components).fit_transform(z_np)

    # Apply UMAP (reducing 128D -> 2D)
    reducer = umap.UMAP(n_components=2, random_state=42)
    z_2d = reducer.fit_transform(z_np)

    # Plot the 2D representation
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.7, c=labels)
    plt.colorbar(scatter, label="Labels")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.title("UMAP Visualization of Feature Representations")
    plt.show()


if __name__ == "__main__":
    # load vae model
    model_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model"
    model_id = '20250507-173646'
    vae = VAE(input_shape=(1, 128, 128, 128), featuremap_size=32, base_channel=128, flatten_latent_dim=None)
    checkpoint = torch.load(os.path.join(model_dir, model_id, 'checkpoint.pth'), map_location='cpu')
    vae.load_state_dict(checkpoint['vae_state_dict'])

    # Create dataset instance
    class1_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\healthy"
    class2_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\defective"
    dataset = MedicalImageDataset(class1_dir, class2_dir)

    z, labels = vae_latent(vae, dataset)
    visualize_umap(z, labels, pca_components=50)