import numpy as np
import torch
from torch.utils.data import DataLoader
from generation.component.vae import VAE
from generation.component.dataset import MedicalImageDataset
from visualization.visualization import plot_volume
import os
import nibabel as nib
from generation.component.diffuser import LatentDiffusion


def linear(w,v0,v1):
    return v0 + w * (v1 - v0)


def slerp(w, v0, v1):
    """Spherical linear interpolation between two PyTorch tensors."""
    # Ensure float
    w = torch.tensor(w, dtype=v0.dtype, device=v0.device)

    # Normalize input vectors
    v0_norm = v0 / torch.norm(v0, dim=-1, keepdim=True)
    v1_norm = v1 / torch.norm(v1, dim=-1, keepdim=True)

    # Compute dot product along last dimension
    dot = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    theta_0 = torch.acos(dot)  # angle between input vectors
    sin_theta_0 = torch.sin(theta_0)

    # Small angle case: use linear interpolation
    small_angle = sin_theta_0 < 1e-6
    theta = theta_0 * w
    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0

    # Handle small angles to avoid NaNs
    s0 = torch.where(small_angle, 1 - w, s0)
    s1 = torch.where(small_angle, w, s1)

    return s0 * v0 + s1 * v1

def interpolate_latents(latent_vec1, latent_vec2, interpolation, diffuser, num_steps=11):
    interpolated_latents = []
    if diffuser:
        print('Using diffuser')
        latent_vec1 = diffuser.q_sample(latent_vec1, t=diffuser.timesteps - 1)
        latent_vec2 = diffuser.q_sample(latent_vec2, t=diffuser.timesteps - 1)
    for w in torch.linspace(0, 1, num_steps):
        if interpolation == 'slerp':
            z_interpolated = slerp(w, latent_vec1, latent_vec2)
        elif interpolation == 'linear':
            z_interpolated = linear(w, latent_vec1, latent_vec2)
        else:
            raise ValueError('Only support linear or slerp interpolation')

        if diffuser:
            z_interpolated = diffuser.denoise(z_interpolated, steps=100)
        interpolated_latents.append(z_interpolated)
    return interpolated_latents

def vae_interpolate(vae, dataset, interpolation:str ='slerp', diffuser=None):
    batch_size = 1
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vae.eval()
    with torch.no_grad():
        x_0_flag = False
        x_1_flag = False
        for data in test_loader:
            x, label = data
            if label == 0:
                x_0 = x
                x_0_flag = True
            elif label == 1:
                x_1 = x
                x_1_flag = True
            if x_0_flag and x_1_flag:
                break
        mu_0, logvar_0 = vae.encode(x_0)
        mu_1, logvar_1 = vae.encode(x_1)
        std_0 = torch.exp(0.5 * logvar_0)

        std_1 = torch.exp(0.5 * logvar_1)


        z_sampled_0 = mu_0 + torch.randn_like(std_0) * std_0
        z_sampled_1 = mu_1 + torch.randn_like(std_1) * std_1

        # Interpolate between the sampled latent vectors
        interpolated_latents = interpolate_latents(z_sampled_0, z_sampled_1, interpolation, diffuser, num_steps=11)

        # Decode interpolated latent vectors
        origin_healthy = x_0.squeeze().cpu().numpy()
        generated_images = [origin_healthy]
        generated_images += [vae.decode(latent).squeeze().cpu().numpy() for latent in interpolated_latents]

        origin_defective = x_1.squeeze().cpu().numpy()
        generated_images.append(origin_defective)

    # Visualize one example (modify for 3D)
    plot_volume(origin_healthy)
    plot_volume(generated_images[1])
    plot_volume(origin_defective)
    plot_volume(generated_images[-2])

    return generated_images


if __name__ == "__main__":
    # load vae model
    model_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model"
    model_id = '20250614-104844'
    save_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\VAE_interpolation"
    vae = VAE(input_shape=(1,128,128,128), featuremap_size=32, base_channel=256, flatten_latent_dim=None)
    checkpoint = torch.load(os.path.join(model_dir, model_id,'checkpoint.pth'), map_location='cpu')
    vae.load_state_dict(checkpoint['vae_state_dict'])

    # load diffuser
    #diffuser = LatentDiffusion(input_dim=(1, 32, 32, 32), emb_dim=512, base_channel=128)
    #diffuser_checkpoint = torch.load(os.path.join(model_dir, model_id, 'diffuser_checkpoint.pth'))
    #diffuser.load_state_dict(diffuser_checkpoint['diffuser_state_dict'])

    # Create dataset instance
    class1_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\healthy"
    class2_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\defective"
    dataset = MedicalImageDataset(class1_dir, class2_dir)

    images = vae_interpolate(vae, dataset, interpolation='slerp', diffuser=None)
    for i, arr in enumerate(images):
        arr = ((arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 255).astype(np.uint8)
        img = nib.Nifti1Image(arr, np.eye(4))
        nib.save(img, os.path.join(save_dir,f'{i}.nii'))