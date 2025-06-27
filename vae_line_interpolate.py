import numpy as np
import torch
from component.vae import VAE
from utils.visualization import plot_volume
import os
import nibabel as nib
from component.diffuser import LatentDiffusion
import json


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


def line_interpolate(model_dir, save_dir, healthy_dir, defective_dir, diffusion=True, interpolation:str ='slerp'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', 'GPU' if torch.cuda.is_available() else 'CPU')

    # load VAE
    vae_pth = os.path.join(model_dir, 'checkpoint.pth')
    vae_hyperparameter_pth = os.path.join(model_dir, 'vae_hyperparameter.json')
    with open(vae_hyperparameter_pth, 'r') as file:
        vae_hyperparameter = json.load(file)
    vae = VAE(input_shape=(1, 128, 128, 128),
              featuremap_size=vae_hyperparameter["vae_featuremap_size"],
              base_channel=vae_hyperparameter["vae_base_channel"],
              flatten_latent_dim=None,
              with_residual=True)
    vae_checkpoint = torch.load(vae_pth, map_location=device)
    vae.load_state_dict(vae_checkpoint['vae_state_dict'])
    vae = vae.to(device)
    vae.eval()

    diffuser = None
    # load Diffuser
    if diffusion:
        diffuser_pth = os.path.join(model_dir, 'diffuser_best.pth')
        diffuser_hyperparameter_pth = os.path.join(model_dir, 'diffuser_hyperparameter.json')
        with open(diffuser_hyperparameter_pth, 'r') as file:
            diffuser_hyperparameter = json.load(file)
        diffuser = LatentDiffusion(input_dim=(1, 32, 32, 32),
                                   emb_dim=512,
                                   base_channel=diffuser_hyperparameter['base channel'])
        diffuser_checkpoint = torch.load(diffuser_pth, map_location=device)
        diffuser.load_state_dict(diffuser_checkpoint['diffuser_state_dict'])
        diffuser = diffuser.to(device)
        diffuser.eval()

    with torch.no_grad():
        img_0 = nib.load(healthy_dir).get_fdata()
        img_0 = (img_0 - np.min(img_0)) / (np.max(img_0) - np.min(img_0))  # Normalize between 0 and 1
        img_1 = nib.load(defective_dir).get_fdata()
        img_1 = (img_1 - np.min(img_1)) / (np.max(img_1) - np.min(img_1))  # Normalize between 0 and 1
        x_0 = torch.tensor(img_0, requires_grad=False, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        x_1 = torch.tensor(img_1, requires_grad=False, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
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

    for i, arr in enumerate(generated_images):
        arr = ((arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 255).astype(np.uint8)
        img = nib.Nifti1Image(arr, np.eye(4))
        nib.save(img, os.path.join(save_dir, f'{i}.nii'))


if __name__ == "__main__":
    # load vae model
    model_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model\20250614-104844"
    healthy_pth = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\healthy\A34.nii"
    defective_pth = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\defective\A08.nii"
    save_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\VAE_line_interpolation"

    # Create dataset instance
    line_interpolate(model_dir, save_dir, healthy_pth, defective_pth, interpolation='slerp', diffusion=False)
