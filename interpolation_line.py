import numpy as np
import torch
from component import VAE, LatentDiffusion
from utils.visualization import plot_volume
import os
import nibabel as nib
from utils.load_models import load_vae, load_diffuser
import json
import matplotlib.pyplot as plt


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


def compute_z(dir, vae, device, with_original=True):
    with torch.no_grad():
        img_0 = nib.load(dir).get_fdata()
        img_0 = (img_0 - np.min(img_0)) / (np.max(img_0) - np.min(img_0))  # Normalize between 0 and 1
        x_0 = torch.tensor(img_0, requires_grad=False, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        mu_0, logvar_0 = vae.encode(x_0)
        std_0 = torch.exp(0.5 * logvar_0)
        z_sampled_0 = mu_0 + torch.randn_like(std_0) * std_0
        if with_original:
            return img_0, z_sampled_0
        else:
            return z_sampled_0


def interpolate_latents(latent_vec1, latent_vec2, interpolation, diffuser, num_steps):
    with torch.no_grad():
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


def main(model_dir, save_dir, healthy_dir, defective_dir, num_steps=10, show_latent=False, diffusion=False, interpolation:str ='slerp'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', 'GPU' if torch.cuda.is_available() else 'CPU')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # load VAE
    vae = load_vae(model_dir, device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    diffuser = None
    # load Diffuser
    if diffusion:
        diffuser = load_diffuser(model_dir, device)
        diffuser.eval()
        for param in diffuser.parameters():
            param.requires_grad = False

    img0, z0 = compute_z(healthy_dir, vae, device, with_original=True)
    img1, z1 = compute_z(defective_dir, vae, device, with_original=True)
    # Interpolate between the sampled latent vectors
    interpolated_latents = interpolate_latents(z0, z1, interpolation, diffuser, num_steps=num_steps)
    mid_slice = 64
    if show_latent:
        generated_images = [latent.squeeze().cpu().numpy() for latent in interpolated_latents]
        mid_slice = 16
    else:
    # Decode interpolated latent vectors
        generated_images = [vae.decode(latent).squeeze().cpu().numpy() for latent in interpolated_latents]
    #generated_images.insert(0, img0)
    #generated_images.append(img1)

    # Visualize one example (modify for 3D)
    fig, axes = plt.subplots(nrows=2, ncols=num_steps, figsize=(4*num_steps, 8))
    for i,ax in enumerate(generated_images):
        ax = axes[0,i]
        ax.imshow(generated_images[i][:,mid_slice,:].T, cmap='gray', origin='lower')
        ax.axis('off')

        ax = axes[1,i]
        ax.imshow(generated_images[i][mid_slice,:,:].T, cmap='gray', origin='lower')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    for i, arr in enumerate(generated_images):
        arr = ((arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 255).astype(np.uint8)
        img = nib.Nifti1Image(arr, np.eye(4))
        nib.save(img, os.path.join(save_dir, f'{i}.nii'))


if __name__ == "__main__":
    # load vae model
    model_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model\20250626-021325"
    healthy_pth = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\healthy\A34.nii"
    defective_pth = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\defective\A08.nii"
    save_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\VAE_line_interpolation"

    # Create dataset instance
    main(model_dir, save_dir, healthy_pth, defective_pth, num_steps=10, show_latent=True, interpolation='slerp', diffusion=False)
