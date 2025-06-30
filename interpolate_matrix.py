import numpy as np
import torch
from component import VAE
import os
import nibabel as nib
from component import LatentDiffusion
import json
from interpolate_line import interpolate_latents, compute_z
from utils.load_models import load_vae, load_diffuser


def matrix_interpolate(model_dir, save_dir, topleft_dir, lowerleft_dir, lowerright_dir, topright_dir,
                       diffusion=True, interpolation:str ='slerp'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', 'GPU' if torch.cuda.is_available() else 'CPU')

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

    img_topleft, z_topleft = compute_z(topleft_dir, vae, device, with_original=True)
    img_lowerleft, z_lowerleft = compute_z(lowerleft_dir,  vae, device, with_original=True)
    img_lowerright, z_lowerright = compute_z(lowerright_dir, vae, device, with_original=True)
    img_topright, z_topright = compute_z(topright_dir, vae, device, with_original=True)

    # Interpolate between the sampled latent vectors
    left_interpolated_latents = interpolate_latents(z_topleft, z_lowerleft, interpolation, diffuser, num_steps=11)
    right_interpolated_latents = interpolate_latents(z_topright, z_lowerright, interpolation, diffuser, num_steps=11)
    # Decode interpolated latent vectors
    for row in range(len(left_interpolated_latents)):
        left_z = left_interpolated_latents[row]
        right_z = right_interpolated_latents[row]
        row_interpolated = interpolate_latents(left_z, right_z, interpolation, diffuser, num_steps=11)
        row_generated_images = [vae.decode(latent).squeeze().cpu().numpy() for latent in row_interpolated]

        for i, arr in enumerate(row_generated_images):
            arr = ((arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 255).astype(np.uint8)
            img = nib.Nifti1Image(arr, np.eye(4))
            nib.save(img, os.path.join(save_dir, f'{row}_{i}.nii'))
    nib.save(nib.Nifti1Image(img_topleft, np.eye(4)),os.path.join(save_dir, 'topleft.nii'))
    nib.save(nib.Nifti1Image(img_lowerleft, np.eye(4)), os.path.join(save_dir, 'lowerleft.nii'))
    nib.save(nib.Nifti1Image(img_lowerright, np.eye(4)), os.path.join(save_dir, 'lowerright.nii'))
    nib.save(nib.Nifti1Image(img_topright, np.eye(4)), os.path.join(save_dir, 'topright.nii'))

if __name__ == "__main__":
    # load vae model
    model_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model\20250614-104844"
    topleft_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\healthy\A34.nii"
    lowerleft_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\defective\A08.nii"
    lowerright_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\healthy\D46.nii"
    topright_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\defective\E08.nii"
    save_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\VAE_matrix_interpolation"

    # Create dataset instance
    matrix_interpolate(model_dir, save_dir, topleft_dir, lowerleft_dir, lowerright_dir, topright_dir, interpolation='slerp', diffusion=False)
