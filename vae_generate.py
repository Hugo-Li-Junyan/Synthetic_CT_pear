import numpy as np
import torch
from component.vae import VAE
import os
import nibabel as nib
from component.diffuser import LatentDiffusion
import json


def vae_generate(model_dir, save_dir, batch_size:int=2, batches:int=16):
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
    for param in vae.parameters():
        param.requires_grad = False

    # load Diffuser
    diffuser_pth = os.path.join(model_dir, 'diffuser_best.pth')
    diffuser_hyperparameter_pth = os.path.join(model_dir, 'diffuser_hyperparameter.json')
    with open(diffuser_hyperparameter_pth, 'r') as file:
        diffuser_hyperparameter = json.load(file)
    diffuser = LatentDiffusion(input_dim=(1, 32, 32, 32),
                               emb_dim=512,
                               base_channel=diffuser_hyperparameter['base channel'])
    diffuser_checkpoint = torch.load(diffuser_pth, map_location=device)
    diffuser.load_state_dict(diffuser_checkpoint['diffuser_state_dict'])
    for param in diffuser.parameters():
        param.requires_grad = False

    # move to GPU
    vae = vae.to(device)
    diffuser = diffuser.to(device)

    vae.eval()
    diffuser.eval()
    count = 0
    with torch.no_grad():
        for batch in range(batches):
            print(f'start to generate batch {batch}')
            z = torch.randn(batch_size, 1, 32, 32, 32, device=device, requires_grad=False)  # Sample from latent space
            z = diffuser.denoise(z, steps=100)
            arr = vae.decode(z).squeeze().cpu().numpy()
            for i in range(batch_size):
                img = arr[i, :, :, :]
                img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
                img = nib.Nifti1Image(img, np.eye(4))
                nib.save(img, os.path.join(save_dir, f'{count}.nii'))
                count += 1


if __name__ == "__main__":
    # load vae model
    model_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model\20250614-104844"
    save_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\VAE_generation"
    vae_generate(model_dir, save_dir, batch_size=2, batches=16) # batch_size * batches ~= 10 * dataset_size
