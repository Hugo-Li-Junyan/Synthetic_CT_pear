import json
import os
import torch
from component import VAE, LatentDiffusion


def load_vae(model_dir, device, with_rand_state=False):
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
    vae.to(device)
    if with_rand_state:
        seed = vae_checkpoint['random_state']
        return vae, seed
    else:
        return vae


def load_diffuser(model_dir, device):
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
    diffuser.to(device)
    return diffuser
