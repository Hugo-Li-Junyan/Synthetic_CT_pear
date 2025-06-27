import numpy as np
import torch
from component.vae import VAE
import os
import nibabel as nib
from component.diffuser import LatentDiffusion


def vae_generate(vae, diffuser, save_dir, batch_size=16, batches=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', 'GPU' if torch.cuda.is_available() else 'CPU')
    vae = vae.to(device)
    diffuser = diffuser.to(device)

    vae.eval()
    diffuser.eval()
    count = 0
    with torch.no_grad():
        for batch in range(batches):
            print(f'start to generate batch {batch}')
            z = torch.randn(batch_size, 1, 32, 32, 32, device=device, requires_grad=False)  # Sample from latent space

            if diffuser:
                z = diffuser.denoise(z, steps=100)
            arr = vae.decode(z).squeeze().cpu().numpy()  # Decode to 3D data
            # Visualize one example (modify for 3D)
            for i in range(batch_size):
                img = arr[i, :, :, :]
                img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
                img = nib.Nifti1Image(img, np.eye(4))
                nib.save(img, os.path.join(save_dir,f'{count}.nii'))
                count += 1


if __name__ == "__main__":
    # load vae model
    model_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model"
    model_id = '20250614-104844'
    model_dir = os.path.join(model_dir, model_id)
    save_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\VAE_generation"
    # load diffuser
    diffuser = LatentDiffusion(input_dim=(1,32,32,32), emb_dim=512, base_channel=128)
    diffuser_checkpoint = torch.load(os.path.join(model_dir, 'diffuser_best.pth'))
    diffuser.load_state_dict(diffuser_checkpoint['diffuser_state_dict'])
    #std = diffuser_checkpoint['std_first_batch']

    vae = VAE(input_shape=(1,128,128,128), featuremap_size=32, base_channel=256, flatten_latent_dim=None, with_residual=True)
    checkpoint = torch.load(os.path.join(model_dir, 'checkpoint.pth'))
    vae.load_state_dict(checkpoint['vae_state_dict'])
    vae_generate(vae, diffuser, save_dir=save_dir, batch_size=2, batches=16) # batch_size * batches ~= 10 * dataset_size
