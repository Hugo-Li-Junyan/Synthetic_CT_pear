import argparse
import numpy as np
import torch
import os
import nibabel as nib
from utils.load_models import load_vae, load_diffuser


def vae_generate(model_dir, save_dir, batch_size:int=2, batches:int=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', 'GPU' if torch.cuda.is_available() else 'CPU')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # load VAE
    vae = load_vae(model_dir, device)
    for param in vae.parameters():
        param.requires_grad = False

    # load Diffuser
    diffuser = load_diffuser(model_dir, device)
    for param in diffuser.parameters():
        param.requires_grad = False

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

def main():
    parser = argparse.ArgumentParser(description="generate for vae")
    # dir parser
    parser.add_argument("--model_dir", type=str, required=True, help="model_dir")
    parser.add_argument("--save_dir", type=str, required=True, help="save_dir")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--batches", type=int, default=3000, help="number of batches")
    args = parser.parse_args()
    vae_generate(args.model_dir, args.save_dir, batch_size=args.batch_size, batches=args.batches)  # batch_size * batches ~= 10 * dataset_size


if __name__ == "__main__":
    main()
