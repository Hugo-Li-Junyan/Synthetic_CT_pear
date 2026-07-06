import argparse
import torch
import os
import time
from utils.load_models import load_vae, load_diffuser
from tqdm import tqdm
from utils.volumes import save_nifti, to_uint16


def vae_generate(model_dir, save_dir, batch_size: int = 2, batches: int = 16,
                 sampler: str = "ddim", ddim_steps: int = 200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', 'GPU' if torch.cuda.is_available() else 'CPU')

    os.makedirs(save_dir, exist_ok=True)

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
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(range(batches), desc="Generating", unit="batch"):
            z = torch.randn(batch_size, 1, 32, 32, 32, device=device, requires_grad=False)  # Sample from latent space
            z = diffuser.denoise(z, steps=ddim_steps, sampler=sampler)
            arr = vae.decode(z).squeeze().cpu().numpy()
            for i in range(batch_size):
                img = to_uint16(arr[i, :, :, :])
                save_nifti(img, os.path.join(save_dir, f'{count}.nii'))
                count += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    average_time = total_time / count if count else 0.0
    print(f"Generated {count} volumes in {total_time:.2f} seconds")
    print(f"Average generation time: {average_time:.4f} seconds per volume")


def main():
    parser = argparse.ArgumentParser(description="generate for vae")
    # dir parser
    parser.add_argument("--model_dir", type=str, required=True, help="model_dir")
    parser.add_argument("--save_dir", type=str, required=True, help="save_dir")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--batches", type=int, default=3000, help="number of batches")
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim",
                        help="diffusion sampler; DDPM always uses the full trained schedule")
    parser.add_argument("--ddim_steps", type=int, default=200,
                        help="number of DDIM sampling steps (ignored for DDPM)")
    args = parser.parse_args()
    vae_generate(
        args.model_dir,
        args.save_dir,
        batch_size=args.batch_size,
        batches=args.batches,
        sampler=args.sampler,
        ddim_steps=args.ddim_steps,
    )  # batch_size * batches ~= 10 * dataset_size


if __name__ == "__main__":
    main()
