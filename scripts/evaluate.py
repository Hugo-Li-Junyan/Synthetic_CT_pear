import argparse

from utils.load_models import load_vae
import torch
from torch.utils.data import DataLoader
from component import CsvVolumeDataset
from utils.metrics import mae, ssim, psnr
from tqdm import tqdm
from utils.splits import split_train_val


def main(model_dir, image_dir, labels_csv, filename_column="filename", label_column="label", batch_size=4, val_split=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', 'GPU' if torch.cuda.is_available() else 'CPU')

    vae, random_state = load_vae(model_dir, device, with_rand_state=True)
    for param in vae.parameters():
        param.requires_grad = False

    dataset = CsvVolumeDataset(
        image_dir,
        labels_csv,
        filename_column=filename_column,
        label_column=label_column,
    )
    _, val_dataset = split_train_val(dataset, val_split, random_state)
    val_size = len(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    with torch.no_grad():
        MAE, SSIM, PSNR = 0.0, 0.0, 0.0
        for data in tqdm(val_loader, desc="Validating", unit="batch"):
            x, _ = data
            x = x.to(device)
            reconstructed_x, _, _ = vae(x)
            MAE += mae(x, reconstructed_x).item() * x.size(0)
            SSIM += ssim(x, reconstructed_x).item() * x.size(0)
            PSNR += psnr(x, reconstructed_x).item() * x.size(0)
        MAE /= val_size
        SSIM /= val_size
        PSNR /= val_size
    print(f"{100*val_split}% validation data with {val_size} instances")
    print(f"MAE = {MAE:.4f}, SSIM = {SSIM:.4f}, PSNR = {PSNR:.4F}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained VAE model")
    parser.add_argument("--model_dir", type=str, required=True, help="directory containing the trained model")
    parser.add_argument("--image_dir", type=str, required=True, help="folder containing labeled NIfTI volumes")
    parser.add_argument("--labels_csv", type=str, required=True, help="CSV containing filenames and labels")
    parser.add_argument("--filename_column", type=str, default="filename", help="CSV filename column")
    parser.add_argument("--label_column", type=str, default="label", help="CSV label column")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--val_split", type=float, default=0.1, help="validation split ratio")
    args = parser.parse_args()

    main(
        args.model_dir,
        args.image_dir,
        args.labels_csv,
        filename_column=args.filename_column,
        label_column=args.label_column,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
