from utils.load_models import load_vae
import torch
from torch.utils.data import DataLoader, random_split
from component import MedicalImageDataset
from utils.metrics import mae, ssim, psnr
from tqdm import tqdm


def main(model_dir, healthy_dir, defective_dir, batch_size=2, val_split=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', 'GPU' if torch.cuda.is_available() else 'CPU')

    vae, random_state = load_vae(model_dir, device, with_rand_state=True)
    for param in vae.parameters():
        param.requires_grad = False

    dataset = MedicalImageDataset(healthy_dir, defective_dir)
    generator = torch.Generator().manual_seed(random_state)
    _, val_dataset = random_split(dataset, [1-val_split, val_split], generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    val_size = int(len(dataset) * val_split)

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
    model_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model\20250614-104844"
    healthy_dir = r"D:\Hugo\healthy"
    defective_dir = r"D:\Hugo\defective"
    main(model_dir, healthy_dir, defective_dir)