from component.gan import PatchGAN
import nibabel as nib
import torch
import numpy as np
import os


def load_nib(img_path):
    img = nib.load(img_path).get_fdata()
    # Convert to PyTorch tensor
    # Normalize the image if necessary
    img = img / np.max(img)  # Normalize between 0 and 1
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0).unsqueeze(0)
    return img


if __name__ == "__main__":
    fake_path = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\VAE_interpolation\11.nii"
    real_path = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\VAE_interpolation\12.nii"

    fake_x = load_nib(fake_path)
    real_x = load_nib(real_path)
    gan = PatchGAN(x_shape=(1,128,128,128), patch_size=32, with_residual=True, base_channel=32, weight_fn='weighted')
    gan_checkpoint_path = r'J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model\20250614-104844\checkpoint.pth'
    if os.path.exists(gan_checkpoint_path):
        gan.load_state_dict(torch.load(gan_checkpoint_path, map_location='cpu')['gan_state_dict'])
    with torch.no_grad():
        gan_loss = gan.loss_function(real_x, fake_x)
        adv_loss = gan.adversarial_loss(real_x, fake_x)
        print('gan_loss:', gan_loss)