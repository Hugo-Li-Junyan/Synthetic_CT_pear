import os
import nibabel as nib
import numpy as np
import imageio
from tqdm import tqdm


def _extract_orthogonal_slices(volume, save_dir, file_name):
    center_idx = [int(s // 2) for s in volume.shape]
    slice_0 = volume[center_idx[0], :, :]
    slice_0_uint8 = (slice_0 * 255).astype(np.uint8)
    slice_1 = volume[:, center_idx[1], :]
    slice_1_uint8 = (slice_1 * 255).astype(np.uint8)
    file_path_0 = os.path.join(save_dir, file_name + f'_{0}.png')
    file_path_1 = os.path.join(save_dir, file_name + f'_{1}.png')
    imageio.imwrite(file_path_0, slice_0_uint8)
    imageio.imwrite(file_path_1, slice_1_uint8)


def extract_dataset(in_dir, out_dir, label=''):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print(f"extracting radial slices for {in_dir}")
    for f in tqdm(os.listdir(in_dir), desc="extracting", unit="sample"):
        img = nib.load(os.path.join(in_dir, f)).get_fdata()
        # Convert to PyTorch tensor
        # Normalize the image if necessary
        img = (img-np.min(img)) / (np.max(img)-np.min(img))  # Normalize between 0 and 1

        out_file_name = f.split('.')[0] + f'_{label}'
        _extract_orthogonal_slices(img, out_dir, out_file_name)


if __name__ == '__main__':
    #class1_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\healthy"
    #class2_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\defective"
    #save_real_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\real_slices"

    fake_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model\20250626-021325\VAE_generation"
    save_fake_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model\20250626-021325\fake_slices"

    #extract_dataset(class1_dir, save_real_dir, label='healthy')
    #extract_dataset(class2_dir, save_real_dir, label='defective')
    extract_dataset(fake_dir, save_fake_dir, label='fake')