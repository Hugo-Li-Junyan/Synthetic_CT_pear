import os
import nibabel as nib
import numpy as np
import imageio


def extract_two_slices(volume, save_dir, file_name):
    center_idx = [int(s // 2) for s in volume.shape]
    slice_0 = volume[center_idx[0], :, :]
    slice_0_uint8 = (slice_0 * 255).astype(np.uint8)
    slice_1 = volume[:, center_idx[1], :]
    slice_1_uint8 = (slice_1 * 255).astype(np.uint8)
    file_path_0 = os.path.join(save_dir, file_name + f'_{0}.png')
    file_path_1 = os.path.join(save_dir, file_name + f'_{1}.png')
    imageio.imwrite(file_path_0, slice_0_uint8)
    imageio.imwrite(file_path_1, slice_1_uint8)


def extract_radial_dataset(in_dir, out_dir, label=''):
    print(f"extracting radial slices for {in_dir}")
    for f in os.listdir(in_dir):
        img = nib.load(os.path.join(in_dir, f)).get_fdata()
        # Convert to PyTorch tensor
        # Normalize the image if necessary
        img = (img-np.min(img)) / (np.max(img)-np.min(img))  # Normalize between 0 and 1

        out_file_name = f.split('.')[0] + f'_{label}'
        extract_two_slices(img, out_dir, out_file_name)


if __name__ == '__main__':
    class1_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\healthy"
    class2_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\defective"
    save_real_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\radial_slice_real"

    fake_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\VAE_generation"
    save_fake_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\radial_slice_fake"

    extract_radial_dataset(class1_dir, save_real_dir, label='healthy')
    extract_radial_dataset(class2_dir, save_real_dir, label='defective')
    extract_radial_dataset(fake_dir, save_fake_dir, label='fake')