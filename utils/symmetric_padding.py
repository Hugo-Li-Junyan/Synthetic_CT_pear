import os
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
from utils.volumes import save_nifti, to_uint16


def min_max_16bit_to_8bit(img):
    min_val = np.min(img)
    max_val = np.max(img)

    img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return img

def resize_pad(img, target_size):
    scale_factor = target_size / max(img.shape)  # Scale the largest axis to target_size

    # Step 2: Compute new resized shape
    new_shape = tuple([int(dim * scale_factor) for dim in img.shape])

    # Step 3: Resize the image using interpolation (SciPy zoom)
    resized_image = zoom(img, (new_shape[0] / img.shape[0],
                                 new_shape[1] / img.shape[1],
                                 new_shape[2] / img.shape[2]), order=1, mode='nearest')

    # Step 4: Pad the resized image
    pad_x = (target_size - new_shape[0]) // 2
    pad_y = (target_size - new_shape[1]) // 2
    pad_z = (target_size - new_shape[2]) // 2

    pad_x_extra = (target_size - new_shape[0]) % 2
    pad_y_extra = (target_size - new_shape[1]) % 2
    pad_z_extra = (target_size - new_shape[2]) % 2

    padded_image = np.pad(resized_image,
                          ((pad_x, pad_x + pad_x_extra),
                           (pad_y, pad_y + pad_y_extra),
                           (pad_z, pad_z + pad_z_extra)),
                          mode='constant', constant_values=np.min(img))
    return padded_image


def preprocess_folder(folder_path, output_path, new_size, compressed=False):
    samples_names = os.listdir(folder_path)


    for sample_name in samples_names:
        sample_path = os.path.join(folder_path, sample_name)
        sample = nib.load(sample_path)
        sample_array = sample.get_fdata().astype(np.int16)
        #sample_8bit = min_max_16bit_to_8bit(sample_array)
        sample_resized = resize_pad(sample_array, new_size)

        sample_out_path = os.path.join(output_path, sample_name)
        if compressed:
            sample_out_path+='.gz'
        save_nifti(to_uint16(sample_resized), sample_out_path)



if __name__ == '__main__':
    path = r'D:\Hugo\synthetic_paper\train_new_year\data'
    out_path = r'D:\Hugo\synthetic_paper\train_new_year\data++'
    preprocess_folder(path, out_path, new_size=128)

