import os
import matplotlib.pyplot as plt
import nibabel as nib


def main(save_dir):
    files = os.listdir(save_dir)
    num_rows = int(len(files) ** 0.5)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_rows, figsize=(4*num_rows, 4*num_rows))
    for row in range(num_rows):
        for col in range(num_rows):
            ax = axes[row][col]
            img = nib.load(os.path.join(save_dir,f'{row}_{col}.nii')).get_fdata()
            ax.imshow(img[:,64,:].T, cmap='gray', origin='lower')
            ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    save_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\VAE_matrix_interpolation"
    main(save_dir)