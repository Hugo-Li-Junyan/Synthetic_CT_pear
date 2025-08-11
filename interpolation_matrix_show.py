import os
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib import gridspec


def main(save_dir):
    files = os.listdir(save_dir)
    num_rows = int(len(files) ** 0.5)
    fig = plt.figure(figsize=(4*num_rows, 4*num_rows))
    gs = gridspec.GridSpec(num_rows, num_rows,
                           wspace=0, hspace=0,
                           top=1, bottom=0, left=0, right=1)  # No space or margins

    for row in range(num_rows):
        for col in range(num_rows):
            ax = fig.add_subplot(gs[row, col])
            img = nib.load(os.path.join(save_dir,f'{row}_{col}.nii')).get_fdata()
            ax.imshow(img[:,64,:].T, cmap='gray', origin='lower')
            ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(0)
    plt.savefig("interpolation_matrix.png", bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    save_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\VAE_matrix_interpolation"
    main(save_dir)