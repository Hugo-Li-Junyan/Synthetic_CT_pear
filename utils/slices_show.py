import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib import gridspec


def show_random_images_grid(folder_path, n_rows, n_cols):
    # Get all image paths ending with _1.png
    all_images = [f for f in os.listdir(folder_path) if f.endswith('_1.png')]

    required_num = n_rows * n_cols
    if len(all_images) < required_num:
        raise ValueError(f"Not enough images ending with '_1.png'. Found {len(all_images)}, need {required_num}.")

    # Randomly select images
    selected_images = random.sample(all_images, required_num)

    # Set up plot grid
    fig = plt.figure(figsize=(n_cols * 2, n_rows * 2))
    gs = gridspec.GridSpec(n_rows, n_rows,
                           wspace=0, hspace=0,
                           top=1, bottom=0, left=0, right=1)  # No space or margins

    i=0
    for row in range(n_rows):
        for col in range(n_rows):
            ax = fig.add_subplot(gs[row, col])
            img_path = os.path.join(folder_path, all_images[i])
            img = Image.open(img_path)
            ax.imshow(np.array(img).T, cmap='gray', origin='lower')
            ax.axis('off')
            i+=1
            #ax.set_title(img_name, fontsize=8)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(0)
    plt.tight_layout()
    plt.savefig("output.png", bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    show_random_images_grid(r'J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\model\20250626-021325\fake_slices', 8,8)