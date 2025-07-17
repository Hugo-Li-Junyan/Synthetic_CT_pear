import os
import matplotlib.pyplot as plt


def main(save_dir):
    files = os.listdir(save_dir)
    num_rows = int(len(files) ** 0.5)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_rows, figsize=(4*num_rows, 4*num_rows))


if __name__ == "__main__":
    save_dir = r"J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\VAE_matrix_interpolation"
    main(save_dir)