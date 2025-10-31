import os
import random
import shutil



def randomize_folders(output_folder, input_paths, num_folds:int=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(num_folds):
        if not os.path.exists(os.path.join(output_folder, 'folder_'+str(i))):
            os.makedirs(os.path.join(output_folder, 'folder_'+str(i)))

    all_file_paths = []
    for path in input_paths:
        files = os.listdir(path)
        all_file_paths += [os.path.join(path,file) for file in files]
    random.shuffle(all_file_paths)
    num_samples = len(all_file_paths)
    for j in range(num_samples):
        for i in range(num_folds):
            if i * num_samples//num_folds <= j < (i+1)*num_samples//num_folds:
                shutil.copy(all_file_paths[j], os.path.join(output_folder, 'folder_'+str(i)))
    print('done')


if __name__ == '__main__':
    path_1 = r'J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\real_slices'
    paths = [path_1]
    output_folder = r'J:\SET-Mebios_CFD-VIS-DI0327\HugoLi\PomestoreID\Pear\for_training\randomized_real_slices'
    randomize_folders(output_folder, paths)