#!/bin/bash

#SBATCH --job-name=train_vae_job
#SBATCH --output=train_vae_job.out
#SBATCH --error=train_vae_job.err

#SBATCH --partition=gpu_h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user hugo.li@kuleuven.be

# Activate your environment
source /data/leuven/362/vsc36240/miniconda3_wice/etc/profile.d/conda.sh
conda activate base
cd /data/leuven/362/vsc36240/project/synthetic_data/Synthetic_CT_pear
# Run your script with parameters
python ./train_diffuser.py --class1_dir ../healthy --class2_dir ../defective --save_dir ../model --model_id 20250626-021325 --epochs 800 --random_state 369

# Optional: deactivate conda env (not necessary, but clean)
conda deactivate