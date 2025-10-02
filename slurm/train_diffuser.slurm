#!/bin/bash

#SBATCH --job-name=train_diffusion_job
#SBATCH --output=train_diffusion_job.out
#SBATCH --error=train_diffusion_job.err

#SBATCH --partition=gpu_h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=10:00:00
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
python ./train_diffuser.py --class1_dir ../healthy --class2_dir ../defective --save_dir ../model --model_id	20250926-120749 --vae_base_channel 256 --epochs 1200 --random_state 369

# Optional: deactivate conda env (not necessary, but clean)
conda deactivate