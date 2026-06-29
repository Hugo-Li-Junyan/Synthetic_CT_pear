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
python ./train_diffuser.py --image_dir ../images --labels_csv ../labels.csv --label_column browning --save_dir ../model --model_id beta1e-3gamma0.01 --vae_base_channel 256 --epochs 1000 --random_state 369

# Optional: deactivate conda env (not necessary, but clean)
conda deactivate
