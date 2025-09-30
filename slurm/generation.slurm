#!/bin/bash

#SBATCH --job-name=generate_job
#SBATCH --output=generate_job.out
#SBATCH --error=generate_job.err

#SBATCH --partition=gpu_a100
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
python ./main.py --model_dir ../model/20250926-120749 --save_dir ../generation

# Optional: deactivate conda env (not necessary, but clean)
conda deactivate