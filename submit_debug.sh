#!/bin/bash
#SBATCH --account=robustmicrob
#SBATCH --time=1:00:00
#SBATCH --job-name=esm2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -o /projects/robustmicrob/jlaw/inputs/meltome/embeddings/outputs/%j-log.out
#SBATCH -e /projects/robustmicrob/jlaw/inputs/meltome/embeddings/outputs/%j-log.out
##SBATCH --mail-user=jlaw@nrel.gov
##SBATCH --mail-type=END

source ~/.bashrc
module load gcc cudnn/8.1.1/cuda-11.2
conda activate tm

srun python -u build_embeddings.py
