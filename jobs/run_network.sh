#!/bin/bash
#SBATCH --job-name=network_job_tp53
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=network_job_output.log
#SBATCH --error=network_job_error.log


module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tp53

cd $HOME/tp53

python src/network_constructor.py

