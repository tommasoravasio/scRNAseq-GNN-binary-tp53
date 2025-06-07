#!/bin/bash
#SBATCH --job-name=model_comparison
#SBATCH --output=results/model_comparison_output_%j.out
#SBATCH --error=results/model_comparison_error_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Attiva ambiente
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tp53

# Vai alla directory del progetto
cd $HOME/tp53

# Lancia lo script
python src/model_comparison.py