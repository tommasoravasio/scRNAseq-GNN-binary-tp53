#!/bin/bash
#SBATCH --job-name=correlation_tp53
#SBATCH --output=correlation_tp53_%j.out
#SBATCH --error=correlation_tp53_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00



module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tp53
cd $HOME/tp53
python src/correlation_comparison.py 