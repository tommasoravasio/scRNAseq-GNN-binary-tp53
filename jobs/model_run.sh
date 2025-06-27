#!/bin/bash
#SBATCH --job-name=run_model
#SBATCH --output=model_output_%j.out
#SBATCH --error=model_error_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1




module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tp53
cd $HOME/tp53

# #Simply train the model:
# python src/model_constructor.py --mode train --config configs/simple_run_template.json

# Run optuna hyperparameter optimization: 
python src/model_constructor.py --mode optuna --config configs/optuna_template.json