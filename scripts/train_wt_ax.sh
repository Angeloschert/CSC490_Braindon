#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12


#SBATCH --mem=16G
#SBATCH --time=0-12:00
#SBATCH --output=logs/%N-%j.out

source env/bin/activate 
python train.py config17/train_wt_ax.txt
