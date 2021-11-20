#!/bin/bash
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=12


#SBATCH --mem=64G
#SBATCH --time=1-12:00
#SBATCH --output=logs/%N-%j.out

source env/bin/activate 
#python train.py config17/train_wt_ax.txt
#python -u CSC490_Main.py
python -u new_train.py config17/train_wt_cr.txt
