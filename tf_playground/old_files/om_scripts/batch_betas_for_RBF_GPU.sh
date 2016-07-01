#!/bin/bash
#SBATCH --job-name=HBF1_model_train
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com
#SBATCH --gres=gpu:1

matlab -nodesktop -nosplash -nojvm -r "run choosing_beta.m;exit"
