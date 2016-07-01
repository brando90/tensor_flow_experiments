#!/bin/bash
#SBATCH --job-name=HBF1_MNIST
#SBATCH --nodes=1
#SBATCH --mem=5500
#SBATCH --time=6-23
#SBATCH --array=1-20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com
#SBATCH --gres=gpu:1

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
matlab -nodesktop -nosplash -nojvm -r "get_best_trained_hbf1_model( $SLURM_JOBID, $SLURM_ARRAY_TASK_ID);exit"
