#!/bin/bash
#SBATCH --job-name=HBF1_model_train
#SBATCH --nodes=1
#SBATCH --mem=4500
#SBATCH --time=6-23
#SBATCH --array=1-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
matlab -nodesktop -nosplash -nojvm -r "get_best_trained_hbf1_model( $SLURM_JOBID, $SLURM_ARRAY_TASK_ID);exit"
