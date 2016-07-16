#!/bin/bash
#SBATCH --job-name=HBF12_multiple_S
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1-200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

python main_nn.py $SLURM_ARRAY_TASK_ID $SLURM_JOBID HBF1_12_multiple_S_random_HP True om_results

