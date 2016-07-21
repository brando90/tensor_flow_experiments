#!/bin/bash
#SBATCH --job-name=HBF1_6_multiple_S_random_HP
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1-25
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID HBF1_6_multiple_S_random_HP_test True om_results multiple_S_test 6
