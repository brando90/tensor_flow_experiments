#!/bin/bash
#SBATCH --job-name=HBF1_6_multiple_S_random_HP
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1-25
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID HBF1_12_multiple_S_random_HP_task2_test True om_results multiple_S 2 multiple_S f_2D_task2
