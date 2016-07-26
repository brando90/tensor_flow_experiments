#!/bin/bash
#SBATCH --job-name=HBF2_6_6
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1-500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

python main_nn.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID HBF2_6_6_multiple_S_random_HP_search_H_500 True om_results multiple_S_hbf2_depth_2_500 6,6 multiple_S f_2d_task2_xsinglog1_x_depth2
