#!/bin/bash
#SBATCH --job-name=HBF1
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1-9
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

python test.py $SLURM_JOBID $SLURM_ARRAY_TASK_ID

