#!/bin/bash
#SBATCH --job-name=KRLS_TEST
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

python krls_collect_data.py f_2D_task2 ./test_result_krls_f_2D_task2_results 2 2 2,3,4
