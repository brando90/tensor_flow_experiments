#!/bin/bash
#SBATCH --job-name=KRLS
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

python krls_collect_data.py f_2D_task2 ./tmp_result_krls_f_2D_task2_results 50 50 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34

python krls_collect_data.py f_2D_task2 ./tmp_result_krls_f_2D_task2_results 3 3 2,3,4
