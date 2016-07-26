#!/bin/bash
#SBATCH --job-name=KRLS
#SBATCH --nodes=1
#SBATCH --mem=7000
#SBATCH --time=6-23
#SBATCH --array=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene_sax14@yahoo.com

python krls_collect_data.py MNIST_flat om_krls krls_MNIST_flat_50_50_units_6_12_24_48_96_182_246_363 2 2 6,12,24,48,96,182,246,363
