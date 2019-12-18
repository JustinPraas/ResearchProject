#!/usr/bin/env bash
#SBATCH -p r930
#SBATCH -J test_job
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s1782215

module load anaconda3

echo | ./generate_small_datasets.sh

