#!/usr/bin/env bash
#SBATCH -p r930
#SBATCH -J test_job
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --mail-type=END,FAIL       # email status changes
#SBATCH --mail-user -u

module load anaconda3

echo | ./large_hm_single_it500_RFF.sh

