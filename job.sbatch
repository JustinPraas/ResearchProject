#!/usr/bin/env bash
#SBATCH -p m610
#SBATCH -J test_job
#SBATCH -n 2
#SBATCH -c 16
#SBATCH --mail-type=END,FAIL       # email status changes

module load anaconda3

srun -N1 -n1 -c16 small_heatmaps.sh

