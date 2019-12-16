#!/usr/bin/env bash
#SBATCH -p m610
#SBATCH -J test_job
#SBATCH -n 2
#SBATCH -c 16
#SBATCH --mail-type=END,FAIL       # email status changes
srun -N1 -n1 -c16 --slurmd-debug=4 -o main.txt main.sh &
srun -N1 -n1 -c16 --slurmd-debug=4 --mail-user -o small_heatmaps.txt small_heatmaps.sh

