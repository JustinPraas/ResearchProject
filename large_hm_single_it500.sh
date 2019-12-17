#!/usr/bin/env bash
#SBATCH -p m610
#SBATCH -J test_job
#SBATCH -N 8
#SBATCH --mail-type=END,FAIL       # email status changes
#SBATCH --mail-user -u

module load anaconda3

srun -N8 -n1 -c=$SLURM_CPUS_ON_NODE --output=large_hm_single_it500_RFF.txt large_hm_single_it500_RFF.sh

