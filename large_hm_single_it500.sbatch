#!/usr/bin/env bash
#SBATCH -p m610
#SBATCH -J test_job
#SBATCH -n 6
#SBATCH -c 16
#SBATCH --mail-type=END,FAIL       # email status changes
#SBATCH --mail-user

module load anaconda3

srun -N1 -n1 -c16 --output=large_hm_betweenness_it500_RFF.txt large_hm_betweenness_it500_RFF.sh
srun -N1 -n1 -c16 --output=large_hm_closeness_it500_RFF.txt large_hm_closeness_it500_RFF.sh
srun -N1 -n1 -c16 --output=large_hm_degree_it500_RFF.txt large_hm_degree_it500_RFF.sh
srun -N1 -n1 -c16 --output=large_hm_katz_it500_RFF.txt large_hm_katz_it500_RFF.sh
srun -N1 -n1 -c16 --output=large_hm_pagerank_it500_RFF.txt large_hm_pagerank_it500_RFF.sh
srun -N1 -n1 -c16 --output=large_hm_eigenvector_it500_RFF.txt large_hm_eigenvector_it500_RFF.sh

