#SBATCH --partition=main
#SBATCH -N32
srun -N1 -n1 job-step.sh
