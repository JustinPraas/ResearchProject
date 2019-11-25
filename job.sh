#SBATCH --partition=caserta
#SBATCH -N32
#SBATCH -C100
srun -N1 -n1 -c65 cross_validation.sh
