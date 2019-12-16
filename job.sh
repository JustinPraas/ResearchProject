#SBATCH -p m610
#SBATCH -N1
#SBATCH -C16
srun -N1 -n1 -c16 main.sh &
srun -N1 -n1 -c16 small_heatmaps.sh

