#SBATCH -p m610
#SBATCH -N1
#SBATCH -C64
srun -N1 -n1 main.sh &
srun -N1 -n1 small_heatmaps.sh

