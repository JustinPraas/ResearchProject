#SBATCH -p m610
#SBATCH -N1
#SBATCH -C16
srun -N1 -n1 -c64 -slurmd-debug=4 -o main.txt main.sh &
srun -N1 -n1 -c64 -slurmd-debug=4 -o small_heatmaps.txt small_heatmaps.sh

