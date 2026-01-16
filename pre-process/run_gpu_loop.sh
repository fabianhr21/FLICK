#!/bin/bash
#SBATCH --job-name=mGPU_STL2Geo
#SBATCH --output=out_%j.out
#SBATCH --error=out_%j.err

# Request enough GPUs for all angles (example: 12 GPUs)
#SBATCH --nodes=8
#SBATCH --ntasks=8                # one task per angle (tweak if needed)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --qos=acc_debug
#SBATCH --account=bsc21
#SBATCH --gres=gpu:1              # request total GPUs (must match number of angles)

module purge
module load oneapi/2024.0 hdf5 python/3.12.1
source /gpfs/scratch/bsc21/bsc084826/envs/jm_venv/bin/activate

ulimit -u 40494

# ANGLES=("112-5Neg" "22-5Pos" "45Pos" "67-5Pos" "90Neg" "135Neg" "135Pos" "180deg" "67-5Neg" "22-5Neg" "45Neg" "90Pos")
ANGLES=(
    "0.0"    # 0deg
    "22.5"   # 22-5Pos
    "45.0"   # 45Pos
    "67.5"   # 67-5Pos
    "90.0"   # 90Pos
    "112.5"  # 112-5Pos
    "135.0"  # 135Pos
    "157.5"  # 157-5Pos
    # "180.0"  # 180deg
    # "202.5"  # 157-5Neg
    # "225.0"  # 135Neg
    # "247.5"  # 112-5Neg
    # "270.0"  # 90Neg
    # "292.5"  # 67-5Neg
    # "315.0"  # 45Neg
    # "337.5"  # 22-5Neg
)

# [ "0", "22.5", "45", "67.5", "90", "112.5", "135","157.5", "180","202.5", "225", "247.5", "270", "292.5", "315", "337.5" ]
SCRIPT="STL2GeoTool.py"
STL_BASENAME="sanjeronimo"
N_POINTS=2048

date

i=0
for angle in "${ANGLES[@]}"; do
    echo ">>> Launching angle $angle on GPU task $i"
    srun -n 1 --nodes=1 --gres=gpu:1 --cpus-per-task=80 --exclusive python "$SCRIPT" -wind_direction $angle -stl_basename "$STL_BASENAME" -n_points $N_POINTS &
    ((i++))
done

wait   # wait for all background srun jobs to finish
date
