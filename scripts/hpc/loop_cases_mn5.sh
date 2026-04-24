#!/bin/bash
#SBATCH --job-name=stl2geotool
#SBATCH --output=stlout_%j.out
#SBATCH --error=stlout_%j.err
## SBATCH --ntasks-per-node=80

#SBATCH --qos=acc_debug
#SBATCH --account=bsc21

### Run configuration
### Rule: {ntasks-per-node} \times {cpus-per-task} = 80
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4

module purge
module load oneapi/2024.0 hdf5 python/3.12.1
# source  /gpfs/scratch/bsc21/bsc084826/envs/40_venv/bin/activate
# source /gpfs/scratch/bsc21/bsc084826/envs/venv/bin/activate
source /gpfs/scratch/bsc21/bsc084826/envs/jm_venv/bin/activate
# export HDF5_USE_FILE_LOCKING=FALSE
cases=("caso00" "caso01" "caso02" "caso04" "caso05" "caso06" "caso07" "caso08" "caso10" "caso12" "caso14" "caso16" "caso17" "caso18" "caso19" "caso20" "caso21" "caso23" "caso26" "caso27" "caso30" "caso32" "caso36" "caso37" "caso42" "caso56" "caso59" "caso60" "caso69" "caso75")
angles=("0.0" "60.0" "120.0")
SCRIPT="STL2GeoTool.py"
STL_BASENAME="caso6"
STEP_SIZE=710
PX_RESOLUTION=0.5
# batch_size=24576
# batch_size=49152
batch_size=32768
STL_DIR='/gpfs/scratch/bsc21/bsc084826/cities/bettair_cities/'

date
i=0
# angle=${angles[0]}
for case in "${cases[@]}"; do
    for angle in "${angles[@]}"; do
        echo ">>> Processing case: $case at angle: $angle, launching on GPU task $i"
        STL_BASENAME="${case}_simplified"
        # Run the Python script with srun
        srun  -n 1 --nodes=1 --gres=gpu:1 --cpus-per-task=20 python ./${SCRIPT} -wind_direction $angle -stl_basename ${STL_BASENAME} -step_size ${STEP_SIZE} -px_resolution ${PX_RESOLUTION} -batch_size ${batch_size} -stl_dir ${STL_DIR} &
        ((i++))
    done
done

wait   # wait for all background srun jobs to finish
date