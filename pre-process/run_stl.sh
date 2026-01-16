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
#SBATCH --ntasks=20
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

module purge
module load oneapi/2024.0 hdf5 python/3.12.1
# source  /gpfs/scratch/bsc21/bsc084826/envs/40_venv/bin/activate
# source /gpfs/scratch/bsc21/bsc084826/envs/venv/bin/activate
source /gpfs/scratch/bsc21/bsc084826/envs/jm_venv/bin/activate
# export HDF5_USE_FILE_LOCKING=FALSE
SCRIPT="STL2GeoTool.py"
STL_BASENAME="caso6_1420"
STEP_SIZE=710
PX_RESOLUTION=0.5
batch_size=24576
STL_DIR='/gpfs/scratch/bsc21/bsc084826/cities/bettair_cities/'
date
# Run the Python script with srun
# srun -n 1 python ./STL2GeoTool.py -wind_direction 0 -stl_basename "bcn_2560" -step_size 1280 -batch_size 24576
srun -n 1 python ./${SCRIPT} -wind_direction 0 -stl_basename ${STL_BASENAME} -step_size ${STEP_SIZE} -px_resolution ${PX_RESOLUTION} -batch_size ${batch_size} -stl_dir ${STL_DIR}
date