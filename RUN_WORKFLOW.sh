#!/bin/bash
#SBATCH --job-name=WORKFLOW
#SBATCH --output=WORKFLOW_%j.out
#SBATCH --error=WORKFLOW_%j.err
#SBATCH --ntasks=12
#SBATCH --time=02:00:00
#SBATCH --qos=gp_debug
#SBATCH --account=bsc21
#SBATCH --mail-type=END
#SBATCH --mail-user=fabian.hernandez@bsc.es

# module load intel/2023.2.0 impi/2021.10.0 oneapi/2023.2.0 hdf5/1.14.1-2 python/3.12.1
# Activate your virtual environment
# source $VENV

# # Set the number of CPUs per task
# export SRUN_CPUS_PER_TASK=224  # Set the desired number of CPUs per task

# Run the Python script with srun
cd Pre-Process
srun python ./STL2GeoTool_loop.py
cd ../Wind-NN
srun -n 1 python inference-script.py
cd ../Post-Process
srun -n 1 python overlap.py
