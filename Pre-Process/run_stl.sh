#!/bin/bash
#SBATCH --job-name=STL2GEO
#SBATCH --output=STL2GEO_%j.out
#SBATCH --error=STL2GEO_%j.err
#SBATCH --ntasks=224
#SBATCH --time=02:00:00
#SBATCH --qos=gp_bsccase
#SBATCH --account=bsc21
#SBATCH --mail-type=END
#SBATCH --mail-user=fabian.hernandez@bsc.es

# Activate your virtual environment
source $VENV

# # Set the number of CPUs per task
# export SRUN_CPUS_PER_TASK=224  # Set the desired number of CPUs per task

# Run the Python script with srun
srun python ./STL2GeoTool_loop.py