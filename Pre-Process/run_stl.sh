#!/bin/bash
#SBATCH --job-name=STL2GEO
#SBATCH --output=STL2GEO_%j.out
#SBATCH --error=STL2GEO_%j.err
#SBATCH --ntasks=224
#SBATCH --time=02:00:00
#SBATCH --qos=gp_bsccase
### Queue and account
#SBATCH --account=bsc21
#SBATCH --mail-type=end
#SBATCH --mail-user=fabian.hernandez@bsc.es


export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

mpirun python ./STL2GeoTool.py