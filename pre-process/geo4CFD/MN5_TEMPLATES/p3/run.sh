#!/bin/bash

### Job name on queue
#SBATCH --job-name=p3_{{BASENAME}}

### Output and error files directory
#SBATCH -D .

### Output and error files
#SBATCH --output=mpi_%j.out
#SBATCH --error=mpi_%j.err

### Run configuration
### Rule: {ntasks-per-node} \times {cpus-per-task} = 80
#SBATCH --nodes=3
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:4
#SBATCH --exclusive

### Queue and account
#SBATCH --qos=acc_resa
#SBATCH --account=upc76

### MN% modules
module purge
module load nvidia-hpc-sdk/24.3 hdf5/1.14.1-2-nvidia-nvhpcx

mpirun -np 12 --map-by ppr:4:node:PE=20 --report-bindings ./mn5_bind.sh  ./sod2d WindFarmSolverIncomp
# --report-bindings ./mn5_bind.sh 
