#!/bin/bash

#SBATCH --job-name=interp_{{BASENAME}}   # Job name
#SBATCH --chdir=.
#SBATCH --output=meshInter_%j.out
#SBATCH --error=meshInter_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=112
#SBATCH --time=02:00:00
#SBATCH --qos=gp_resc
#SBATCH --account=upc76

module purge
#module load openmpi/4.1.5-gcc ucx/1.15.0-gcc hdf5/1.14.1-2-gcc-openmpi cmake
module load impi intel mkl hdf5 python
unset LDFLAGS

MSH_P2=../p2/{{BASENAME}}_Buildings_p2-4.hdf
MSH_P3=./{{BASENAME}}_Buildings_p3-12.hdf
RESTART_P2=../p2/restart_{{BASENAME}}_Buildings_p2-4_2.h5
RESTART_P3=./restart_{{BASENAME}}_Buildings_p3-12_2.h5

mpirun -np 1 python3 interpolate.py "${MSH_P2}" "${MSH_P3}" "${RESTART_P2}" "${RESTART_P3}"
sbatch run.sh