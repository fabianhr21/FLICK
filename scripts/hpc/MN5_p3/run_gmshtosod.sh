#!/bin/bash
#SBATCH -J p3_{{BASENAME}}   # Job name

#SBATCH --time=02:00:00 # Max 24:00:00
#SBATCH -o out-%j.out   # STDOUT
#SBATCH -e out-%j.err   # STDERR

## Run configuration
## Rule: {ntasks} \times {cpus-per-task} = 40
## Rule: --gres=gpu:{ntasks}
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH -q gp_bsccase  # Partition associated to your user
#SBATCH -A bsc21

### MN% modules
module purge
#module load gcc openmpi hdf5 gmsh python
module load impi intel mkl hdf5 python

ln -sf ../{{BASENAME}}_Buildings.msh .
ln -sf ../witness.txt .
# ln toolmesh
# ln -sf /gpfs/scratch/upc76/fabian/sims_sod2d/GEO_CASES/sod2d_gitlab/msh_p3/tool_meshConversorPar/tool_meshConversorPar . 
# ln -sf /gpfs/scratch/upc76/fabian/sims_sod2d/GEO_CASES/sod2d_gitlab/run_p3/src/app_sod2d/sod2d .

ln -sf /gpfs/scratch/bsc21/bsc084826/SOD2D/sod2d_gitlab/msh_p3/tool_meshConversorPar/tool_meshConversorPar . 
ln -sf /gpfs/scratch/bsc21/bsc084826/SOD2D/sod2d_gitlab/run_p3/src/app_sod2d/sod2d .

gmsh {{BASENAME}}_Buildings_p3.geo -save
python3 gmsh2sod2d.py {{BASENAME}}_Buildings_p3 -r 3 -m 8,x,100 -p 11 -s 21000000

## GMSHTOSOD
module purge
module load openmpi/4.1.5-gcc ucx/1.15.0-gcc hdf5/1.14.1-2-gcc-openmpi
# Run
date
# Do not forget to change your script
# use as many tasks per node in mpirun as gpu to target
mpirun -n 12 tool_meshConversorPar input_file.dat
date
sbatch interpol.sh