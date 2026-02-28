#!/bin/bash
#SBATCH -J p2_{{BASENAME}}   # Job name
#SBATCH --time=02:00:00 # Max 24:00:00
#SBATCH -o out-%j.out   # STDOUT
#SBATCH -e out-%j.err   # STDERR

## Run configuration
## Rule: {ntasks} \times {cpus-per-task} = 40
## Rule: --gres=gpu:{ntasks}
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH --qos=gp_debug
#SBATCH -A bsc21

### MN% modules
module purge
#module load gcc openmpi hdf5 gmsh python
module load impi intel mkl hdf5 python

ln -sf ../{{BASENAME}}_Buildings.msh .
ln -sf /gpfs/scratch/bsc21/bsc084826/SOD2D/sod2d_gitlab/msh_p2/tool_meshConversorPar/tool_meshConversorPar .
ln -sf /gpfs/scratch/bsc21/bsc084826/SOD2D/sod2d_gitlab/run_p2/src/app_sod2d/sod2d .

gmsh {{BASENAME}}_Buildings_p2.geo -save
python3 gmsh2sod2d.py {{BASENAME}}_Buildings_p2 -r 2 -m 8,x,100 -p 11 -s 21000000

module purge
module load openmpi/4.1.5-gcc ucx/1.15.0-gcc hdf5/1.14.1-2-gcc-openmpi
# Do not forget to change your script
# use as many tasks per node in mpirun as gpu to target
mpirun -n 4 tool_meshConversorPar input_file.dat