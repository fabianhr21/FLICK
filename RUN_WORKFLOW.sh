#!/bin/bash
#SBATCH --job-name=WORKFLOW
#SBATCH --output=WORKFLOW_%j.out
#SBATCH --error=WORKFLOW_%j.err
#SBATCH --nodes=6
#SBATCH --ntasks=672
#SBATCH --time=06:00:00
#SBATCH --qos=gp_bsccase
#SBATCH --account=bsc21

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 basename"
    exit 1
fi

BASENAME=$1
real_path=$(realpath "$BASENAME")
STEP_SIZE=128
input_xydim=$((STEP_SIZE * 4))
echo "Input xdim and ydim: $input_xydim"

# Run the Python script with srun
cd pre-process/
python ./STL2GeoTool.py -stl_basename "${BASENAME}" -step_size $STEP_SIZE -stl_dir "../" 
cd ../wind-nn/
mpirun -n 1 python new-inference-script.py -data_sample_basename "$BASENAME" -input_xdim $input_xydim -input_ydim $input_xydim 