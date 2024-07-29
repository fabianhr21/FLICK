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

# Run the Python script with srun
cd Pre-Process
srun python ./STL2GeoTool_loop.py -stl_basename "$BASENAME"
srun -n 1 python ./ADD_FEAT.py -stl_basename "$BASENAME"
cd ../Wind-NN
srun -n 1 python inference-script.py -data_sample_basename "$BASENAME"
cd ../Post-Process
srun -n 1 python overlap.py -basename "$BASENAME"