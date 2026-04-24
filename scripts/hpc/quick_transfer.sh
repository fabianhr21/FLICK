#!/bin/bash

DIR="/home/fabianh/GEO_CASES/round_2/"
CITIES=("VALENCIA" "SEVILLA" "ZARAGOZA") # "BARCELONA" "MADRID"
ansa_path=/home/fabianh/ANSA/BETA_CAE_Systems24.1/ansa_v24.1.2/ansa64.sh
working_directory="/home/fabianh/FLICK_untouched/pre-process/geo4CFD/ANSA_SCRIPTS/"

# MN5 setup
REMOTE_USER="bsc084826"
REMOTE_HOST="transfer1.bsc.es"
REMOTE_BASE="/gpfs/scratch/upc76/fabian/sims_sod2d/GEO_CASES"
SOCKET="/tmp/ssh_mux_${REMOTE_USER}_${REMOTE_HOST}"

# Open a single SSH master connection (authenticates once)
ssh -M -S "$SOCKET" -o ControlPersist=yes -N "${REMOTE_USER}@${REMOTE_HOST}" &
SSH_PID=$!
echo "Authenticating to ${REMOTE_HOST}... (complete the prompt if asked)"
until ssh -S "$SOCKET" -O check "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null; do
    sleep 1
done
echo "SSH master connection established."

for city in "${CITIES[@]}"; do
    for d in $DIR$city/*/ ; do
        # Ignore __pycache__ directories
        [[ "$d" == *"__pycache__"* ]] && continue
        # echo "Processing directory: $d"
    dirname=$(basename "$d")
    buildings_file="${d}output/${dirname}_Buildings.ansa"
    dimensions_file="${d}output/domain_dimensions.txt"
    echo "Using buildings file: $buildings_file"
    # echo "RUnning Path: '${d}output/'"
    # $ansa_path -nogui -noopencl -execscript "${working_directory}args_RbNDivide.py|main('${buildings_file}','${d}output/','${city}/','${dirname}')"
    SRC="${d}output/MN5/p3/"
    echo "Sending MN5 directory to remote host: ${SRC}"
    rsync --progress -e "ssh -S $SOCKET" -r "${SRC}"*gmshtosod.sh "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/${city}/${dirname}/p3/"
    # break
    done
done
ssh -S "$SOCKET" -O exit "${REMOTE_USER}@${REMOTE_HOST}"
echo "All done. Closed SSH connection to ${REMOTE_HOST}."

# FOr makinf the precursor and split2hexa
#~/ANSA/BETA_CAE_Systems24.1/ansa_v24.1.2/ansa64.sh -nogui -noopencl -execscript "RbNDivide.py|main('/home/fabianh/GEO_CASES/BARCELONA/267-43/output/267-43_Buildings.ansa','267-43_Buildings','/home/fabianh/GEO_CASES/BARCELONA/267-43/output/')"
