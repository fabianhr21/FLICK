#!/bin/bash
# rbndivide_loop.sh — Loop RbNDivide over city directories and sync to HPC.
#
# Configure the variables below before use:
#   REMOTE_USER  : your HPC username
#   REMOTE_HOST  : HPC transfer node hostname
#   REMOTE_BASE  : base path on HPC scratch
#   DIR          : local directory containing city case folders
#   ansa_path    : path to the ANSA executable

DIR="/path/to/GEO_CASES/"
CITIES=("CITY_A" "CITY_B")
ansa_path="/path/to/ansa64.sh"
working_directory="/path/to/FLICK/flick_urban/preprocess/geo4cfd/ansa/"

# HPC setup — fill in your credentials
REMOTE_USER="${FLICK_HPC_USER:-your_hpc_username}"
REMOTE_HOST="${FLICK_HPC_HOST:-transfer.hpc.example.org}"
REMOTE_BASE="${FLICK_HPC_SCRATCH:-/scratch/user/sims_sod2d/GEO_CASES}"
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
    for d in $DIR$city/*/; do
        [[ "$d" == *"__pycache__"* ]] && continue
        dirname=$(basename "$d")
        buildings_file="${d}output/${dirname}_Buildings.ansa"
        echo "Using buildings file: $buildings_file"
        SRC="${d}output/MN5/p3/WindFarmSolverIncomp.json"
        echo "Sending MN5 directory to remote host: ${SRC}"
        rsync --progress -e "ssh -S $SOCKET" -r "${SRC}" \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/${city}/${dirname}/p3/"
    done
done
ssh -S "$SOCKET" -O exit "${REMOTE_USER}@${REMOTE_HOST}"
echo "All done. Closed SSH connection to ${REMOTE_HOST}."
