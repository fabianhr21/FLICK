#!/bin/bash
# =============================================================================
# FLICK — Generic SLURM Job Template
# =============================================================================
# Copy this file, fill in the CONFIGURATION section, and submit with:
#
#   sbatch slurm_template.sh
#
# Tested on MareNostrum 5 (BSC). Adapt partition names, modules, and paths
# for your own HPC cluster.
# =============================================================================

# ---- SLURM directives -------------------------------------------------------
#SBATCH --job-name=flick_preprocess
#SBATCH --output=logs/flick_%j.out
#SBATCH --error=logs/flick_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1                 # remove if no GPU available
#SBATCH --time=02:00:00              # HH:MM:SS — adjust to your workload
#SBATCH --account=${SLURM_ACCOUNT}   # your HPC project account, e.g. bsc21
#SBATCH --partition=${SLURM_PARTITION}  # e.g. gpu, acc_debug, gp_bsc

# =============================================================================
# CONFIGURATION — edit these variables before submitting
# =============================================================================

# Path to your FLICK installation
FLICK_ROOT="${HOME}/FLICK"

# Python virtual environment (created with: python -m venv ${HOME}/flick_venv)
VENV="${HOME}/flick_venv"

# Input STL file
STL_DIR="${SCRATCH}/cities/my_city/"   # directory containing the STL
STL_BASENAME="my_city"                 # filename without .stl extension

# Output directory
OUTPUT_PATH="${SCRATCH}/flick_output/"

# Preprocessing parameters
WIND_DIRECTION=0          # degrees; 0 = wind from south
STEP_SIZE=640             # domain tile size in metres
PX_RESOLUTION=0.25        # spatial resolution in metres
P_OVERLAP=1               # 1 = no overlap; < 1 adds overlap between tiles
USE_GPU=True              # True / False

# =============================================================================
# SETUP
# =============================================================================

set -e
mkdir -p logs "${OUTPUT_PATH}"

# Load HPC modules — adapt module names to your cluster
# module load nvidia-hpc-sdk        # for GPU / CUDA
# module load openmpi               # for MPI
# module load hdf5                  # for HDF5

# Activate Python environment
source "${VENV}/bin/activate"

cd "${FLICK_ROOT}"

echo "========================================================"
echo "FLICK preprocessing"
echo "  STL:          ${STL_DIR}${STL_BASENAME}.stl"
echo "  Output:       ${OUTPUT_PATH}"
echo "  Wind angle:   ${WIND_DIRECTION} deg"
echo "  Resolution:   ${PX_RESOLUTION} m"
echo "  Nodes:        ${SLURM_NNODES}, Tasks: ${SLURM_NTASKS}"
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "========================================================"

# =============================================================================
# STEP 1 — Preprocessing (STL → HDF5 feature maps)
# =============================================================================

mpirun -n "${SLURM_NTASKS}" python -m flick_urban.preprocess.stl2geo \
    -stl_dir       "${STL_DIR}" \
    -stl_basename  "${STL_BASENAME}" \
    -output_path   "${OUTPUT_PATH}" \
    -wind_direction "${WIND_DIRECTION}" \
    -step_size      "${STEP_SIZE}" \
    -px_resolution  "${PX_RESOLUTION}" \
    -p_overlap      "${P_OVERLAP}" \
    -use_gpu        "${USE_GPU}"

echo "Preprocessing done."

# =============================================================================
# STEP 2 — Neural network inference
# =============================================================================
# Requires model weights — see model_weights/README.md
#
# python -m flick_urban.nn.inference \
#     -dataset_base_path "${OUTPUT_PATH}" \
#     -data_sample_basename "${STL_BASENAME}" \
#     -output_path "${OUTPUT_PATH}/nn_output/" \
#     -model_loading_path "${FLICK_ROOT}/model_weights/" \
#     -model_basename generator

# =============================================================================
# STEP 3 — Post-processing (stitch tiles → full wind field)
# =============================================================================
#
# python -m flick_urban.postprocess.overlap \
#     -dataset_path "${OUTPUT_PATH}/nn_output/output${WIND_DIRECTION}-${STL_BASENAME}/" \
#     -basename "${STL_BASENAME}" \
#     -wind_direction "${WIND_DIRECTION}"

echo "Job complete: ${SLURM_JOB_ID}"
