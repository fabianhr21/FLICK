# FMUC

FMUC - Fast Modeling Urban Climate is a package developed in collaboration with the Barcelona Supercomputing Center and the Universitat Politècnica de Catalunya to model wind at urban scales using Neural Networks.

## Functionality
This repository is a guideline for generating urban wind analysis from mesoscale weather models and urban geometry.

## Pre-Process
The `pre-process` directory contains scripts to convert STL or BIM geometry into the format expected by the neural network. Geometry is georeferenced so that the wind fields can later be mapped back. BIM files already contain both information and are therefore recommended.

You can obtain BIM models from all of Catalunya at <https://geoportalcartografia.amb.cat/AppGeoportalCartografia2/index.html> and from all Spain at <https://centrodedescargas.cnig.es/CentroDescargas/buscar-mapa>.

The City4CFD project can also be used to generate CFD domains. The workflow provided here focuses on preparing geometry for simulation with the SOD2D solver.

## Wind-NN
The `wind-nn` folder hosts the surrogate neural network used to predict wind behaviour. The model outputs normalized velocity components.

## Post-Process
Scripts in `post-process` scale the predicted wind velocity and generate visualisations.

## Environment prerequisites
- **Python** 3.8 or newer with the packages listed in `requirements.txt`.
- **pyAlya** library (contact Arnau Miró at <arnau.mirojane@bsc.es> for access).
- A CUDA-capable GPU is recommended for running the neural network.
- Optional: an MPI environment for running the pre-processing scripts in parallel.

## Compiling City4CFD and pyQvarsi
All external tools can be built by running `compile_tools.sh` from the repository root:
```bash
bash compile_tools.sh
```
This installs the required dependencies with `apt`, compiles City4CFD inside `City4CFD/build` and creates a symlink in `pre-process/geo4CFD/` to the resulting `city4cfd` binary. To build pyQvarsi manually, run `make` inside the `pyqvarsi` directory.

## Executing the workflow
### Locally
Each step can be executed separately. Example commands using the provided `grid_of_cubes.stl` are:
```bash
# Pre-processing
mpirun -n 1 python pre-process/STL2GeoTool_loop.py -stl_basename grid_of_cubes
python pre-process/ADD_FEAT.py -stl_basename grid_of_cubes

# Inference
python wind-nn/inference-script.py -data_sample_basename grid_of_cubes

# Post-processing
python post-process/overlap.py -basename grid_of_cubes
```
### HPC clusters
The `RUN_WORKFLOW.sh` script wraps the entire process for SLURM clusters. Submit it with the STL base name as argument:
```bash
sbatch ./RUN_WORKFLOW.sh grid_of_cubes
```
Adjust the SBATCH parameters in the script to match your system configuration.

## Directory overview
- **pre-process** – geometry preparation tools and scripts.
- **wind-nn** – the trained neural network and inference code.
- **post-process** – utilities to compose the final wind field output.
- **City4CFD** – source code of the City4CFD geometry generator.
- **pyqvarsi** – post‑processing library used by the workflow.


