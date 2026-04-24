# FLICK

[![CI](https://github.com/fabianhr21/FLICK/actions/workflows/python-tests.yml/badge.svg)](https://github.com/fabianhr21/FLICK/actions/workflows/python-tests.yml)
[![Docs](https://github.com/fabianhr21/FLICK/actions/workflows/docs.yml/badge.svg)](https://github.com/fabianhr21/FLICK/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

**FLICK** (Fast Modelling of Urban Climate) predicts urban wind fields from mesoscale weather models and urban geometry using a U-Net neural network. Developed in collaboration with the [Barcelona Supercomputing Center](https://www.bsc.es/) and the Universitat Politècnica de Catalunya.

## Pipeline

```
STL/BIM geometry
      ↓
[flick_urban.preprocess]  →  H5 file (MASK, HEIGHT, WIND_DIST features)
      ↓
[flick_urban.nn]          →  normalized U, V velocity fields (U-Net)
      ↓
[flick_urban.postprocess] →  composited wind field + visualizations
```

## Installation

**1. Clone with submodules**

```bash
git clone https://github.com/fabianhr21/FLICK.git
cd FLICK
git submodule update --init --recursive
```

**2. Install Python package**

```bash
pip install -e .             # base
pip install -e .[gpu]        # + PyTorch/CUDA
pip install -e .[hpc]        # + MPI (mpi4py)
pip install -e .[gpu,hpc]    # full
pip install -e .[dev]        # + pytest (for development)
```

**3. Compile external tools**

```bash
bash scripts/compile_tools.sh
```

Installs system dependencies, compiles City4CFD, and creates the `city4cfd` symlink.

## Model Weights

The neural network weights are **not included** in this repository.  
Request them from: fabian.hernandez@bsc.es  
Expected location after receiving: `170625_weights/`

## Geometry Sources

BIM models (recommended — contain geometry and metadata):
- Catalonia: https://geoportalcartografia.amb.cat/AppGeoportalCartografia2/index.html
- Spain: https://centrodedescargas.cnig.es/CentroDescargas/buscar-mapa

City4CFD can generate CFD domains from LiDAR data. The workflow targets the SOD2D solver.

<img width="1730" height="223" alt="image" src="https://github.com/user-attachments/assets/afcbd7ca-1ed1-4127-936d-507c983230f4" />

## Running the Workflow

### Locally

```bash
# Pre-process STL geometry
python -c "from flick_urban.preprocess.stl2geo import main; main()" --input grid_of_cubes.stl

# Inference (requires model weights)
python -c "from flick_urban.nn.inference import main; main()" --data_sample_basename name

# Post-process
python -c "from flick_urban.postprocess.overlap import main; main()"
```

### HPC (MareNostrum 5 / SLURM)

```bash
sbatch scripts/RUN_WORKFLOW.sh grid_of_cubes
```

Adjust SBATCH parameters in the script. SLURM templates for MN5 partitions p2/p3 are in `scripts/hpc/`.

## Package Structure

```
flick_urban/
├── preprocess/       # STL/BIM → H5 feature extraction
│   └── geo4cfd/      # City4CFD integration + ANSA meshing automation
├── nn/               # U-Net model, inference
└── postprocess/      # tile stitching, velocity maps, visualization
```

## External Dependencies (submodules)

| Submodule | URL | Purpose |
|-----------|-----|---------|
| City4CFD | https://github.com/tudelft3d/City4CFD | 3D urban geometry from LiDAR + 2D polygons |
| pyqvarsi | https://gitlab.com/ArnauMiro/pyQvarsi | CFD post-processing library |

## Contact

fabian.hernandez@bsc.es
