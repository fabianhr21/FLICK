# FLICK

[![CI](https://github.com/fabianhr21/FLICK/actions/workflows/python-tests.yml/badge.svg)](https://github.com/fabianhr21/FLICK/actions/workflows/python-tests.yml)
[![Docs](https://github.com/fabianhr21/FLICK/actions/workflows/docs.yml/badge.svg)](https://github.com/fabianhr21/FLICK/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

**FLICK** (Fast and Light Inference for Climate Knowledge) predicts urban wind fields from mesoscale weather models and urban geometry using a U-Net neural network. Developed in collaboration with the [Barcelona Supercomputing Center](https://www.bsc.es/) and the Universitat Politècnica de Catalunya.

## Fast and Light Inference for Climate Knowledge
FLICK is a designed tool to help in preprocessing for both simulation and some ready machine learning tools inference. Please check the wiki for instructions on how to deploy and contribute to the tool. Here you can find the code documentation and examples on every module.

## Of the current work
Until now, the wind neural network has demonstrated good accuracy in the dataset, as Calafell et al.1 describe in their publication. Furthermore, The model has proven to be easy and adaptable to different geometries and contextual windows.

As part of the new extension of the research and applications, we have found that the Neural Network can make inferences on much bigger domains than the ones it was trained on. As shown in the next picture.
image

<img width="486" height="390" alt="image" src="https://github.com/user-attachments/assets/0ae35912-8368-4222-9311-be7a9e5ce450" />

Furthermore, a comparison was made the comparission between high-fidelity simulations performed with SOD2D<sup>2</sup>. The solver has been
validated over idealized<sup>3</sup> and realistic<sup>4,</sup><sup>5</sup> urban configurations.
<img width="1790" height="427" alt="image" src="https://github.com/user-attachments/assets/7e04ff63-facd-4ece-aea3-a28974124ec8" />

Even though the structure of the field seems very similar, the average error is ~40%, and the error is attributed to the complete change in the geometry, as the unseen geometry differs from the geometries on which the model was trained. Currently, some extensive work is being done on improving the dataset and obtaining a more comprehensive dataset.

[1] Calafell Sandiumenge, Joan and Bustillo, Jaime and Mateu Armengol, Jan and Gómez, Samuel and Ramírez Jávega, Francisco and Lehmkuhl, Oriol, Building a General and Data-Efficient Convolutional Neural Network-Based Model for Fast Urban Flow Estimation. Available at SSRN: https://ssrn.com/abstract=5970896 or http://dx.doi.org/10.2139/ssrn.5970896

[2] Gasparino, L., Spiga, F., & Lehmkuhl, O. (2024). SOD2D: A GPU-enabled spectral finite elements method for compressible scale-resolving simulations. Computer Physics Communications, 297, 109067. https://www.sciencedirect.com/science/article/pii/S0010465523004125

[3] Teng, M., Duró Diaz, J. M., Mestres, E., Muela Castro, J., Lehmkuhl, O., & Rodriguez, I. (2025). Atmospheric boundary layer over urban roughness: Validation of large-eddy simulation. Physics of Fluids, 37(6). https://pubs.aip.org/aip/pof/article/37/6/065129/3349087

[4] Teng, M., Duro, J. M., Munoz, N., Mestres, E., Muela, J., Lehmkuhl, O., & Rodriguez, I. (2025). Toward high-fidelity simulations of urban flows: mean-flow statistics. In ICHMT DIGITAL LIBRARY ONLINE. Begel House Inc. https://www.dl.begellhouse.com/references/1bb331655c289a0a,7224fbfa56c3c688,526111d85f38377b.html

[5] Rodríguez, I., Duró, J. M., Mestres, E., Teng, M., & Lehmkuhl, O. (2025). Impact of Wind Direction on Flow Over a Realistic Urban Area: A Large-Eddy Simulation Study. arXiv preprint arXiv:2510.11247. https://arxiv.org/abs/2510.11247

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
