# FLICK

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20526909.svg)](https://doi.org/10.5281/zenodo.20526909)

FLICK (Fast and Light Inference for Climate Knowledge) is an open-source Python package for rapid urban wind field prediction. It combines a geometry preprocessing pipeline, a U-Net neural network surrogate model, and a tile-stitching postprocessor into a single installable package (flick_urban).

Geo4CFD converts any urban LiDAR point cloud or BIM model into a CFD-ready surface mesh. It leverages City4CFD for building reconstruction from OSM footprints and LiDAR data, and drives ANSA to automatically generate SOD2D-compatible domain geometry with appropriate boundary conditions, size boxes, and precursor domain.

Neural network pipeline takes the preprocessed geometry — encoded as 256×256 raster feature maps (building mask, height field, wind distance) — and predicts normalized horizontal wind velocity components (U, V) using a Pix2Pix-based Generator. Inputs tile the domain at configurable resolution; outputs are composited via exponential-decay overlap stitching to produce city-scale wind fields.

Postprocessor reconstructs the full-domain wind field from overlapping tiles and computes velocity magnitude and direction maps ready for visualization or downstream analysis.

The package targets urban climate researchers and city planners who need fast wind environment assessments without running full CFD simulations. Developed at Universitat Politècnica de Catalunya (UPC) and the Barcelona Supercomputing Center (BSC).

## Fast and Light Inference for Climate Knowledge
FLICK is a designed tool to help in preprocessing for both simulation and some ready machine learning tools inference. Please check the [wiki](https://github.com/fabianhr21/FLICK/wiki) for instructions on how to deploy and contribute to the tool. Here you can find the code documentation and examples on every module.

## Cite this repo!

```bash
@misc{FLICK,
  author    = {Hernández, Fabián and Duró, Josep and Miro, Arnau and Lehmkuhl, Oriol and Rodríguez, Ivette},
  title     = {FLICK: Fast and Light Inference for Climate Knowledge},
  year      = {2026},
  publisher = {Universitat Politecnica de Catalunya},
  journal   = {GitHub repository},
  url       = {https://github.com/fabianhr21/FLICK/},
  doi       = {https://doi.org/10.5281/zenodo.20526909}
}
```

## Acknowledgements

The research leading to this software was financially supported by project 'Under the skin of the city: Urban simulations for nature-based solutions', funded by the Agència de Gestió d'Ajuts Universitaris i de Recerca (AGAUR) under the call PROJECTES DE RECERCA PER A LA MITIGACIÓ I ADAPTACIÓ AL CANVI CLIMÀTIC, with grant agreement No 2023 CLIMA 00097

## Contact

For any inquiries regarding the project, contact ivette.rodriguez@upc.edu; for inquiries regarding the code or functionality, contact fabian.alexis.hernandez@upc.edu
