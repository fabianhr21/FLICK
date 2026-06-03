# FLICK

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://doi.org/10.5281/zenodo.20526909)

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
}
```

## Of the current work
Until now, the wind neural network has demonstrated good accuracy in the dataset, as Calafell et al.1 describe in their publication. Furthermore, The model has proven to be easy and adaptable to different geometries and contextual windows.

As part of the new extension of the research and applications, we have found that the Neural Network can make inferences on much bigger domains than the ones it was trained on. As shown in the next picture.
image

<img width="486" height="390" alt="image" src="https://github.com/user-attachments/assets/0ae35912-8368-4222-9311-be7a9e5ce450" />

Furthermore, a comparison was made the comparission between high-fidelity simulations performed with SOD2D. The solver has been
validated over idealized<sup>3</sup> and realistic<sup>4,</sup><sup>5</sup> urban configurations.
<img width="1790" height="427" alt="image" src="https://github.com/user-attachments/assets/7e04ff63-facd-4ece-aea3-a28974124ec8" />

Even though the structure of the field seems very similar, the average error is ~40%, and the error is attributed to the complete change in the geometry, as the unseen geometry differs from the geometries on which the model was trained. Currently, some extensive work is being done on improving the dataset and obtaining a more comprehensive dataset.

[1] Calafell Sandiumenge, Joan and Bustillo, Jaime and Mateu Armengol, Jan and Gómez, Samuel and Ramírez Jávega, Francisco and Lehmkuhl, Oriol, Building a General and Data-Efficient Convolutional Neural Network-Based Model for Fast Urban Flow Estimation. Available at SSRN: https://ssrn.com/abstract=5970896 or http://dx.doi.org/10.2139/ssrn.5970896

[2] Teng, M., Duró Diaz, J. M., Mestres, E., Muela Castro, J., Lehmkuhl, O., & Rodriguez, I. (2025). Atmospheric boundary layer over urban roughness: Validation of large-eddy simulation. Physics of Fluids, 37(6). https://pubs.aip.org/aip/pof/article/37/6/065129/3349087

[3] Teng, M., Duro, J. M., Munoz, N., Mestres, E., Muela, J., Lehmkuhl, O., & Rodriguez, I. (2025). Toward high-fidelity simulations of urban flows: mean-flow statistics. In ICHMT DIGITAL LIBRARY ONLINE. Begel House Inc. https://www.dl.begellhouse.com/references/1bb331655c289a0a,7224fbfa56c3c688,526111d85f38377b.html

[4] Rodríguez, I., Duró, J. M., Mestres, E., Teng, M., & Lehmkuhl, O. (2025). Impact of Wind Direction on Flow Over a Realistic Urban Area: A Large-Eddy Simulation Study. arXiv preprint arXiv:2510.11247. https://arxiv.org/abs/2510.11247

## Contact

fabian.alexis.hernandez@upc.edu
