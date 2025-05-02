[![License](https://img.shields.io/badge/license-MIT-orange)](https://opensource.org/licenses/mit)

# Quick Visualization, Analysis and Reporting of SImulations Python module

The Quick Visualization, Analysis and Reporting of SImulations Python module (in short pyQvarsi) are a set of tools developed in python to interface with CFD codes using mpi-enabled tools to compute post-processing quantities in parallel.

This tool has been developed with Python 3 in mind and might not be compatible with lower version of Python.

The instructions on how to use pyQvarsi can be found in its [wiki](https://gitlab.com/ArnauMiro/pyQvarsi/-/wikis/home). In particular, on the left sidebar there are thorough instructions on how to build and deploy this tool on various platforms. The user is referred there for detailed instructions and also to find a bit of a manual on how pyQvarsi works and some examples of how it can be used.

pyQvarsi also includes some executables that run tools to perform usual operations. The user is refered [here](https://gitlab.com/ArnauMiro/pyQvarsi/-/wikis/Scripts) for a thorough explanation.

Finally, a number of examples are provided as a means to demostrate the capabilities of the tool:
* example_FEM: A little example to show the FEM operations on a silly mesh.
* example_FEM_parallel: A little example to show the FEM operations on the cavtri_03 case.
* example_COMMU: A little example to read and compute the communications matrix (used for validation).
* example_MASSM: A little example to read and compute the mass matrix (used for validation).
* example_output: A little example how to use the output formats of this tool.
* example_output_parallel: A little example how to use the output formats of this tool in parallel.
* example_avg_parallel: An example on how to compute temporal averages and reduce with the tool.
* example_avgXZ_parallel: An example on how to compute temporal averages, average on the X and Z direction, reduce and compute the BL statistics from a channel flow using VELOC.
* example_avgXZ_AVVEL_parallel: An example on how to compute temporal averages, average on the X and Z direction, reduce and compute the BL statistics from a channel flow using AVVEL.
* example_dissi_parallel: An example on how to compute the dissipation and the Kolmogorov length and time scales.
* example_checkpint_parallel: An example on how to use the checkpoint when computing the dissipation.
* example_MEP: An example on how to apply MEP to obtain the regression.
* example_GEOM: An example on how to use the Geometry module in 2D.
* example_GEOM_3D: An example on how to use the Geometry module in 3D.

Please read the instructions carefully and address any questions to [arnau.mirojane(at)bsc.es](mailto:arnau.mirojane@bsc.es) or [benet.eiximeno(at)bsc.es](mailto:benet.eiximeno@bsc.es).