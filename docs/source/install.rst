Installation
============

Requirements
------------

- Python 3.8+
- Git (with submodule support)
- For GPU support: CUDA-enabled PyTorch
- For HPC: MPI (mpi4py)
- For City4CFD compilation: CMake, Boost, CGAL, Eigen, GDAL

Clone and Initialize Submodules
---------------------------------

.. code-block:: bash

   git clone https://github.com/fabianhr21/FLICK.git
   cd FLICK
   git submodule update --init --recursive

Install Python Package
-----------------------

Base installation:

.. code-block:: bash

   pip install -e .

With GPU (PyTorch/CUDA) support:

.. code-block:: bash

   pip install -e .[gpu]

With HPC (MPI) support:

.. code-block:: bash

   pip install -e .[hpc]

Full (GPU + HPC):

.. code-block:: bash

   pip install -e .[gpu,hpc]

Compile External Tools
-----------------------

.. code-block:: bash

   bash scripts/compile_tools.sh

This installs system dependencies, compiles City4CFD, and links the binary.

Model Weights
--------------

The neural network weights are **not included** in the repository.
Request them from: fabian.hernandez@bsc.es

Expected path after receiving: ``170625_weights/`` in the repo root.
