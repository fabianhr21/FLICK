#!/usr/bin/env python
#
# Example to show prove that it is possible to read and split 
# an h5 mesh and field with N processors 
# (the plotting part is thought only for 1st order elements)
#
# Last rev: 23/07/2024
from __future__ import print_function, division

import numpy as np, matplotlib.pyplot as plt
import pyQvarsi

# Load H5 data
mesh  = pyQvarsi.MeshAlya.load('examples/data/io_examples/sphere.h5',compute_massMatrix=False)
print(mesh)
field = pyQvarsi.FieldAlya.load('examples/data/io_examples/sphere.h5',inods=mesh.lninv)
print(field)

xyz   = pyQvarsi.utils.mpi_gather(mesh.xyz,root=0)
velox = pyQvarsi.utils.mpi_gather(field['VELOX'],root=0)

if pyQvarsi.utils.is_rank_or_serial():
    # Plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=velox, cmap='RdBu_r')
    plt.show()