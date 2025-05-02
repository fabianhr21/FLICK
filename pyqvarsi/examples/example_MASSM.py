#!/usr/bin/env python
#
# Example of the MESH class, reading and
# computing the mass matrix in order to
# verify the calculations of FEM.
#
# Last revision: 23/11/2020
from __future__ import print_function, division

import numpy as np

import pyQvarsi


BASEDIR   = './'
CASESTR   = 'cavtri03'


# Create the subdomain mesh
mesh  = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=True,read_massm=True)

# Create a Field class to store the arrays
field = pyQvarsi.Field(xyz=pyQvarsi.truncate(mesh.xyz,6),
					 MASSM1=mesh.mass_matrix,
					 MASSM2=mesh.computeMassMatrix() # Compute the mass matrix
					)

# Reduce the field (using allreduce)
fieldG = field.reduce(op=pyQvarsi.fieldFastReduce)

# Print
pyQvarsi.printArray('MASSM read:',fieldG['MASSM1'],rank=0,precision=6)
pyQvarsi.printArray('MASSM computed:',fieldG['MASSM2'],rank=0,precision=6)
pyQvarsi.printArray('MASSM diff:',fieldG['MASSM1']-fieldG['MASSM2'],rank=0,precision=12)

pyQvarsi.cr_info()