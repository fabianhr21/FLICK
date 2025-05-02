#!/usr/bin/env python
#
# Example of the MESH class, communications
# and FEM operations with a simple example
# in parallel.
#
# Last revision: 21/10/2020
from __future__ import print_function, division

import numpy as np

import pyQvarsi


BASEDIR   = './'
CASESTR   = 'cavtri03'


# Create the subdomain mesh
mesh  = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,ngauss=4,read_commu=True,read_massm=False)

# Read the mass matrix as an external field
# just for comparison purposes
field, _ = pyQvarsi.Field.read(CASESTR,['MASSM'],0,mesh.xyz,basedir=BASEDIR)

# Create a random scalar field to test the 2D gradient routines
field['SCAF'] = np.ones((len(field),),dtype=np.double)
if mesh.comm.rank > 0: field['SCAF'][0:2] = 2.
field['GSCA'] = mesh.gradient(field['SCAF'])

# Create a random vectorial field to test the 2D gradient routines
field['VECF'] = np.ones((len(field),mesh.ndim),dtype=np.double)
if mesh.comm.rank > 0: 
	field['VECF'][0:3,1:mesh.ndim] = 2
	field['VECF'][2:5,0]           = 3
field['GVEC'] = mesh.gradient(field['VECF'])


# Print the mass matrix to check that both are equal
pyQvarsi.printArray("M1 "+str(mesh.comm.rank),mesh.mass_matrix,rank=0,precision=6)
pyQvarsi.printArray("M2 "+str(mesh.comm.rank),field['MASSM'],rank=0,precision=6)


# Reduce the field (using allreduce)
fieldG = field.reduce(op=pyQvarsi.fieldFastReduce)

pyQvarsi.pprint(0,'')
pyQvarsi.printArray('SCAF',fieldG['SCAF'],rank=0)
pyQvarsi.printArray('GSCA',fieldG['GSCA'],rank=0)
pyQvarsi.pprint(0,'')
pyQvarsi.printArray('VECF',fieldG['VECF'],rank=0)
pyQvarsi.printArray('GVEC',fieldG['GVEC'],rank=0)

pyQvarsi.cr_info()