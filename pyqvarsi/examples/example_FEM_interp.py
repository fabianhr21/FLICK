#!/usr/bin/env python
#
# Example of the MESH class interpolation.
#
# Last revision: 12/04/2021
from __future__ import print_function, division

import numpy as np

import pyQvarsi


# Define mesh parameters
xyz = np.array([[0,0,0],   # 1
				[1.2,1,0], # 2
				[0,0.3,0], # 3
				[0,1,0.2], # 4
				[0,1.5,0]  # 5
	  		   ])
codno = np.ones((xyz.shape[0],3),dtype=np.int32)
lnods = np.array([[1,2,3,4],[3,2,5,4]],dtype=np.int32) - 1
ltype = np.array([30,30],dtype=np.int32)
lninv = np.array([1,2,3,4,5],dtype=np.int32) - 1
leinv = np.array([1,2],dtype=np.int32) - 1

# Create the domain mesh
mesh = pyQvarsi.MeshAlya(xyz,lnods,ltype,lninv,leinv,codno=codno,ngauss=1)

# Create a Field class to store the arrays
field = pyQvarsi.FieldAlya(xyz=mesh.xyz)

# Create a random scalar field
field['SCAF'] = np.ones((len(field),),dtype=np.double)
field['SCAF'][0:2] = 2.

# Create a random vectorial field
field['VECF'] = np.ones((len(field),mesh.ndim),dtype=np.double)
field['VECF'][0:3,1:mesh.ndim] = 2
field['VECF'][2:5,0]           = 3

# Interpolate to points
p   = np.array([[0.,0.,0.],[0.1,0.1,0.],[0.3,0.575,0.05]])
out = pyQvarsi.meshing.interpolateFEM(mesh,p,field,fact=2.0,r_incr=1.0,ball_max_iter=5,root=-1)

print(mesh)
print(field)
print(out)

pyQvarsi.cr_info()