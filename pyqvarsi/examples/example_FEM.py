#!/usr/bin/env python
#
# Example of the MESH class, communications
# and FEM operations with a simple example.
#
# Last revision: 21/10/2020
from __future__ import print_function, division

import numpy as np

import pyQvarsi


# Define mesh parameters
xyz = np.array([[0,0,0],   # 1
				[1.2,1,0], # 2
				[0,0.3,0], # 3
				[0,1,0.2], # 4
				[0,1.5,0]  # 5
	  		   ],dtype=np.double)
codno = np.ones((xyz.shape[0],),dtype=np.int32)
lnods = np.array([[1,2,3,4],[3,2,5,4]],dtype=np.int32) - 1
ltype = np.array([30,30],dtype=np.int32)
lninv = np.array([1,2,3,4,5],dtype=np.int32) - 1
leinv = np.array([1,2],dtype=np.int32) - 1

# Create the domain mesh
mesh = pyQvarsi.MeshAlya(xyz,lnods,ltype,lninv,leinv,codno=codno,ngauss=4,consistent_mass=True)

# Create a Field class to store the arrays
field = pyQvarsi.FieldAlya(xyz=mesh.xyz)

# Create a random scalar field to test the 2D gradient routines
field['SCAF'] = np.ones((len(field),),dtype=np.double)
field['SCAF'][0:2] = 2.
field['GSCA'] = mesh.gradient(field['SCAF'])
field['LSCA'] = mesh.laplacian(field['SCAF'])

# Now test the gradient computed at the Gauss points and projected
# to the nodes
gsca_gp = mesh.gradient(field['SCAF'],on_Gauss=True)
field['GSC2'] = mesh.gauss2Nodes(gsca_gp)

# Create a random vectorial field to test the 2D gradient routines
field['VECF'] = np.ones((len(field),mesh.ndim),dtype=np.double)
field['VECF'][0:3,1:mesh.ndim] = 2
field['VECF'][2:5,0]           = 3
field['GVEC'] = mesh.gradient(field['VECF'])
field['DVEC'] = mesh.divergence(field['VECF'])
field['LVEC'] = mesh.laplacian(field['VECF'])

# Now test the gradient computed at the Gauss points and projected
# to the nodes
gvec_gp = mesh.gradient(field['VECF'],on_Gauss=True)
field['GVE2'] = mesh.gauss2Nodes(gvec_gp)

pyQvarsi.printArray("VOLU",mesh.volume,6)
print('')
pyQvarsi.printArray('SCAF',field['SCAF'])
pyQvarsi.printArray('GSCA',field['GSCA'])
pyQvarsi.printArray('GP  ',gsca_gp)
pyQvarsi.printArray('GSC2',field['GSC2'])
pyQvarsi.printArray('LSCA',field['LSCA'])
print('')
pyQvarsi.printArray('VECF',field['VECF'])
pyQvarsi.printArray('GVEC',field['GVEC'])
pyQvarsi.printArray('GP  ',gvec_gp)
pyQvarsi.printArray('GVE2',field['GVE2'])
pyQvarsi.printArray('DVEC',field['DVEC'])
pyQvarsi.printArray('LVEC',field['LVEC'])

pyQvarsi.cr_info()
