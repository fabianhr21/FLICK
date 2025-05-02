#!/usr/bin/env python
#
# Example of the MESH clip feature.
#
# Last revision: 22/06/2021
from __future__ import print_function, division

import numpy as np

import pyQvarsi


## Function
fun = lambda xyz : np.sin(2.0*np.pi*xyz[:,0])*np.sin(2.0*np.pi*xyz[:,1])


## Create a simple mesh square
n  = 32
p1 = np.array([0.,0.,0.])
p2 = np.array([1.,0.,0.])
p4 = np.array([0.,1.,0.])
mesh = pyQvarsi.MeshAlya.plane(p1,p2,p4,n,n)
#print(mesh)
# else read a mesh from Alya
#mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=False,read_massm=False,read_codno=False)

# Build a field
field  = pyQvarsi.FieldAlya(xyz=mesh.xyz,SCAF=fun(mesh.xyz))

#print(field)

# Store mesh and field into HDF5, also create XDMF
mesh.save('mesh.h5')
field.save('field_0.h5')
pyQvarsi.io.xdmf_save('test.xdmf',
	{
		'mesh' : {
			'nnod' : mesh.nnodG,
			'nel'  : mesh.nelG,
			'type' : mesh.eltype[0],
			'file' : 'mesh.h5'
		},
		'variables' : [
				{'name':'SCAF','type':'Node','ndim':1,'file':'field_%d.h5'},
		],
	},
	np.array([1])
)
# Or store the field as MPIO
#field.write(CASESTR,0,0.,basedir=BASEDIR)


## Crop a part of the mesh
rect = pyQvarsi.Geom.SimpleRectangle(-0.1,0.5,0.5,1.2)
cmesh, mask = mesh.clip(rect)
cfield = field.selectMask(mask)


## Store mesh and field into HDF5, also create XDMF
cmesh.save('cmesh.h5')
cfield.save('cfield_0.h5')
pyQvarsi.io.xdmf_save('ctest.xdmf',
	{
		'mesh' : {
			'nnod' : cmesh.nnodG,
			'nel'  : cmesh.nelG,
			'type' : cmesh.eltype[0],
			'file' : 'cmesh.h5'
		},
		'variables' : [
				{'name':'SCAF','type':'Node','ndim':1,'file':'cfield_%d.h5'},
		],
	},
	np.array([1])
)
# or store the new cropped mesh as MPIO
#cmesh.write('c'+CASESTR)
#cfield.write('c'+CASESTR,0,0.,basedir=BASEDIR)

pyQvarsi.cr_info()