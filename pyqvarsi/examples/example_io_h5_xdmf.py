#!/usr/bin/env python
#
# Example outputting in XDMF format.
# NOTE: this is currently deprecated and kept
# only for legacy issues
#
# Last rev: 02/09/2024
from __future__ import print_function, division

import os, numpy as np
import pyQvarsi

## Parameters
CASESTR = 'burgers'
OUTDIR  = 'out'


## Flow definition
sr = 1.  # Strain rate
nu = 1.  # Viscosity
Re = 10. # Reynolds number
C  = 0.  # Shearing motion

Gamma = 2*np.pi*Re*nu
ro2   = 1.5852*1.5852

# Flow description
def burgers_vel(xyz):
	Cpar = C*Re/ro2
	r2   = xyz[:,0]*xyz[:,0] + xyz[:,1]*xyz[:,1]
	uvw  = np.zeros_like(xyz)
	uvw[:,0] = -sr*xyz[:,0] - Gamma/(2.*np.pi*r2)*(1. - np.exp(-r2*sr/(2.*nu)))*xyz[:,1] - Cpar*sr*xyz[:,1]
	uvw[:,1] = -sr*xyz[:,1] + Gamma/(2.*np.pi*r2)*(1. - np.exp(-r2*sr/(2.*nu)))*xyz[:,0]
	uvw[:,2] = 2.*sr*xyz[:,2]
	return uvw


## Mesh creation
n  = 32
p1 = np.array([-4.,-4., 0.])
p2 = np.array([ 4.,-4., 0.])
p4 = np.array([-4., 4., 0.])
p5 = np.array([-4.,-4.,-4.])
mesh = pyQvarsi.MeshAlya.cube(p1,p2,p4,p5,n,n,n,use_consistent=True)
pyQvarsi.pprint(0,mesh,flush=True)


## Create a field and load velocity
field = pyQvarsi.FieldAlya(xyz=mesh.xyz,VELOC=burgers_vel(mesh.xyz))
pyQvarsi.pprint(0,field,flush=True)


## Write output - XDMF
mesh.save('mesh.h5')
field.save('field_0.h5')
pyQvarsi.io.xdmf_save('burgers.xdmf',
	{
		'mesh' : {
			'nnod' : mesh.nnodG,
			'nel'  : mesh.nelG,
			'type' : mesh.eltype[0],
			'file' : 'mesh.h5'
		},
		'variables' : [
				{'name':'VELOC','type':'Node','ndim':3,'file':'field_%d.h5'},
		],
	},
	np.array([1])
)


pyQvarsi.cr_info()