#!/usr/bin/env python
#
# Example postprocessing vortex detection quantities
# using burger's vortex solution.
#
# Last rev: 27/08/2021
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


## Write output - VTKH5
# In VTKHDF always store the mesh first and then the field/s
mesh.write('burgers', fmt='vtkh5')
field.write('burgers',fmt='vtkh5')
# For multiple time-instant datasets, mesh linking can be used
# so that the mesh is not stored at each instant. For instance:
mesh.write('burgers_mesh', fmt='vtkh5')  # First store the mesh
# To store the field always start by linking the mesh
mesh.write('burgers_field', fmt='vtkh5', linkfile='burgers_mesh') # extensions added automatically
field.write('burgers_field',fmt='vtkh5')


pyQvarsi.cr_info()