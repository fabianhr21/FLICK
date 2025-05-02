#!/usr/bin/env python
#
# Example rotating a mesh and a vectorial
# field. We will use the Burger's vortex as
# a toy case.
#
# Last rev: 22/07/2024
from __future__ import print_function, division

import numpy as np
import pyQvarsi


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
p1 = np.array([-4.,-4., 0.],np.double)
p2 = np.array([ 4.,-4., 0.],np.double)
p4 = np.array([-4., 4., 0.],np.double)
p5 = np.array([-4.,-4.,-4.],np.double)
mesh = pyQvarsi.MeshAlya.cube(p1,p2,p4,p5,n,n,n)
pyQvarsi.pprint(0,mesh,flush=True)


## Create a field and load velocity
field = pyQvarsi.FieldAlya(xyz=mesh.xyz,VELOC=burgers_vel(mesh.xyz))
pyQvarsi.pprint(0,field,flush=True)

mesh.write('original',fmt='vtkh5')
field.write('original',fmt='vtkh5')


## Rotate the mesh and the field 
angles = np.array([0.,0.,45.],np.double)
center = np.array([-4.,-4., 0.],np.double)
mesh.rotate(angles,center)
field.rotate(angles,center)

pyQvarsi.pprint(0,mesh,flush=True)
pyQvarsi.pprint(0,field,flush=True)
mesh.write('rotated',fmt='vtkh5')
field.write('rotated',fmt='vtkh5')