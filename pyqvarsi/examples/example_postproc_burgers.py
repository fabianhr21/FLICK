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


## Compute gradient of velocity
gradv = mesh.gradient(field['VELOC'])
# Store gradients
field['GRADV'] = gradv


## Compute vortex detection methods
field['VORTI'] = pyQvarsi.postproc.vorticity(gradv)
field['QCRIT'] = pyQvarsi.postproc.QCriterion(gradv)
field['LAMB2'] = pyQvarsi.postproc.Lambda2Criterion(gradv)
field['OMEGA'] = pyQvarsi.postproc.OmegaCriterion(gradv,epsilon=0.001,modified=False)
field['OMEGM'] = pyQvarsi.postproc.OmegaCriterion(gradv,epsilon=0.001,modified=True)
field['RORTX'] = pyQvarsi.postproc.RortexCriterion(gradv)
field['OMERX'] = pyQvarsi.postproc.OmegaRortexCriterion(gradv,epsilon=0.001,modified=False)
field['OMRXM'] = pyQvarsi.postproc.OmegaRortexCriterion(gradv,epsilon=0.001,modified=True)
pyQvarsi.pprint(0,field,flush=True)


## Write output - VTKH5
mesh.write('burgers',fmt='vtkh5')
field.write('burgers',fmt='vtkh5')


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
				{'name':'GRADV','type':'Node','ndim':9,'file':'field_%d.h5'},
				{'name':'VORTI','type':'Node','ndim':3,'file':'field_%d.h5'},
				{'name':'QCRIT','type':'Node','ndim':1,'file':'field_%d.h5'},
				{'name':'LAMB2','type':'Node','ndim':1,'file':'field_%d.h5'},
				{'name':'OMEGA','type':'Node','ndim':1,'file':'field_%d.h5'},
				{'name':'OMEGM','type':'Node','ndim':1,'file':'field_%d.h5'},
				{'name':'RORTX','type':'Node','ndim':3,'file':'field_%d.h5'},
				{'name':'OMERX','type':'Node','ndim':1,'file':'field_%d.h5'},
				{'name':'OMRXM','type':'Node','ndim':1,'file':'field_%d.h5'},
		],
	},
	np.array([1])
)

## Plot with pyVista
# Will only work if pyVista is enabled
pyQvarsi.plotting.pvplot(mesh,field,['OMEGA'],cmap='jet',screenshot='burgers_omega.png')

pyQvarsi.cr_info()
