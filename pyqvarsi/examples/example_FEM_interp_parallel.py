#!/usr/bin/env python
#
# Example of the MESH class interpolation.
# Parallel case on a channel flow.
#
# Last revision: 25/05/2021
from __future__ import print_function, division

import numpy as np

import pyQvarsi


## Parameters
BASEDIR        = './'
CASESTR        = 'channel'
VARLIST        = ['AVPRE','AVVEL']


## Read data
# Create the subdomain mesh
mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=True,read_massm=False)

# Read fields
field,_ = pyQvarsi.Field.read(CASESTR,VARLIST,0,mesh.xyz,basedir=BASEDIR)


## Interpolate
# Create an interpolating plane mesh
meshp = pyQvarsi.Mesh.plane(
	np.array([6.,0.,0.]),
	np.array([6.,2.,0.]),
	np.array([6.,0.,4./3.*np.pi]),
	65,65,ngauss=1,bunching='all',f=1.2
)
pyQvarsi.pprint(0,meshp,flush=True)

# Interpolate to the plane mesh
# Output the mask so that lninv can be stored inside the mesh
fieldp = pyQvarsi.meshing.interpolateFEM(mesh,meshp.xyz,field,fact=2.0,r_incr=1.0,ball_max_iter=100,root=0)
pyQvarsi.pprint(0,fieldp)


## Operate on master only
if pyQvarsi.utils.is_rank_or_serial():
	meshp.save('mesh.h5',mpio=False)
	fieldp.save('field_0.h5',mpio=False)
	pyQvarsi.io.xdmf_save('plane.xdmf',
		{
			'mesh' : {
				'nnod' : meshp.nnodG,
				'nel'  : meshp.nelG,
				'type' : meshp.eltype[0],
				'file' : 'mesh.h5'
			},
			'variables' : [
					{'name':'AVPRE','type':'Node','ndim':1,'file':'field_%d.h5'},
					{'name':'AVVEL','type':'Node','ndim':3,'file':'field_%d.h5'},
			],
		},
		np.array([1])
	)

pyQvarsi.cr_info()