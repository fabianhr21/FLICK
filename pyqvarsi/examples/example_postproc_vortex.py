#!/usr/bin/env python
#
# Example postprocessing vortex detection quantities.
#
# Last rev: 27/04/2021
from __future__ import print_function, division

# Please do not delete this part otherwise it will not work
# you have been warned after a long weekend of debugging
import mpi4py
mpi4py.rc.recv_mprobe = False

import numpy as np
import pyQvarsi

# Parameters
rho, mu = 1.0, 0.00556

BASEDIR        = '1.t-0-60-copy.bin'
CASESTR        = 'chan'
VARLIST        = ['VELOC']
START, DT, END = 1,1,415+1

# In case of restart, load the previous data
listOfInstants = [ii for ii in range(START,END,DT)]


## Create the subdomain mesh
mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,fmt='ensi')

pyQvarsi.pprint(0,'Run (%d instants)...' % len(listOfInstants),flush=True)


## Loop the instants and compute OMEGA
for instant in listOfInstants:
	if instant%100 == 0: pyQvarsi.pprint(1,'Instant %d...'%instant,flush=True)
	
	# Read field
	field, header = pyQvarsi.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=BASEDIR,fmt='ensi')

	# Compute and smooth the gradient of velocity
	gradv = mesh.smooth(mesh.gradient(field['VELOC']),iters=3)

	# Store gradients
	field['GRADV'] = mesh.newArray(ndim=6)
	field['GRADV'][:,0] = gradv[:,0] # XX
	field['GRADV'][:,1] = gradv[:,4] # YY
	field['GRADV'][:,2] = gradv[:,8] # ZZ
	field['GRADV'][:,3] = gradv[:,1] # XY
	field['GRADV'][:,4] = gradv[:,2] # XZ
	field['GRADV'][:,5] = gradv[:,5] # YZ

	# Compute Vorticity, Q and Omega from the gradient
	field['VORTI'] = pyQvarsi.postproc.vorticity(gradv)
	field['QCRIT'] = pyQvarsi.postproc.QCriterion(gradv)
	field['LAMB2'] = pyQvarsi.postproc.Lambda2Criterion(gradv)
	field['OMEGA'] = pyQvarsi.postproc.OmegaCriterion(gradv,epsilon=0.001,modified=False)
	field['RORTX'] = pyQvarsi.postproc.RortexCriterion(gradv)
	field['OMERX'] = pyQvarsi.postproc.OmegaRortexCriterion(gradv,epsilon=0.001,modified=False)

	# Use mpio saving capabilities to store in aux
	field.write(CASESTR,instant,header.time,basedir=BASEDIR,fmt='ensi',exclude_vars=['VELOC'])

pyQvarsi.cr_info()