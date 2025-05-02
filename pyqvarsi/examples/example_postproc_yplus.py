#!/usr/bin/env python
#
# Example how to compute the YPLUS on a channel.
#
# Last revision: 01/07/2021
from __future__ import print_function, division

# Please do not delete this part otherwise it will not work
# you have been warned after a long weekend of debugging
import numpy as np, mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import pyQvarsi

mu = 6.173e-5
mpi_size = MPI.COMM_WORLD.Get_size()


## Load field
field = pyQvarsi.Field.load('channel180.h5')
if (mpi_size) > 1: pyQvarsi.pprint(0,field,flush=True)


## Compute yplus
field['YPLUS'], field['WALLD'] = pyQvarsi.postproc.yplus_xyz_3D(field.xyz,
							 field['GRAVV'][:,:3], # Grad(u) since the flow is on x direction
							 mu,
							 np.nan*np.ones((len(field),),dtype=np.double), # NaN since there is no wall on x
							 np.nan*np.ones((len(field),),dtype=np.double), # NaN since there is no wall on x
							 2.*np.ones((len(field),)    ,dtype=np.double),
							 np.zeros((len(field),)      ,dtype=np.double),
							 np.nan*np.ones((len(field),),dtype=np.double), # NaN since there is no wall on x
							 np.nan*np.ones((len(field),),dtype=np.double), # NaN since there is no wall on x
							 4,4,4)
if mpi_size > 1:
	fieldG = field.reduce(root=0,op=pyQvarsi.fieldFastReduce)
	pyQvarsi.pprint(0,fieldG,flush=True)
else:
	pyQvarsi.pprint(0,field,flush=True)


## Finish
pyQvarsi.cr_info()