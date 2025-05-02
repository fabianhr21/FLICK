#!/usr/bin/env python
#
# Example how to compute the YPLUS on a channel.
#
# Last revision: 19/08/2021
from __future__ import print_function, division

# Please do not delete this part otherwise it will not work
# you have been warned after a long weekend of debugging
import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import numpy as np
import pyQvarsi

comm     = MPI.COMM_WORLD
rank     = comm.Get_rank()
size     = comm.Get_size()

## Parameters
BASEDIR = '../'
CASESTR = 'channel'
VARLIST = ['GRAVV']
mu      = 6.173e-5


## Load mesh and field
mesh    = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=True,read_massm=False)
field,_ = pyQvarsi.Field.read(CASESTR,VARLIST,0,mesh.xyz,basedir=BASEDIR)


## Compute yplus
field['YPLUS'], field['WALLD'] = pyQvarsi.postproc.yplus_xyz_3D(
							field.xyz,
							field['GRAVV'][:,:3], # Grad(u) since the flow is on x direction
							mu,
							np.nan*np.ones((len(field),),dtype=np.double), # NaN since there is no wall on x
							np.nan*np.ones((len(field),),dtype=np.double), # NaN since there is no wall on x
							2.*np.ones((len(field),)    ,dtype=np.double),
							np.zeros((len(field),)      ,dtype=np.double),
							np.nan*np.ones((len(field),),dtype=np.double), # NaN since there is no wall on x
							np.nan*np.ones((len(field),),dtype=np.double), # NaN since there is no wall on x
							nbx=4, nby=4, nbz=4, fact=0.1
)

fieldG = field.reduce(root=0,op=pyQvarsi.fieldFastReduce)
pyQvarsi.pprint(0,fieldG,flush=True)


## Finish
pyQvarsi.cr_info()