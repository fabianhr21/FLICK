#!/usr/bin/env python
#
# Example of the computation of the surface
# integral on Alya's witness meshes.
#
# Last revision: 11/04/2021
from __future__ import print_function, division

import numpy as np
import pyQvarsi


BASEDIR = './BC/'
CASESTR = 'sphere-BC'
VARLIST = ['GRATE','EXNOR']


## Read mesh
mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=False,read_massm=False)


## Read variables
# EXNOR already treated when reading so no need to do any fix
field, _ = pyQvarsi.Field.read(CASESTR,VARLIST,199667,mesh.xyz,basedir=BASEDIR)

# Compute the heat flux
field['HEATF'] = np.sum(field['GRATE']*field['EXNOR'],axis=1) # q = dT/dn * n

# To compute the surface create a field equal to 1 that
# will be used within the integral to compute the surface
field['SAREA'] = np.ones((mesh.xyz.shape[0],),dtype=np.double)

# We also need to create a mask for the integral
# i.e., we can discard some nodes (but we're not interested in this example)
mask = np.ones((mesh.xyz.shape[0],),dtype=bool)


## Integrate to compute the surface and heat on the partition
heat = mesh.integral(field['HEATF'],mask,kind='surf')
surf = mesh.integral(field['SAREA'],mask,kind='surf')


## Master reduces and computes the total heat and surface
heat_g = pyQvarsi.Communicator.allreduce(heat)
surf_g = pyQvarsi.Communicator.allreduce(surf)


## Print
pyQvarsi.pprint(0,'Surface = %f, Nu = %f' % (surf_g,heat_g/surf_g))

pyQvarsi.cr_info()