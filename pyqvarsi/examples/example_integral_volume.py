#!/usr/bin/env python
#
# Example of the computation of the volume
# integral on Alya's witness meshes.
#
# Last revision: 11/04/2021
from __future__ import print_function, division

import numpy as np
import pyQvarsi


BASEDIR = './'
CASESTR = 'sphere'


## Read mesh
mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=False,read_massm=False)


## Create an array for the area and the mask
# In a real case use the Field.read to read a proper variable and integrate it
# over a specified patch of volume
area = np.ones((mesh.xyz.shape[0],),dtype=np.double)

# We also need to create a mask for the integral
# i.e., we can discard some nodes (but we're not interested in this example)
mask = np.ones((mesh.xyz.shape[0],),dtype=bool)


## Integrate to compute the surface and heat on the partition
vol = mesh.integral(area,mask,kind='volume')


## Master reduces and computes the total heat and surface
vol_g = pyQvarsi.Communicator.allreduce(vol)


## Print
pyQvarsi.pprint(0,'Volume = %f' % (vol_g))

pyQvarsi.cr_info()