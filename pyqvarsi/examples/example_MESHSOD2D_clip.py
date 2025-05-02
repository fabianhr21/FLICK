#!/usr/bin/env python
#
# Example of the MESH clip feature.
#
# Last revision: 22/06/2021
from __future__ import print_function, division

import numpy as np

import pyQvarsi

BASEDIR = '/home/benet/Dropbox/UNIVERSITAT/PhD/test_cases/mms_parallel/p3'
CASESTR = 'cube_bound_N_4'

## Read mesh and extract boundary condition
mesh = pyQvarsi.MeshSOD2D.read(CASESTR,basedir=BASEDIR)

x0    = 0.5
omega = 3*np.pi/2
phi   = np.pi/2-omega*x0
scaf  = np.cos(omega*mesh.x+phi)*np.sin(omega*mesh.y-phi)*np.cos(omega*mesh.z+phi)
field = pyQvarsi.FieldSOD2D(xyz=mesh.xyz,ptable=mesh.partition_table,scaf=scaf)

rect = pyQvarsi.Geom.SimpleRectangle(-0.1,0.5,0.5,1.2)
cmesh, mask = mesh.clip(rect)
cfield = field.selectMask(mask)

print(cmesh.xyz.shape, cfield.xyz.shape)
print('scaf.shape',cfield['scaf'].shape)

cmesh.write('eldandy')
cfield.write('eldandy')

pyQvarsi.cr_info()