#!/usr/bin/env python
#
# Example hot to plot a field. Currently only implemented for
# 2D triangular grids. So, if a quad give is read (as in this example)
# we convert the quad elements to tri elements using "quad2tri" and then
# obtain the connectivity list, which we then pass to the plotting routine.
#
# Last revision: 23/09/2021
from __future__ import print_function, division

import os, numpy as np
import pyQvarsi

# Tri03 example (no conversion needed)

# Quad04 example (conversion to tri elements needed)
PATH    = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.join(PATH,'data','CurvedBackstep')
FIELDSTR = 'curvedbackstep_0.h5'
MESHSTR = 'curvedbackstep_grid.h5'

mesh  = pyQvarsi.Mesh.load(os.path.join(BASEDIR,MESHSTR))
field = pyQvarsi.Field.load(os.path.join(BASEDIR,FIELDSTR))

pyQvarsi.pprint(0,field)
pyQvarsi.pprint(0,mesh)

# Create triangular elements and connectivity list
elemList = pyQvarsi.FEM.quad2tri(mesh.elem)
connecTriList = [np.asarray(e.nodes) for e in elemList]

# Plot streamwise velocity field
plotter = pyQvarsi.utils.Plotter()
plotter.plot2Dtri(field['VELOC'][:,0], connecTriList, mesh.x, mesh.y, contour=True, cbar_flag=True)
# Compute and plot vorticity
gradv = mesh.gradient(field['VELOC'])
vortZ = pyQvarsi.postproc.vorticity(gradv)[:,-1]
plotter.plot2Dtri(vortZ, connecTriList, mesh.x, mesh.y, contour=True, cbar_flag=True, cmap='bwr')

pyQvarsi.cr_info()