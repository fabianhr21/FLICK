#!/usr/bin/env python
#
# Example how to perform IO in VTK format
# and create the pyQvarsi Mesh and Field classes.
#
# Last revision: 15/12/2021
from __future__ import print_function, division

import os, numpy as np
import pyQvarsi

PATH     = os.path.dirname(os.path.abspath(__file__))
FNAME1   = os.path.join(PATH,'data','ChannelVTK','outputVTK.vtu') # Non-partitioned VTU data
FNAME2   = os.path.join(PATH,'data','CavTri03','CavTri03VTK-coarse','cavtri03_00000001.pvtu') # Partitioned VTU data
VARLIST1 = ['State']
VARLIST2 = ['VELOC']

## Get VTK data and print info
pyQvarsi.pprint(0,'Get VTK data and print info')
dataVTK = pyQvarsi.io.vtkIO(filename=FNAME1, mode='read', varlist=VARLIST1)
pyQvarsi.pprint(0,dataVTK)

## Convert cell data to point data
pyQvarsi.pprint(0,'Convert cell data to point data')
pyQvarsi.io.vtkCelltoPointData(dataVTK,VARLIST1,keepdata=False)
pyQvarsi.pprint(0,dataVTK)

## Convert VTK data to numpy arrays
pyQvarsi.pprint(0,'Convert VTK data to numpy arrays')
points    = pyQvarsi.io.vtk_to_numpy(dataVTK.pointsVTK)
pointData = {k:pyQvarsi.io.vtk_to_numpy(v) for k,v in dataVTK.pointDataVTK.items()}
cellData  = {k:pyQvarsi.io.vtk_to_numpy(v) for k,v in dataVTK.cellDataVTK.items()}
pyQvarsi.pprint(0,'points',points)
pyQvarsi.pprint(0,'pointData',pointData)
pyQvarsi.pprint(0,'cellData',cellData)

## Alternatively, we can get VTK data in numpy format directly
pyQvarsi.pprint(0,'Get VTK data in numpy format directly')
data     = dataVTK.get_vars()
#dataTest = dataVTK.get_vars(['Test']) # no data found!
#data_3D  = dataVTK.get_vars_3D()
pyQvarsi.pprint(0,'data',data)

## Create Mesh and Fields pyQvarsi classes
pyQvarsi.pprint(0,'Create Mesh and Fields pyQvarsi classes')
xyz   = dataVTK.points
lnods = dataVTK.connectivity
ltype = dataVTK.cellTypes
lninv = np.arange(xyz.shape[0],dtype=np.int32)
leinv = np.arange(ltype.shape[0],dtype=np.int32)
mesh  = pyQvarsi.Mesh(xyz,lnods,ltype,lninv,leinv)
pyQvarsi.pprint(0,mesh)

field = pyQvarsi.Field(xyz=mesh.xyz,STATE=data['State_pointdata'][:,0])
pyQvarsi.pprint(0,field)

## Test pvtu (partitioned data)
pyQvarsi.pprint(0,'Test pvtu (partitioned data)')
dataVTK = pyQvarsi.io.vtkIO(filename=FNAME2, mode='read', varlist=VARLIST2)
data    = dataVTK.get_vars()
pyQvarsi.pprint(0,dataVTK)
pyQvarsi.pprint(0,data)

pyQvarsi.cr_info()