#!/usr/bin/env python
#
# Example how to perform IO in VTK format
# and create the pyQvarsi Mesh and Field classes.
#
# Last revision: 15/12/2021
from __future__ import print_function, division

import os
import pyQvarsi


PATH    = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.join(PATH,'data','ChannelVTK')
CASESTR = 'outputVTK.vtu'
VARLIST = ['State']


## Read VTK to Mesh class
mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,fmt='vtk')
pyQvarsi.pprint(0,mesh)


## Read fields to Field class
fields, header = pyQvarsi.Field.read(CASESTR,VARLIST,0,mesh.xyz,basedir=BASEDIR,fmt='vtk')
pyQvarsi.pprint(0,fields)


pyQvarsi.cr_info()