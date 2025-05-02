#!/usr/bin/env python
#
# Example how to perform IO in MPIO format
# using the Mesh and Field classes.
#
# Last revision: 18/03/2021
from __future__ import print_function, division

import numpy as np

import pyQvarsi


BASEDIR = './'
CASESTR = 'channel'
VARLIST = ['VELOC','PRESS']


## Create the subdomain mesh
mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=True,read_massm=False)


# Read an instant of the list
field, header = pyQvarsi.Field.read(CASESTR,VARLIST,153679,mesh.xyz,basedir=BASEDIR)
time = header.time

## Create an output field with a dummy variable
# It is generally preferable to create a new field for outputting the variables
# otherwise use the exclude_vars in the write method
outf = pyQvarsi.Field(xyz=mesh.xyz,DUMMY=field['PRESS'])

# Write MPIO
outf.write(CASESTR,153679,time,basedir=BASEDIR,fmt='mpio')

pyQvarsi.pprint(0,'Done!')
pyQvarsi.cr_info()