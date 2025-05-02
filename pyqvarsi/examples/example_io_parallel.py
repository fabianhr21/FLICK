#!/usr/bin/env python
#
# Example loading and saving in parallel.
#
# Last revision: 05/03/2021
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False

import numpy as np
import pyQvarsi


## Parameters
FILENAME = 'V1/Statistics.h5'


## Load from the database and create a field (in parallel)
db = pyQvarsi.io.HiFiTurbDB_Reader(FILENAME,return_matrix=True,parallel=True)

# Create a field 
field = pyQvarsi.Field(xyz   = db.points,
					 GRADP = db.pressure_gradient,
					 GRADV = db.velocity_gradient,
					)
field.save('test.h5')

# Try reloading the field we have just saved
field2 = pyQvarsi.Field.load('test.h5')
pyQvarsi.pprint(-1,field,flush=True)

pyQvarsi.pprint(0,'done!')