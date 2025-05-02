#!/usr/bin/env python
#
# Example how to perform IO in MPIO format
# using the RAW IO routines.
#
# Last revision: 18/03/2021
from __future__ import print_function, division

import os,numpy as np

import pyQvarsi


BASEDIR = './'
CASESTR = 'c'


## Filename and other parameters that we need
#  Using the helpers inside the IO module for convenience although
#  the sting name can be easily hardcoded
partition_file = None # Not needed if dealing with a serial file
coord_file     = 'c-COORD.mpio.bin'

# Read using MPIO
data, header = pyQvarsi.io.AlyaMPIO_read(os.path.join(BASEDIR,coord_file),partition_file)


## Modify your data if needed
#  Now we have the node positions in data as a numpy array that we are free
#  to modify as we wish. For the sake of the example, let's set the first
#  point to zero.
data[0,:] = np.array([0.,0.,0.],dtype=np.double) # could be done in a simpler way too

# We could also modify the header if we so desired...


## Now we store the file back to MPIO
coord_file = 'c-COORD-modif.mpio.bin'
pyQvarsi.io.AlyaMPIO_write(coord_file,data,header,partition_file)


pyQvarsi.pprint(0,'Done!')
pyQvarsi.cr_info()