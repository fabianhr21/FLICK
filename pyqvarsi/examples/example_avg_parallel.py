#!/usr/bin/env python
#
# Example how to perform time averages in
# parallel using the Mesh and Field classes.
#
# Last revision: 22/10/2020
from __future__ import print_function, division

import numpy as np

import pyQvarsi


BASEDIR = './'
CASESTR = 'channel'
VARLIST = ['VELOC','PRESS']

START, END, DT = 10000, 200000, 100
listOfInstants = [ii for ii in range(START,END+DT,DT)]

# Parameters
rho = 1.0
mu  = 5.3566e-05


# Create the subdomain mesh
# Ensure in the channel.ker.dat to have:
#	POSTPROCESS COMMU $ Communications matrix
#	POSTPROCESS MASSM $ Mass matrix
mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=True,read_massm=False)

pyQvarsi.pprint(0,'Averaging %d instants' % len(listOfInstants),flush=True)

## Temporal average algorithm
# Loop the instants and temporally average the data. 
# The data is stored inside Field class.

# Read the first instant of the list
avgField, header = pyQvarsi.Field.read(CASESTR,VARLIST,listOfInstants[0],mesh.xyz,basedir=BASEDIR)
time = header.time

for instant in listOfInstants[1:]:
	field, header = pyQvarsi.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=BASEDIR)

	if mesh.comm.rank == 0: continue # Skip master

	# Compute time-weighted average (Welford's online algorithm)
	dt        = header.time - time         # weight
	time      = header.time                # sum_weights
	avgField += dt/time*(field - avgField) # weight/sum_weights(value-meanval)

# Rename the variables within the field
avgField.rename('AVVEL','VELOC') # VELOC -> AVVEL
avgField.rename('AVPRE','PRESS') # PRESS -> AVPRE


## Reduction step
# At this point we can use a normal reduction operation with the
# fast reduce operator to obtain the global average. 
avgFieldG = avgField.reduce(root=0,op=pyQvarsi.fieldFastReduce)

# Master prints the averaged results
pyQvarsi.pprint(0,'Reduction step done!',flush=True)
pyQvarsi.pprint(0,avgFieldG)

pyQvarsi.cr_info()