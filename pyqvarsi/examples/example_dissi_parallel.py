#!/usr/bin/env python
#
# Example how to extract the dissipation and
# the Kolmogorov scales in parallel.
#
# Last rev: 18/11/2020
from __future__ import print_function, division

import numpy as np

import pyQvarsi

# Parameters
rho, mu = 1.0, 5.3566e-05

BASEDIR        = './'
CASESTR        = 'channel'
VARLIST        = ['VELOC']
START, DT, END = 1000,100,244600 

# Build list of instants
listOfInstants = [ii for ii in range(START,END+DT,DT)]

# Create the subdomain mesh
mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=True,read_massm=False)

pyQvarsi.pprint(0,'Averaging %d instants' % len(listOfInstants),flush=True)


## Accumulate the statistics (auxiliar according to Table 5)
stats = pyQvarsi.Field(xyz   = pyQvarsi.truncate(mesh.xyz,6),
					 RESTR = mesh.newArray(ndim=9),  # Reynolds stresses
					 TURKE = mesh.newArray(),        # Dissipation
					 DISSI = mesh.newArray(),        # Turbulent kinetic energy
					 TAYMS = mesh.newArray(),        # Taylor microscale
					 KOLLS = mesh.newArray(),        # Kolmogorov lenghtscale
					 KOLTS = mesh.newArray(),        # Kolmogorov timescale
				    )


## First loop in time
# Only accumulate pressure, velocity and temperature (if available) to
# obtain the fluctuations on the next loop

# Loop over instants
for instant in listOfInstant:
	fields, header = pyQvarsi.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=BASEDIR)

	if mesh.comm.rank == 0: continue # Skip master

	# Compute time-weighted average 
	dt   = header.time - time  # weight
	time = header.time         # sum_weights

	stats['AVVEL'] += pyQvarsi.stats.addS1(stats['AVVEL'],fields['VELOC'],w=1. if instant == START else dt/time)


## Do a second loop in time
# This time compute all the necessary magnitudes 
# and accumulate them as needed 

# Loop over instants
for instant in listOfInstants:
	fields, header = pyQvarsi.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=BASEDIR)

	if mesh.comm.rank == 0: continue # Skip master

	# Compute time-weighted average 
	dt   = header.time - time  # weight
	time = header.time         # sum_weights

	# Fluctuations
	fields['VFLUC'] = fields['VELOC'] - stats['AVVEL'] # u'   = u - <u>
	fields['RESTR'] = pyQvarsi.stats.reynoldsStressTensor(fields['VFLUC'])
	fields['GRVFL'] = mesh.gradient(fields['VFLUC'])
	fields['STRAF'] = pyQvarsi.stats.strainTensor(fields['GRVFL'])

	# Accumulate S'ijS'ij from the dissipation
	fields['DISSI'] = pyQvarsi.math.doubleDot(fields['STRAF'],fields['STRAF'])

	# Accumulate statistics
	stats['RESTR'] += pyQvarsi.stats.addS1(stats['RESTR'],fields['RESTR'],w=1. if instant == START else dt/time)
	stats['DISSI'] += pyQvarsi.stats.addS1(stats['DISSI'],fields['DISSI'],w=1. if instant == START else dt/time)

# Compute TKE and dissipation
stats['TURKE']  = pyQvarsi.stats.TKE(stats['RESTR'])
stats['DISSI'] *= 2.0*mu # e = 2*mu*<S'_ij S'_ij>

# Compute Taylor microscale and Kolmogorov length and time scales
stats['TAYMS'] = pyQvarsi.stats.taylorMicroscale(mu/rho,k,dissi)
stats['KOLLS'] = pyQvarsi.stats.kolmogorovLengthScale(mu/rho,dissi)
stats['KOLTS'] = pyQvarsi.stats.kolmogorovTimeScale(mu/rho,dissi)


## Reduction step
# At this point we can use a normal reduction operation with the
# fast reduce operator to obtain the global average. 
statsG = stats.reduce(root=0,op=pyQvarsi.fieldFastReduce)

# Master prints the averaged results
pyQvarsi.pprint(0,'Reduction step done!',flush=True)
pyQvarsi.pprint(0,statsG)

pyQvarsi.cr_info()