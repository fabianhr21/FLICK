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
CHEKP_FREQ     = 100 # Frequency of checkpoints

## Create a checkpoint class
# This creates a new class to manage the checkpoints or reads
# any existing checkpoint for this code.
checkp = pyQvarsi.Checkpoint.create(CHEKP_FREQ,START,END,step=DT)

# Are we restarting or are we running for the first time?
if checkp.restarted:
	# We are restarting a case
	pyQvarsi.pprint(0,'Restarting (%d instants)...' % len(checkp.listrange),flush=True)

	# Recover mesh and stats from checkpoint
	time  = checkp['time']  # We store the time for the averaging algorithm
	mesh  = checkp['mesh']  # We store the mesh so we don't have to read it again
	stats = checkp['stats'] # We store the stats which are our results
else:
	# We are not restarting
	pyQvarsi.pprint(0,'First run (%d instants)...' % len(checkp.listrange),flush=True)
	time = 0 # Set time to 0 for the first loop

	# Create the subdomain mesh
	mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=True,read_massm=False)


	## Accumulate the statistics (auxiliar according to Table 5)
	stats = pyQvarsi.Field(xyz   = pyQvarsi.truncate(mesh.xyz,6),
						 RESTR = mesh.newArray(ndim=9),  # Reynolds stresses
						 TURKE = mesh.newArray(),        # Dissipation
						 DISSI = mesh.newArray(),        # Turbulent kinetic energy
						 TAYMS = mesh.newArray(),        # Taylor microscale
						 KOLLS = mesh.newArray(),        # Kolmogorov lenghtscale
						 KOLTS = mesh.newArray(),        # Kolmogorov timescale
					    )

	# Force first restart
	# Set a flag of 1 for the first loop (so that we know where we are)
	# and set the initial instant to -1 (flag to allow the correct range to be used)
	# Mesh only needs to be saved once at the start since it does not change
	checkp.force_save(1,-1,'Checkpoint start first loop',time=time,mesh=mesh,stats=stats)

## First loop in time
# Only accumulate pressure, velocity and temperature (if available) to
# obtain the fluctuations on the next loop
# We will only enter here if the checkpoint flag is equal to 1, else skip this part (already done)
if checkp.enter_part(flag=1):
	# Loop over instants
	for instant in checkp.listrange:
		# Save a checkpoint if needed
		# The frequency condition is already inside the checkpoint as well as the counter
		# if we don't need to save we increase the counter
		checkp.save(1,instant,'Checkpoint instant %d...'%instant,time=time,stats=stats)

		# Read field
		fields, header = pyQvarsi.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=BASEDIR)

		if mesh.comm.rank == 0: continue # Skip master

		# Compute time-weighted average 
		dt   = header.time - time  # weight
		time = header.time         # sum_weights

		stats['AVVEL'] += pyQvarsi.stats.addS1(stats['AVVEL'],fields['VELOC'],w=1. if instant == START else dt/time)

	# We force the time to 0 here before forcing to save a restart and changing the flag
	time = 0
	checkp.reset(flag=2)
	checkp.force_save(2,-1,'Checkpoint end first loop',time=time,stats=stats)


## Do a second loop in time
# This time compute all the necessary magnitudes 
# and accumulate them as needed
# This is the second part of the code so flag == 2
if checkp.enter_part(flag=2):
	# Loop over instants
	for instant in checkp.listrange:
		# Save a checkpoint if needed
		# The frequency condition is already inside the checkpoint as well as the counter
		# if we don't need to save we increase the counter
		checkp.save(2,instant,'Checkpoint instant %d...'%instant,time=time,stats=stats)

		# Read field
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

	# Force storing a checkpoint here
	# We force a checkpoint before the 3rd part of the code and we change the flag
	checkp.reset(flag=3)
	checkp.force_save(3,-1,'Checkpoint end second loop',time=time,stats=stats)

# Third part of the code here
if checkp.enter_part(flag=3):
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

	# Eventually we can clean all restarts at the end of the code
	# Otherwise we can do it manually
	checkp.clean()