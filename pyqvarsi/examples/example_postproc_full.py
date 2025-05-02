#!/bin/env python
#
# Example how to perform time averages and extract
# budgets in parallel using pyQvarsi.
#
# Last rev: 25/05/2022
from __future__ import print_function, division

# Please do not delete this part otherwise it will not work
# you have been warned after a long weekend of debugging
import mpi4py
mpi4py.rc.recv_mprobe = False

import os, numpy as np
import pyQvarsi


## Parameters
rho, mu     = 1.0, 5.0e-6 # Al tanto meter la correcta

CASESTR     = 'cas'
BASEDIR     = '/gpfs/scratch/bsc21/bsc21703/30p30n_prace/m1_bl'
ALT_BASEDIR = '/gpfs/scratch/bsc21/bsc21703/30p30n_prace/m1_bl/post/k'
BINDIR      = os.path.join(BASEDIR,'/bins')
VARLIST     = ['PRESS', 'VELOC', 'TURBU']


## We control the sampling time here
START, DT, END = 140010,10,150020
listOfInstants = [ii for ii in range(START,END+DT,DT)]


## Create the subdomain mesh
# Commu must be read for correct gradient computation
mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,alt_basedir=ALT_BASEDIR,read_commu=True,read_massm=True)
pyQvarsi.pprint(0,'Run (%d instants)...' % len(listOfInstants),flush=True)


## Accumulate the statistics (auxiliar according to Table 5)
stats = pyQvarsi.Field(xyz  = pyQvarsi.truncate(mesh.xyz,6),
					# Here are the mandatory defined in Table 2 of HiFiTurb (GA 814837)
					# Level 1 - averaged Navier-Stokes equations
					AVPRE = mesh.newArray(),        # Averaged pressure
					AVVEL = mesh.newArray(ndim=3),  # Averaged velocity
					AVTUR = mesh.newArray(),        # Averaged subgrid scale viscosity
					AVTEM = mesh.newArray(),        # Averaged temperature
					GRAVP = mesh.newArray(ndim=3),  # Averaged gradient of pressure
					GRAVV = mesh.newArray(ndim=9),  # Averaged gradient of velocity
					AVHFL = mesh.newArray(ndim=3),	# Averaged heat flux
					AVSTR = mesh.newArray(ndim=9),  # Averaged strain rate
					AVROT = mesh.newArray(ndim=9),  # Averaged rotation rate
					AVSHE = mesh.newArray(ndim=9),  # Averaged shear stresses
					RESTR = mesh.newArray(ndim=9),  # Reynolds stresses
					AVSTF = mesh.newArray(ndim=9),  # Averaged strain rate
					AVRTF = mesh.newArray(ndim=9),  # Averaged rotation rate
					AVTHF = mesh.newArray(ndim=3),	# Averaged turbulent heat flux
					# Level 1 - additional quantities
					AVPF2 = mesh.newArray(),		# Pressure autocorrelation
					AVTF2 = mesh.newArray(),		# Temperature autocorrelation
					TAYMS = mesh.newArray(),        # Taylor microscale
					KOLLS = mesh.newArray(),        # Kolmogorov lenghtscale
					KOLTS = mesh.newArray(),        # Kolmogorov timescale
					# Level 2 - Reynolds stress equations budget terms
					CONVE = mesh.newArray(ndim=9),  # Convection
					PRODU = mesh.newArray(ndim=9),  # Production
					DIFF1 = mesh.newArray(ndim=9),  # Turbulent diffusion 1
					DIFF2 = mesh.newArray(ndim=9),  # Turbulent diffusion 2
					DIFF3 = mesh.newArray(ndim=9),  # Molecular diffusion
					PSTRA = mesh.newArray(ndim=9),  # Pressure strain
					DISSI = mesh.newArray(ndim=9),  # Dissipation
					# Level 2 - Reynolds stress equations - separate terms
					AVPVE = mesh.newArray(ndim=3),  # Pressure velocity correlation
					AVVE3 = mesh.newArray(ndim=27)  # Triple velocity correlation
				   )


## First loop, compute AVVEL, AVPRE and AVTUR
time = 0
for instant in listOfInstants:
	if instant%100 == 0: pyQvarsi.pprint(1,'Loop 1, instant %d...'%instant,flush=True)

	# Read field
	fields,header = pyQvarsi.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=BINDIR)

	# Compute time-weighted average 
	dt    = 1  #header.time - time  # weight
	time += dt #header.time         # sum_weights

	# Accumulate the statistics
	stats['AVTUR'] += pyQvarsi.stats.addS1(stats['AVTUR'],fields['TURBU'],w=1. if instant == START else dt/time)
	stats['AVPRE'] += pyQvarsi.stats.addS1(stats['AVPRE'],fields['PRESS'],w=1. if instant == START else dt/time)
	stats['AVVEL'] += pyQvarsi.stats.addS1(stats['AVVEL'],fields['VELOC'],w=1. if instant == START else dt/time)

	# Compute gradient of velocity
#	gradv = mesh.gradient(fields['VELOC'])
#	fields['VORTI'] = pyQvarsi.postproc.vorticity(gradv)
#	fields['QCRIT'] = pyQvarsi.postproc.QCriterion(gradv)
#
#	# Store field
#	fields.write(CASESTR,instant,time,basedir=BASEDIR,exclude_vars=VARLIST)

# Gradients of averaged velocity and pressure
stats['GRAVP'] = mesh.gradient(stats['AVPRE'])
stats['GRAVV'] = mesh.gradient(stats['AVVEL'])


## Second loop, compute RESTR
time = 0
for instant in listOfInstants:
	if instant%100 == 0: pyQvarsi.pprint(1,'Loop 2, instant %d...'%instant,flush=True)

	# Read field
	fields,header = pyQvarsi.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=BINDIR)

	# Compute time-weighted average 
	dt    = 1  #header.time - time  # weight
	time += dt #header.time         # sum_weights

	# Postprocess fields
	GRADV  = mesh.gradient(fields['VELOC'])
	STRAI  = pyQvarsi.stats.strainTensor(GRADV)
	ROTAT  = pyQvarsi.stats.vorticityTensor(GRADV)
	SHEAR  = 2.*mu*STRAI

	# Fluctuations
	PFLUC  = pyQvarsi.math.linopScaf(1.,fields['PRESS'],-1.,stats['AVPRE'])  # p' = p - <p>
	VFLUC  = pyQvarsi.math.linopArrf(1.,fields['VELOC'],-1.,stats['AVVEL'])  # u' = u - <u>
	PFLU2  = PFLUC*PFLUC
	PVCOR  = pyQvarsi.math.scaVecProd(PFLUC,VFLUC)
	VELO3  = rho*pyQvarsi.stats.tripleCorrelation(VFLUC,VFLUC,VFLUC)

	GRVFL  = pyQvarsi.math.linopArrf(1.,GRADV,-1.,stats['GRAVV'])
	RESTR  = rho*pyQvarsi.stats.reynoldsStressTensor(VFLUC)
	STRAF  = pyQvarsi.stats.strainTensor(GRVFL)
	ROTAF  = pyQvarsi.stats.vorticityTensor(GRVFL)

	# Budgets
	PSTRA  = pyQvarsi.stats.pressureStrainBudget(PFLUC,STRAF)
	DISSI  = pyQvarsi.stats.dissipationBudget(mu+fields['TURBU'],GRVFL)

	# Accumulate statistics
	stats['AVPVE'] += pyQvarsi.stats.addS1(stats['AVPVE'],PVCOR,w=1. if instant == START else dt/time)
	stats['AVPF2'] += pyQvarsi.stats.addS1(stats['AVPF2'],PFLU2,w=1. if instant == START else dt/time)
	stats['AVVE3'] += pyQvarsi.stats.addS1(stats['AVVE3'],VELO3,w=1. if instant == START else dt/time)
	stats['RESTR'] += pyQvarsi.stats.addS1(stats['RESTR'],RESTR,w=1. if instant == START else dt/time)
	stats['AVSTR'] += pyQvarsi.stats.addS1(stats['AVSTR'],STRAI,w=1. if instant == START else dt/time)
	stats['AVROT'] += pyQvarsi.stats.addS1(stats['AVROT'],ROTAT,w=1. if instant == START else dt/time)
	stats['AVSHE'] += pyQvarsi.stats.addS1(stats['AVSHE'],SHEAR,w=1. if instant == START else dt/time)
	stats['AVSTF'] += pyQvarsi.stats.addS1(stats['AVSTF'],STRAF,w=1. if instant == START else dt/time)
	stats['AVRTF'] += pyQvarsi.stats.addS1(stats['AVRTF'],ROTAF,w=1. if instant == START else dt/time)

	stats['PSTRA'] += pyQvarsi.stats.addS1(stats['PSTRA'],PSTRA,w=1. if instant == START else dt/time)
	stats['DISSI'] += pyQvarsi.stats.addS1(stats['DISSI'],DISSI,w=1. if instant == START else dt/time)

# Finish budgets
stats['CONVE'] = pyQvarsi.stats.convectionBudget(stats['AVVEL'],mesh.gradient(stats['RESTR']))
stats['PRODU'] = pyQvarsi.stats.productionBudget(stats['RESTR'],stats['GRAVV'])
stats['DIFF1'] = pyQvarsi.stats.turbulentDiffusion1Budget(rho,stats['AVVE3'],mesh)
stats['DIFF2'] = pyQvarsi.stats.turbulentDiffusion2Budget(stats['AVPVE'],mesh)
stats['DIFF3'] = pyQvarsi.stats.molecularDiffusionBudget(mu,stats['RESTR'],mesh)

# Budgets residual
stats['RESID'] = stats['PRODU'] + stats['DIFF1'] + stats['DIFF2'] \
			   + stats['DIFF3'] + stats['PSTRA'] - stats['DISSI']

# Compute TKE and dissipation
K     = pyQvarsi.stats.TKE(stats['RESTR'])
prod  = 0.5*pyQvarsi.math.trace(stats['PRODU'])
dissi = 0.5*pyQvarsi.math.trace(stats['DISSI']) # e = 1/2*e_ii = 2*mu*<S'_ij S'_ij>

# Compute Taylor microscale and Kolmogorov length and time scales
stats['TAYMS'] = pyQvarsi.stats.taylorMicroscale(mu/rho,K,dissi)
stats['KOLLS'] = pyQvarsi.stats.kolmogorovLengthScale(mu/rho,dissi)
stats['KOLTS'] = pyQvarsi.stats.kolmogorovTimeScale(mu/rho,dissi)


## Store outputs
pyQvarsi.pprint(1,'Writing MPIO...',flush=True)
stats.write(CASESTR,123,0.,basedir=ALT_BASEDIR,exclude_vars=[
	'AVTEM','AVHFL','AVSTR','AVROT','AVSHE','AVSTF','AVRTF','AVTHF',
	'AVPF2','AVTF2','AVPVE','AVVE3'])
pyQvarsi.cr_info()