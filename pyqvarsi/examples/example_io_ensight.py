#!/bin/env python
#
# Example of computing the RANS budgets on a channel flow
# data in ENSIGHT GOLD format.
#
# Last rev: 16/08/2021
from __future__ import print_function, division

# Please do not delete this part otherwise it will not work
# you have been warned after a long weekend of debugging
import mpi4py
mpi4py.rc.recv_mprobe = False

import os,numpy as np
import pyQvarsi


# Parameters
rho, mu = 1.0, 0.00556

BASEDIR        = '1.t-0-60-copy.bin'
ALT_BASEDIR    = 'out'
CASESTR        = 'chan'
VARLIST        = ['VELOC','PRESS']
START, DT, END = 2,1,415

FILE_FMT       = 'ensight' #'mpio'

# In case of restart, load the previous data
listOfInstants = [ii for ii in range(START,END+DT,DT)]


## Create the subdomain mesh
mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,alt_basedir=ALT_BASEDIR,fmt=FILE_FMT)#,read_commu=True,read_massm=True)

pyQvarsi.pprint(0,'Run (%d instants)...' % len(listOfInstants),flush=True)


## Accumulate the statistics (auxiliar according to Table 5)
stats = pyQvarsi.Field(xyz  = pyQvarsi.truncate(mesh.xyz,6),
					AVPRE = mesh.newArray(),        # Averaged pressure
					AVVEL = mesh.newArray(ndim=3),  # Averaged velocity
					GRAVP = mesh.newArray(ndim=3),  # Averaged gradient of pressure
					GRAVV = mesh.newArray(ndim=6),  # Averaged gradient of velocity
					AVSTR = mesh.newArray(ndim=9),  # Averaged strain rate
					AVROT = mesh.newArray(ndim=9),  # Averaged rotation rate
					AVSHE = mesh.newArray(ndim=9),  # Averaged shear stresses
					RESTR = mesh.newArray(ndim=6),  # Reynolds stresses
					AVSTF = mesh.newArray(ndim=9),  # Averaged strain rate
					AVRTF = mesh.newArray(ndim=9),  # Averaged rotation rate
					AVPF2 = mesh.newArray(),		# Pressure autocorrelation
					TAYMS = mesh.newArray(),        # Taylor microscale
					KOLLS = mesh.newArray(),        # Kolmogorov lenghtscale
					KOLTS = mesh.newArray(),        # Kolmogorov timescale
					CONVE = mesh.newArray(ndim=6),  # Convection
					PRODU = mesh.newArray(ndim=6),  # Production
					DIFF1 = mesh.newArray(ndim=6),  # Turbulent diffusion 1
					DIFF2 = mesh.newArray(ndim=6),  # Turbulent diffusion 2
					DIFF3 = mesh.newArray(ndim=6),  # Molecular diffusion
					PSTRA = mesh.newArray(ndim=6),  # Pressure strain
					DISSI = mesh.newArray(ndim=6),  # Dissipation
					AVPVE = mesh.newArray(ndim=3),  # Pressure velocity correlation
					AVVE3 = mesh.newArray(ndim=27)  # Triple velocity correlation
				   )


# Only accumulate pressure, velocity and temperature (if available) to
# obtain the fluctuations on the next loop
time = 0
for instant in listOfInstants:
	if instant%100 == 0: pyQvarsi.pprint(1,'First loop, instant %d...'%instant,flush=True)
	# Read field
	fields, header = pyQvarsi.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=BASEDIR,fmt=FILE_FMT)

	# Compute time-weighted average 
	dt   = 1#header.time - time  # weight
	time += dt#header.time         # sum_weights

	# Accumulate the statistics
	stats['AVPRE'] += pyQvarsi.stats.addS1(stats['AVPRE'],fields['PRESS'],w=1. if instant == START else dt/time)
	stats['AVVEL'] += pyQvarsi.stats.addS1(stats['AVVEL'],fields['VELOC'],w=1. if instant == START else dt/time)

# Gradients of averaged velocity and pressure
# only computed once
stats['GRAVP'] = mesh.gradient(stats['AVPRE'])
gravv          = mesh.gradient(stats['AVVEL'])
stats['GRAVV'][:,0] = gravv[:,0] # XX
stats['GRAVV'][:,1] = gravv[:,4] # YY
stats['GRAVV'][:,2] = gravv[:,8] # ZZ
stats['GRAVV'][:,3] = gravv[:,1] # XY
stats['GRAVV'][:,4] = gravv[:,2] # XZ
stats['GRAVV'][:,5] = gravv[:,5] # YZ

## Do a second loop in time
# This time compute all the necessary magnitudes and accumulate 
# them as needed 
time  = 0
restr = mesh.newArray(ndim=9)
pstra = mesh.newArray(ndim=9)
dissi = mesh.newArray(ndim=9)
for instant in listOfInstants:
	if instant%100 == 0: pyQvarsi.pprint(1,'Second loop, instant %d...'%instant,flush=True)
	
	# Read field
	fields, header = pyQvarsi.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=BASEDIR,fmt=FILE_FMT)

	# Compute time-weighted average 
	dt   = 1#header.time - time  # weight
	time += dt#header.time         # sum_weights

	# Postprocess fields
	fields['GRADV'] = mesh.gradient(fields['VELOC'])
	fields['STRAI'] = pyQvarsi.stats.strainTensor(fields['GRADV'])
	fields['ROTAT'] = pyQvarsi.stats.vorticityTensor(fields['GRADV'])
	fields['SHEAR'] = 2.*mu*fields['STRAI']

	# Fluctuations
	fields['PFLUC'] = pyQvarsi.math.linopScaf(1.,fields['PRESS'],-1.,stats['AVPRE'])  # p' = p - <p>
	fields['VFLUC'] = pyQvarsi.math.linopArrf(1.,fields['VELOC'],-1.,stats['AVVEL'])  # u' = u - <u>
	fields['PFLU2'] = fields['PFLUC']*fields['PFLUC']
	fields['PVCOR'] = pyQvarsi.math.scaVecProd(fields['PFLUC'],fields['VFLUC'])
	fields['VELO3'] = rho*pyQvarsi.stats.tripleCorrelation(fields['VFLUC'],fields['VFLUC'],fields['VFLUC'])

	fields['GRVFL'] = pyQvarsi.math.linopArrf(1.,fields['GRADV'],-1.,stats['GRAVV'])
	fields['RESTR'] = rho*pyQvarsi.stats.reynoldsStressTensor(fields['VFLUC'])
	fields['STRAF'] = pyQvarsi.stats.strainTensor(fields['GRVFL'])
	fields['ROTAF'] = pyQvarsi.stats.vorticityTensor(fields['GRVFL'])

	# Budgets
	fields['PSTRA'] = pyQvarsi.stats.pressureStrainBudget(fields['PFLUC'],fields['STRAF'])
	fields['DISSI'] = pyQvarsi.stats.dissipationBudget(mu,fields['GRVFL'])

	# Accumulate statistics
	stats['AVPVE'] += pyQvarsi.stats.addS1(stats['AVPVE'],fields['PVCOR'],w=1. if instant == START else dt/time)
	stats['AVPF2'] += pyQvarsi.stats.addS1(stats['AVPF2'],fields['PFLU2'],w=1. if instant == START else dt/time)
	stats['AVVE3'] += pyQvarsi.stats.addS1(stats['AVVE3'],fields['VELO3'],w=1. if instant == START else dt/time)
	restr          += pyQvarsi.stats.addS1(stats['RESTR'],fields['RESTR'],w=1. if instant == START else dt/time)
	stats['AVSTR'] += pyQvarsi.stats.addS1(stats['AVSTR'],fields['STRAI'],w=1. if instant == START else dt/time)
	stats['AVROT'] += pyQvarsi.stats.addS1(stats['AVROT'],fields['ROTAT'],w=1. if instant == START else dt/time)
	stats['AVSHE'] += pyQvarsi.stats.addS1(stats['AVSHE'],fields['SHEAR'],w=1. if instant == START else dt/time)
	stats['AVSTF'] += pyQvarsi.stats.addS1(stats['AVSTF'],fields['STRAF'],w=1. if instant == START else dt/time)
	stats['AVRTF'] += pyQvarsi.stats.addS1(stats['AVRTF'],fields['ROTAF'],w=1. if instant == START else dt/time)

	pstra += pyQvarsi.stats.addS1(stats['PSTRA'],fields['PSTRA'],w=1. if instant == START else dt/time)
	dissi += pyQvarsi.stats.addS1(stats['DISSI'],fields['DISSI'],w=1. if instant == START else dt/time)

# Compute TKE and dissipation
k    = pyQvarsi.stats.TKE(restr)
epsi = 0.5*pyQvarsi.math.trace(dissi) # e = 1/2*e_ii = 2*mu*<S'_ij S'_ij>

# Compute Taylor microscale and Kolmogorov length and time scales
stats['TAYMS'] = pyQvarsi.stats.taylorMicroscale(mu/rho,k,epsi)
stats['KOLLS'] = pyQvarsi.stats.kolmogorovLengthScale(mu/rho,epsi)
stats['KOLTS'] = pyQvarsi.stats.kolmogorovTimeScale(mu/rho,epsi)

# Finish budgets
conve = pyQvarsi.stats.convectionBudget(stats['AVVEL'],mesh.gradient(restr))
produ = pyQvarsi.stats.productionBudget(stats['RESTR'],gravv)
diff1 = pyQvarsi.stats.turbulentDiffusion1Budget(rho,stats['AVVE3'],mesh)
diff2 = pyQvarsi.stats.turbulentDiffusion2Budget(stats['AVPVE'],mesh)
diff3 = pyQvarsi.stats.molecularDiffusionBudget(mu,restr,mesh)

prod  = 0.5*pyQvarsi.math.trace(produ)

# Convert to symmetric tensors
stats['RESTR'][:,0] = restr[:,0] # XX
stats['RESTR'][:,1] = restr[:,4] # YY
stats['RESTR'][:,2] = restr[:,8] # ZZ
stats['RESTR'][:,3] = restr[:,1] # XY
stats['RESTR'][:,4] = restr[:,2] # XZ
stats['RESTR'][:,5] = restr[:,5] # YZ

stats['CONVE'][:,0] = conve[:,0] # XX
stats['CONVE'][:,1] = conve[:,4] # YY
stats['CONVE'][:,2] = conve[:,8] # ZZ
stats['CONVE'][:,3] = conve[:,1] # XY
stats['CONVE'][:,4] = conve[:,2] # XZ
stats['CONVE'][:,5] = conve[:,5] # YZ

stats['PRODU'][:,0] = produ[:,0] # XX
stats['PRODU'][:,1] = produ[:,4] # YY
stats['PRODU'][:,2] = produ[:,8] # ZZ
stats['PRODU'][:,3] = produ[:,1] # XY
stats['PRODU'][:,4] = produ[:,2] # XZ
stats['PRODU'][:,5] = produ[:,5] # YZ

stats['DIFF1'][:,0] = diff1[:,0] # XX
stats['DIFF1'][:,1] = diff1[:,4] # YY
stats['DIFF1'][:,2] = diff1[:,8] # ZZ
stats['DIFF1'][:,3] = diff1[:,1] # XY
stats['DIFF1'][:,4] = diff1[:,2] # XZ
stats['DIFF1'][:,5] = diff1[:,5] # YZ

stats['DIFF2'][:,0] = diff2[:,0] # XX
stats['DIFF2'][:,1] = diff2[:,4] # YY
stats['DIFF2'][:,2] = diff2[:,8] # ZZ
stats['DIFF2'][:,3] = diff2[:,1] # XY
stats['DIFF2'][:,4] = diff2[:,2] # XZ
stats['DIFF2'][:,5] = diff2[:,5] # YZ

stats['DIFF3'][:,0] = diff3[:,0] # XX
stats['DIFF3'][:,1] = diff3[:,4] # YY
stats['DIFF3'][:,2] = diff3[:,8] # ZZ
stats['DIFF3'][:,3] = diff3[:,1] # XY
stats['DIFF3'][:,4] = diff3[:,2] # XZ
stats['DIFF3'][:,5] = diff3[:,5] # YZ

stats['PSTRA'][:,0] = pstra[:,0] # XX
stats['PSTRA'][:,1] = pstra[:,4] # YY
stats['PSTRA'][:,2] = pstra[:,8] # ZZ
stats['PSTRA'][:,3] = pstra[:,1] # XY
stats['PSTRA'][:,4] = pstra[:,2] # XZ
stats['PSTRA'][:,5] = pstra[:,5] # YZ

stats['DISSI'][:,0] = dissi[:,0] # XX
stats['DISSI'][:,1] = dissi[:,4] # YY
stats['DISSI'][:,2] = dissi[:,8] # ZZ
stats['DISSI'][:,3] = dissi[:,1] # XY
stats['DISSI'][:,4] = dissi[:,2] # XZ
stats['DISSI'][:,5] = dissi[:,5] # YZ


## Write output
mesh.write(CASESTR,basedir=ALT_BASEDIR,fmt=FILE_FMT)
stats.write(CASESTR,1,0.,basedir=ALT_BASEDIR,fmt=FILE_FMT,exclude_vars=[
		'AVSTR','AVROT','AVSHE','AVSTF','AVRTF','AVPF2','AVTF2','AVPVE','AVVE3'])

pyQvarsi.io.Ensight_writeCase(
	os.path.join(ALT_BASEDIR,'%s.ensi.case'%CASESTR),
	'%s.ensi.geo'%CASESTR,
	[
		{'name':'AVPRE','dims':1,'file':'%s.ensi.AVPRE-******'%CASESTR}, # Averaged pressure
		{'name':'AVVEL','dims':3,'file':'%s.ensi.AVVEL-******'%CASESTR}, # Averaged velocity
		{'name':'GRAVP','dims':3,'file':'%s.ensi.GRAVP-******'%CASESTR}, # Averaged gradient of pressure
		{'name':'GRAVV','dims':6,'file':'%s.ensi.GRAVV-******'%CASESTR}, # Averaged gradient of velocity
		{'name':'RESTR','dims':6,'file':'%s.ensi.RESTR-******'%CASESTR}, # Reynolds stresses
		{'name':'TAYMS','dims':1,'file':'%s.ensi.TAYMS-******'%CASESTR}, # Taylor microscale
		{'name':'KOLLS','dims':1,'file':'%s.ensi.KOLLS-******'%CASESTR}, # Kolmogorov lenghtscale
		{'name':'KOLTS','dims':1,'file':'%s.ensi.KOLTS-******'%CASESTR}, # Kolmogorov timescale
		{'name':'CONVE','dims':6,'file':'%s.ensi.CONVE-******'%CASESTR}, # Convection
		{'name':'PRODU','dims':6,'file':'%s.ensi.PRODU-******'%CASESTR}, # Production
		{'name':'DIFF1','dims':6,'file':'%s.ensi.DIFF1-******'%CASESTR}, # Turbulent diffusion 1
		{'name':'DIFF2','dims':6,'file':'%s.ensi.DIFF2-******'%CASESTR}, # Turbulent diffusion 2
		{'name':'DIFF3','dims':6,'file':'%s.ensi.DIFF3-******'%CASESTR}, # Molecular diffusion
		{'name':'PSTRA','dims':6,'file':'%s.ensi.PSTRA-******'%CASESTR}, # Pressure strain
		{'name':'DISSI','dims':6,'file':'%s.ensi.DISSI-******'%CASESTR}, # Dissipation
	],
	np.array([0.],np.double)
)

pyQvarsi.cr_info()