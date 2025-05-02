#!/usr/bin/env python
#
# Example how to perform time averages and the
# reduction in the y dimension in parallel 
# using the Mesh and Field classes.
#
# Last revision: 27/10/2020
from __future__ import print_function, division

import matplotlib, numpy as np
matplotlib.use('Agg') # Headless plot
import matplotlib.pyplot as plt

import pyQvarsi


BASEDIR        = './'                                                                                                                            
CASESTR        = 'channel'                                                                                                                        
VARLIST        = ['VELOC','PRESS']                                                                                                                
START, DT, END = 500000, 100, 1000000                                                                                                             
listOfInstants = [ii for ii in range(START,END+DT,DT)] 

# Parameters
rho = 1.0
mu  = 0.0003546986585888876
# NOTE: make sure to set the correct viscosity for your channel!


# Create the subdomain mesh
# Ensure in the channel.ker.dat to have:
#	POSTPROCESS COMMU $ Communications matrix
#	POSTPROCESS MASSM $ Mass matrix
mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=True,read_massm=False)

pyQvarsi.pprint(0,'Averaging %d instants' % len(listOfInstants),flush=True)

# Create a stats field where to accumulate the output statistics
stats = pyQvarsi.Field(xyz   = pyQvarsi.truncate(mesh.xyz,6),
					 AVVEL = mesh.newArray(ndim=3), # Vectorial field
					 AVVE2 = mesh.newArray(ndim=3), # Vectorial field
					 RETAU = mesh.newArray()        # Scalar field
				    )


## Temporal average algorithm
# Loop the instants and temporally average the data. 
# The data is stored inside Field class.
time = 0.
for instant in listOfInstants:
	field, header = pyQvarsi.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=BASEDIR)

	if mesh.comm.rank == 0: continue # Skip master

	# Compute time-weighted average (Welford's online algorithm)
	dt    = 1.  # weight
	time += dt  # sum_weights
	# For a normal average set dt = 1 and time += dt

	stats['AVVEL'] += pyQvarsi.stats.addS1(stats['AVVEL'],field['VELOC'],w=dt/time)
	stats['AVVE2'] += pyQvarsi.stats.addS1(stats['AVVE2'],field['VELOC']*field['VELOC'],w=dt/time)

# Compute the derivative of AVVEL to compute the RETAU
# from the averaged field data
gradv = mesh.gradient(stats['AVVEL'])
tw    = mu*np.abs(gradv[:,1]) # du/dy
stats['RETAU'] = np.sqrt(tw/rho)/mu


## Average in Y direction
avgYStats,_ = pyQvarsi.postproc.directionAvg(stats,direction='y')
# At this point we have a line in the y direction per each subdomain
# Add a variable COUNT that will help on the average when reducing
avgYStats['COUNT'] = np.ones((len(avgYStats,)),np.double)


## Reduction step
# At this point we can use a normal reduction operation with the
# sum operator to obtain the global average. 
# Important to have the COUNTS variable (or something similar) before reducing
avgYStatsG = avgYStats.reduce(root=0)


## Proceed with rank 0 for the final aggregation
if pyQvarsi.Communicator.serial(): # Returns true if rank == 0 or MPI_SIZE == 1

	# Finish average reduction
	avgYStatsG /= avgYStatsG['COUNT']
	print('Reduction step done!',flush=True)

	# Sort according y
	# here we use the whole xyz vector since avgYStatsG
	# only contains y directions
	avgYStatsG.sort(array='y')

	# Recover arrays
	y_pos = avgYStatsG.y 
	avvel = avgYStatsG['AVVEL']
	avve2 = avgYStatsG['AVVE2']
	retau = avgYStatsG['RETAU']

	pyQvarsi.printArray('y_pos',y_pos)
	pyQvarsi.printArray('AVVEL',avvel)
	pyQvarsi.printArray('AVVE2',avve2)

	# Midline average
	avvel = pyQvarsi.postproc.midlineAvg(avvel)
	avve2 = pyQvarsi.postproc.midlineAvg(avve2)
	retau = pyQvarsi.postproc.midlineAvg(retau)
	utau  = mu*retau[0]

	print('Re_tau %.2f' % retau[0],flush=True)

	dy      = y_pos[1]-y_pos[0]
	tw      = np.abs( mu*(avvel[1,0]-avvel[0,0])/dy )
	utau2   = np.sqrt(tw/rho)
	Re_tau2 = utau2/mu

	print('Re_tau2 %.2f' % Re_tau2,flush=True)

	# Boundary layer statistics
	mid_sz    = int(0.5*avvel.shape[0])
	bl_ystar  = y_pos[:mid_sz]*utau/mu
	bl_ustar  = avvel[:mid_sz,0]/utau
	bl_uustar = np.sqrt( avve2[:mid_sz,0]-avvel[:mid_sz,0]*avvel[:mid_sz,0] )/utau # sqrt(uu - u*u)/utau
	bl_vvstar = np.sqrt( avve2[:mid_sz,1]-avvel[:mid_sz,1]*avvel[:mid_sz,1] )/utau # sqrt(vv - v*v)/utau
	bl_wwstar = np.sqrt( avve2[:mid_sz,2]-avvel[:mid_sz,2]*avvel[:mid_sz,2] )/utau # sqrt(ww - w*w)/utau

	print('BL averages done!',flush=True)

	# Load data from DNS
	dns = np.loadtxt('./Re180_DNS.dat', skiprows=27)

	# Plot U+
	plt.figure(figsize=(6,8),dpi=100)

	plt.plot(bl_ystar[1:],bl_ustar[1:],linewidth=3.0,label='Alya')
	plt.plot(dns[:,1],dns[:,2],'k--',linewidth=3.0,label='DNS')
	plt.xlim([0.1, 2000])
	plt.ylim([0, 25])
	plt.xscale('log')
	plt.xlabel(r'$y^+$')
	plt.ylabel(r'$U^+$')
	plt.tight_layout()
	plt.savefig('Uplus.png')

	# Plot Urms+
	plt.figure(figsize=(6,8),dpi=100)

	plt.plot(bl_ystar[1:],bl_uustar[1:],linewidth=3.0,label='Alya')
	plt.plot(dns[:,1],dns[:,3],'k--',linewidth=3.0,label='DNS')
	plt.xlim([0.1, 1000])
	plt.ylim([0.0, 5.0])
	plt.xlabel(r'$y^+$')
	plt.ylabel(r'$U^{\prime +}$')
	plt.tight_layout()
	plt.savefig('Urms.png')

	# Plot Vrms+
	plt.figure(figsize=(6,8),dpi=100)

	plt.plot(bl_ystar[1:],bl_vvstar[1:],linewidth=3.0,label='Alya')
	plt.plot(dns[:,1],dns[:,4],'k--',linewidth=3.0,label='DNS')
	plt.xlim([0.1, 1000])
	plt.ylim([0.0, 1.5])
	plt.xlabel(r'$y^+$')
	plt.ylabel(r'$V^{\prime +}$')
	plt.tight_layout()
	plt.savefig('Vrms.png')

	# Plot Wrms+
	plt.figure(figsize=(6,8),dpi=100)

	plt.plot(bl_ystar[1:],bl_wwstar[1:],linewidth=3.0,label='Alya')
	plt.plot(dns[:,1],dns[:,5],'k--',linewidth=3.0,label='DNS')
	plt.xlim([0.1, 1000])
	plt.ylim([0.0, 2.0])
	plt.xlabel(r'$y^+$')
	plt.ylabel(r'$W^{\prime +}$')
	plt.tight_layout()
	plt.savefig('Wrms.png')

	print('Plots done!',flush=True)

pyQvarsi.cr_info()