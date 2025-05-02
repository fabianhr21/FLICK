#!/usr/bin/env python
#
# Method of Manufactured Solutions (MMS)
# Example of the MESH class interpolation.
#
# Last revision: 18/04/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import pyQvarsi


# Function
fun = lambda xyz : np.sin(2.0*np.pi*xyz[:,0])*np.sin(2.0*np.pi*xyz[:,1])*np.sin(2.0*np.pi*xyz[:,2])

# Points of the mesh
p1 = np.array([0.,0.,0.])
p2 = np.array([1.,0.,0.])
p4 = np.array([0.,1.,0.])
p5 = np.array([0.,0.,1.])

n  = np.array([8,16,32,64,128])

# Interpolating points
xyz_i = np.array([
	[0.1,0.1,0.1],
	[0.4,0.1,0.2],
	[0.2,0.5,0.3],
	[0.4,0.3,0.8],
	[0.9,0.4,0.7],
	[0.7,0.9,0.3],
])

# Analytical solution
val_i = fun(xyz_i)


## MMS method
err = np.zeros((5,),dtype=np.double)
for ii,ni in enumerate(n):
	# Build the mesh
	mesh = pyQvarsi.Mesh.cube(p1,p2,p4,p5,ni,ni,ni)
	# Build a field containing the values of the function
	f    = pyQvarsi.Field(xyz=mesh.xyz,SCAF=fun(mesh.xyz))
	# Interpolate the values of the field to the points
	f_i  = mesh.interpolate(xyz_i,f,fact=1.)
	# Compute the maximum of the error
	err[ii] = np.max(np.abs(f_i['SCAF']-val_i))
	# Print
	print('MMS n = %d, err = %.2e'%(ni,err[ii]))


## Plots
# Fit a 1st grade polynomial
m = (np.log(err[-1])-np.log(err[0]))/(np.log(1/n[-1])-np.log(1/n[0]))
a = np.log(err[0]) - m*np.log(1/n[0])

# Plot MMS results
plt.figure(1,(8,6),dpi=100)
plt.plot(1/n,err)
plt.plot(1/n,np.exp(m*np.log(1/n)+a),'--k')
plt.yscale('log')
plt.xscale('log')
plt.title('MMS slope = %.4f'%m)

pyQvarsi.cr_info()
plt.show()
