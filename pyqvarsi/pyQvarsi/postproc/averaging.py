#!/usr/bin/env python
#
# pyQvarsi, postproc.
#
# Averaging routines.
#
# Last rev: 17/03/2021
from __future__ import print_function, division

import numpy as np

from ..field import Field
from ..cr    import cr


@cr('avg.midline')
def midlineAvg(array,do_copy=False):
	'''
	Midline average for arrays
	'''
	midline_sz = array.shape[0]//2
	reminder   = array.shape[0]%2
	array_new  = array.copy() if do_copy else array
	array_new[:midline_sz,]          = 0.5*(array_new[:midline_sz,] + np.flipud(array_new[midline_sz+reminder:,])) 
	array_new[midline_sz+reminder:,] = np.flipud(array_new[:midline_sz,])
	return array_new


@cr('avg.direction')
def directionAvg(field,direction='y'):
	'''
	Average in one direction taking the other two out of
	the scope.

	TODO: This will not work if the domain is split on the
	direction to average!!
	'''
	xyz = field.xyz.copy()
	# Direction to average
	if   direction == 'x': # Average on YZ plane
		xyz[:,1] = 0. # y
		xyz[:,2] = 0. # z	
	elif direction == 'y': # Average on XZ plane
		xyz[:,0] = 0.
		xyz[:,2] = 0.	
	elif direction == 'z': # Average on Z direction
		xyz[:,0] = 0.
		xyz[:,1] = 0.
	else:
		raiseError('Direction %s not recognized!' % direction)

	# Create the output field
	coord, idx  = np.unique(xyz,axis=0,return_inverse=True)
	ofield      = Field(xyz=coord)

	for varname in field.varnames:
		ofield[varname] = np.zeros((coord.shape[0],field[varname].shape[1]),dtype=field[varname].dtype) \
			if len(field[varname].shape) > 1 else np.zeros((coord.shape[0],),dtype=field[varname].dtype)

	counts = np.zeros((coord.shape[0],),dtype=np.int32)

	# Everyone who is not the master will operate and compute the average
	for ii in range(len(ofield)): # For each unique coordinate
		# Find the points that have the same y in field2
		idx = np.where(np.all(xyz == ofield.xyz[ii,:],axis=1))[0]
		# For each variable, sum the values
		for var in field.varnames:
			ofield[var][ii,] = np.sum(field[var][idx,],axis=0)
		# Update the count
		counts[ii] = len(idx)

	return ofield/counts, idx


@cr('avg.plane')
def planeAvg(field,direction='z'):
	'''
	Average in one plane taking one direction out of
	the scope.

	TODO: This will not work if the domain is split on the
	direction to average!!
	'''
	xyz = field.xyz.copy()
	# Direction to average
	if   direction == 'x':
		xyz[:,0] = 0.
	elif direction == 'y':
		xyz[:,1] = 0.
	elif direction == 'z':
		xyz[:,2] = 0.	
	else:
		raiseError('Direction %s not recognized!' % direction)

	# Create the output field
	coord  = np.unique(xyz,axis=0)
	ofield = Field(xyz=field.xyz)

	for varname in field.varnames:
		ofield[varname] = np.zeros(field[varname].shape,dtype=field[varname].dtype)

	counts = np.zeros((len(field),),dtype=np.int32)

	# Everyone who is not the master will operate and compute the average
	for ii in range(len(coord)): # For each unique coordinate
		# Find the points that have the same y in field2
		idx = np.where(np.all(xyz == coord[ii,:],axis=1))[0]
		# For each variable, sum the values
		for var in field.varnames:
			ofield[var][idx,] += np.sum(field[var][idx,],axis=0)
		# Update the count
		counts[idx] += len(idx)

	return ofield/counts