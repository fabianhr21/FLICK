#!/usr/bin/env python
#
# pyQvarsi, utils.
#
# Periodic utilities.
#
# Last rev: 27/07/2021
from __future__ import print_function, division

import os, numpy as np

from ..cr           import cr
from ..utils.common import truncate


@cr('periodic.get_coord')
def get_coordinates(geofile,basedir='./',precision=6):
	'''
	Obtain the coordinates from an Alya geo file
	'''
	# Open file for reading
	filename = os.path.join(basedir,geofile)
	file  = open(filename,'r')
	lines = file.read().split('\n')
	file.close()
	# Find the coordinate indices
	idx   = [il for il,l in enumerate(lines) if 'COORD' in l]
	coord = np.array([ [truncate(float(k),precision) for k in l.split()[1:]] for l in lines[idx[0]+1:idx[1]] ],np.double)
	# Return
	return coord


@cr('periodic.get_bbox')
def get_bounding_box(coord):
	'''
	Obtain the bounding box of a given coordinate set
	'''
	bbox    = np.zeros((6,),np.double)
	bbox[0] = np.min(coord[:,0]) # Min X 
	bbox[1] = np.max(coord[:,0]) # Max X
	bbox[2] = np.min(coord[:,1]) # Min Y 
	bbox[3] = np.max(coord[:,1]) # Max Y
	bbox[4] = np.min(coord[:,2]) # Min Z 
	bbox[5] = np.max(coord[:,2]) # Max Z
	return bbox


@cr('periodic.get_per_nodes')
def get_per_nodes(coord,value1=0.,value2=0.,perDim=0,gzero2=1e-10):
	'''
	Obtain periodic nodes (getPerNodes.cpp)
	'''
	# Loop coordinates and build maps
	no1, no2 = {}, {}
	for ii in range(coord.shape[0]):
		goal  = coord[ii,perDim]
		point = (0.,coord[ii,1],coord[ii,2])
		if perDim == 1: point = (coord[ii,0],0.,coord[ii,2])
		if perDim == 2: point = (coord[ii,0],coord[ii,1],0.)
		if abs(goal-value1) < gzero2: no1[point] = ii + 1 # Because of python...
		if abs(goal-value2) < gzero2: no2[point] = ii + 1 # Because of python...
	# Obtain periodic nodes
	perNodes = np.empty((0,2),np.int32)
	for key1 in no1.keys():
		if key1 in no2.keys():
				crit1 = abs(key1[perDim]-value1) < gzero2
				crit2 = abs(key1[perDim]-value2) < gzero2
				if not crit1 or not crit2:
					perNodes = np.append(perNodes,np.array([[no1[key1],no2[key1]]],np.int32),axis=0)
					no2.pop(key1)
	# Return
	return perNodes


@cr('periodic.unique')
def unique_periodic(perNodes):
	'''
	Clean the periodic nodes list and only obtain the
	ones that are unique combinations.
	'''
	# Build binding dictionary
	binding = {}
	for ip in range(perNodes.shape[0]):
		mast = perNodes[ip,0]
		targ = perNodes[ip,1]
		if targ in binding.keys():
			binding[mast] = targ
		else:
			binding[targ] = mast
	# Clean
	for k in sorted(binding.keys()):
		mast = binding[k]
		targ = k
		if mast in binding.keys():
			shihan = binding[mast]
			for k2 in binding.keys():
				if binding[k2] == mast:
					binding[k2] = shihan
	# Write cleaned list
	perNodes = np.array([[binding[k],k] for k in sorted(binding.keys())],np.int32)
	# Return
	return perNodes


@cr('periodic.write')
def write_periodic(perNodes,filename,basedir='./',slave_master=True):
	'''
	Write periodic nodes
	'''
	fname = os.path.join(basedir,filename)
	file  = open(fname,'w')
	if slave_master:
		# Write in slave/master format
		for ip in range(perNodes.shape[0]):
			file.write('%d %d\n'%(perNodes[ip,1],perNodes[ip,0]))
	else:
		# Write in master/slave format
		for ip in range(perNodes.shape[0]):
			file.write('%d %d\n'%(perNodes[ip,0],perNodes[ip,1]))		
	file.close()