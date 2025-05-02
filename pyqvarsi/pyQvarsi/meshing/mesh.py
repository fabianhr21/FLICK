#!/usr/bin/env python
#
# pyQvarsi, utils.
#
# Meshing utility routines.
#
# Last rev: 10/06/2021
from __future__ import print_function, division

import numpy as np

from ..utils.common import raiseError
from ..cr           import cr

conc_fun = lambda c,H,p : H/2.*(1 + ( np.tanh( p*(2*c/H-1) )/np.tanh(p) ))


@cr('meshing.planeMesh')
def planeMesh(p1,p2,p4,n1,n2,conc=0,f=0.2):
	'''
	3D mesh plane, useful for slices.

	4-------3
	|		|
	|		|
	1-------2

	Can concentrate in axis 1 or 2. 
	A value of 3 concentrates in both 1 and 2.
	A value of 0 keeps the mesh uniform.
	'''
	if conc > 3: raiseError('Invalid concentration parameter %d!'%conc)
	
	# Director unitary vectors
	v12 = p2 - p1; L12 = np.sqrt(np.sum(v12*v12)); v12 /= L12
	v14 = p4 - p1; L14 = np.sqrt(np.sum(v14*v14)); v14 /= L14

	# Create the mesh arrays (points)
	coord = np.zeros((n1*n2,3),dtype=np.double) # points
	lninv = np.zeros((n1*n2,) ,dtype=np.int32)  # global numbering

	# Generate the points
	idx = 0
	for ky in np.linspace(0,1,n2):
		dy = ky*v14*L14 if conc != 2 and conc != 3 else conc_fun(ky*v14*L14,L14,f)
		p = p1.copy() + dy
		for kx in np.linspace(0,1,n1):
			dx = kx*v12*L12 if conc != 1 and conc != 3 else conc_fun(kx*v12*L12,L12,f)
			coord[idx,:] = p + dx
			lninv[idx]   = idx
			idx         += 1

	# Create mesh arrays (elements)
	lnods = np.zeros(((n1-1)*(n2-1),4) ,dtype=np.int32) # points
	ltype = 12*np.ones(((n1-1)*(n2-1),),dtype=np.int32) # QUA04
	idx   = np.lexsort((coord[:,0],coord[:,1]))
	idx2  = idx.reshape((n1,n2),order='F')
	lnods[:,0] = idx2[:-1,:-1].ravel()
	lnods[:,1] = idx2[:-1,1:].ravel()
	lnods[:,2] = idx2[1:,1:].ravel()
	lnods[:,3] = idx2[1:,:-1].ravel()
	leinv = np.arange(0,(n1-1)*(n2-1),dtype=np.int32) # global numbering

	return coord,lnods,ltype,lninv,leinv


@cr('meshing.cubeMesh')
def cubeMesh(p1,p2,p4,p5,n1,n2,n3,conc=0,f=0.2):
	'''
	3D mesh cube, useful for volumes.

	  8-------7
	 /|      /|
	4-------3 |
	| 5-----|-6
	|/      |/
	1-------2

	Can concentrate in axis 1, 2 or 3. 
	A value of 4 concentrates in both 1 and 2.
	A value of 0 keeps the mesh uniform.
	'''
	if conc > 4: raiseError('Invalid concentration parameter %d!'%conc)

	# Director unitary vectors
	v12 = p2 - p1; L12 = np.sqrt(np.sum(v12*v12)); v12 /= L12
	v14 = p4 - p1; L14 = np.sqrt(np.sum(v14*v14)); v14 /= L14
	v15 = p5 - p1; L15 = np.sqrt(np.sum(v15*v15)); v15 /= L15

	# Create the mesh arrays
	coord = np.zeros((n1*n2*n3,3),dtype=np.double) # points
	lninv = np.zeros((n1*n2*n3,) ,dtype=np.int32)  # global numbering
	idx = 0
	for kz in np.linspace(0,1,n3):
		dz = kz*v15*L15 if conc != 3 and conc != 4 else conc_fun(kz*v15*L15,L15,f)
		p = p1.copy() + dz
		for ky in np.linspace(0,1,n2):
			dy = ky*v14*L14 if conc != 2 and conc != 4 else conc_fun(ky*v14*L14,L14,f)
			for kx in np.linspace(0,1,n1):
				dx = kx*v12*L12 if conc != 1 and conc != 3 else conc_fun(kx*v12*L12,L12,f)
				coord[idx,:] = p + dx + dy
				lninv[idx]   = idx
				idx         += 1

	# Create mesh arrays (elements)
	lnods = np.zeros(((n1-1)*(n2-1)*(n3-1),8) ,dtype=np.int32) # points
	ltype = 37*np.ones(((n1-1)*(n2-1)*(n3-1),),dtype=np.int32) # HEX08
	idx   = np.lexsort((coord[:,0],coord[:,1],coord[:,2]))
	idx2  = idx.reshape((n1,n2,n3),order='F')
	lnods[:,0] = idx2[:-1,:-1,:-1].ravel()
	lnods[:,1] = idx2[:-1,:-1,1:].ravel()
	lnods[:,2] = idx2[:-1,1:,1:].ravel()
	lnods[:,3] = idx2[:-1,1:,:-1].ravel()
	lnods[:,4] = idx2[1:,:-1,:-1].ravel()
	lnods[:,5] = idx2[1:,:-1,1:].ravel()
	lnods[:,6] = idx2[1:,1:,1:].ravel()
	lnods[:,7] = idx2[1:,1:,:-1].ravel()
	leinv = np.arange(0,(n1-1)*(n2-1)*(n3-1),dtype=np.int32) # global numbering
	
	return coord,lnods,ltype,lninv,leinv
