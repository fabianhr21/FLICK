#!/usr/bin/env python
#
# pyQvarsi, utils.
#
# Common utility routines.
#
# Last rev: 05/10/2020
from __future__ import print_function, division

import numpy as np

from ..mesh  import Mesh
from ..field import Field
from ..cr    import cr


@cr('meshing.reduceQUA04')
def reduce_conec_QUA04(xyz,f=None,ndiv=2,nj=199):
	'''
	Reduce and obtain the reduced connectivity for QUA04 elements
	'''
	# Total number of points
	N  = xyz.shape[0]
	ni = int(N/nj)
	# Reconstruct and reduce
	szyx  = np.lexsort((xyz[:,1], xyz[:,0]))
	ind   = szyx.reshape((ni,nj))[::ndiv,::ndiv]
	indf  = np.ravel(ind)
	indfi = np.arange(len(indf))
	szyxi = np.zeros_like(szyx)
	szyxi[indf] = indfi
	(ni,nj)  = ind.shape
	dims  = (ni,nj)
	# Create hex element
	lnods = np.zeros(((ni-1)*(nj-1),4),dtype='int32')
	lnods[:,0] = np.ravel(szyxi[ind[0:-1,0:-1]])
	lnods[:,1] = np.ravel(szyxi[ind[0:-1,1:  ]])
	lnods[:,2] = np.ravel(szyxi[ind[1:  ,1:  ]])
	lnods[:,3] = np.ravel(szyxi[ind[1:  ,0:-1]])
	# Reduce number of points and other useful arrays
	xyz_red = xyz[indf,:]
	lninv = np.arange(0,ni*nj,dtype=np.int32)           # global numbering
	leinv = np.arange(0,(ni-1)*(nj-1),dtype=np.int32)   # global numbering
	ltype = 12*np.ones(((ni-1)*(nj-1),),dtype=np.int32) # QUA04
	# Create reconstructed mesh
	mesh = Mesh(xyz_red,lnods,ltype,lninv,leinv)
	# Reduce variables
	field = Field(xyz=xyz_red)
	if not f is None:
		for var in f.varnames:
			array = f[var]
			if len(array.shape) > 1:
				# Vectorial or tensorial array
				field[var] = array[indf,:]
			else:
				# Scalar array
				field[var] = array[indf]
	return mesh, field, dims


@cr('meshing.readuceHEX08')
def reduce_conec_HEX08(xyz,f=None,ndiv=2,nk=399,nj=199):
	'''
	Reduce and obtain the reduced connectivity for HEX08 elements
	'''
	# Total number of points
	N  = xyz.shape[0]
	ni = int(N/nk/nj)
	# Reconstruct and reduce
	szyx  = np.lexsort((xyz[:,2], xyz[:,1], xyz[:,0]))
	ind   = szyx.reshape((ni,nj,nk))[::ndiv,::ndiv,::ndiv]
	indf  = np.ravel(ind)
	indfi = np.arange(len(indf))
	szyxi = np.zeros_like(szyx)
	szyxi[indf] = indfi
	(ni,nj,nk)  = ind.shape
	dims  = (ni,nj,nk)
	# Create hex element
	lnods = np.zeros(((ni-1)*(nj-1)*(nk-1),8),dtype='int32')
	lnods[:,0] = np.ravel(szyxi[ind[0:-1,0:-1,0:-1]])
	lnods[:,1] = np.ravel(szyxi[ind[0:-1,1:  ,0:-1]])
	lnods[:,2] = np.ravel(szyxi[ind[1:  ,1:  ,0:-1]])
	lnods[:,3] = np.ravel(szyxi[ind[1:  ,0:-1,0:-1]])
	lnods[:,4] = np.ravel(szyxi[ind[0:-1,0:-1,1:  ]])
	lnods[:,5] = np.ravel(szyxi[ind[0:-1,1:  ,1:  ]])
	lnods[:,6] = np.ravel(szyxi[ind[1:  ,1:  ,1:  ]])
	lnods[:,7] = np.ravel(szyxi[ind[1:  ,0:-1,1:  ]])
	# Reduce number of points and other useful arrays
	xyz_red = xyz[indf,:]
	lninv = np.arange(0,ni*nj*nk,dtype=np.int32)               # global numbering
	leinv = np.arange(0,(ni-1)*(nj-1)*(nk-1),dtype=np.int32)   # global numbering
	ltype = 37*np.ones(((ni-1)*(nj-1)*(nk-1),),dtype=np.int32) # HEX08
	# Create reconstructed mesh
	mesh = Mesh(xyz_red,lnods,ltype,lninv,leinv)
	# Reduce variables
	field = Field(xyz=xyz_red)
	if not f is None:
		for var in f.varnames:
			array = f[var]
			if len(array.shape) > 1:
				# Vectorial or tensorial array
				field[var] = array[indf,:]
			else:
				# Scalar array
				field[var] = array[indf]
	return mesh, field, dims