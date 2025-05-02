#!/usr/bin/env cython
#
# pyQvarsi, FEM integral.
#
# Small FEM module to compute integrals from Alya 
# output for postprocessing purposes.
#
# Last rev: 7/04/2021
from __future__ import print_function, division

import numpy as np

cimport numpy as np
cimport cython

from ..cr import cr
from ..utils.common import raiseError


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cr('integSurface')
def integralSurface(double[:,:] xyz,double[:] field,np.ndarray[np.npy_bool,ndim=1,cast=True] mask,object[:] elemList):
	'''
	Compute the surface integral of a 3D scalar 
	field given a list of elements.

	IN:
		> xyz(nnod,3):       positions of the nodes
		> field(nnod):       scalar field
		> mask(nnod):        masking field to eliminate nodes
							 from the integral
		> elemList(nel):     list of FEMlib.Element objects

	OUT:
		> integral:          value of the integral on the given area
	'''
	cdef int		 MAXNODES = 100
	cdef int         ielem, inod, idim, ig, nnod, ngauss, masked = 0, \
					 nelem = len(elemList), nnodtot = xyz.shape[0], ndim = xyz.shape[1]
	cdef object      elem
	cdef int[:]      nodes
	cdef double[:,:] elint
	cdef double      integral = 0.

	cdef np.ndarray[np.double_t,ndim=2] elxyz   = np.ndarray((MAXNODES,ndim),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] elfield = np.zeros((MAXNODES,),dtype=np.double)
	for ielem in range(nelem):
		# Get the values of the field, mask and positions of the element
		elem   = elemList[ielem]
		ngauss = elem.ngauss
		nnod   = elem.nnod
		nodes  = elem.nodes
		# Load the values into vectors
		masked = 1 # True by default (masked=compute)
		for inod in range(nnod):
			masked        = 1 if mask[nodes[inod]] else 0
			elfield[inod] = field[nodes[inod]]
			for idim in range(ndim):
				elxyz[inod,idim] = xyz[nodes[inod],idim]
		# Only compute the integral if the nodes of the element are not masked
		if masked:
			elint     = elem.integrative(elxyz)
			# Accumulate integral
			for ig in range(ngauss): # Loop gauss points
				for inod in range(nnod): # Loop element nodes
					integral += elint[inod,ig]*elfield[inod]
	
	return integral


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def integralVolume(double[:,:] xyz,double[:] field,np.ndarray[np.npy_bool,ndim=1,cast=True] mask,object[:] elemList):
	'''
	Compute the volume integral of a 3D scalar 
	field given a list of elements.

	IN:
		> xyz(nnod,3):       positions of the nodes
		> field(nnod):       scalar field
		> mask(nnod):        masking field to eliminate nodes
							 from the integral
		> elemList(nel):     list of FEMlib.Element objects

	OUT:
		> integral:          value of the integral on the given volume
	'''
	cdef int		 MAXNODES = 1000
	cdef int         ielem, inod, idim, ig, nnod, ngauss, masked = 0, \
					 nelem = len(elemList), nnodtot = xyz.shape[0], ndim = xyz.shape[1]
	cdef object      elem
	cdef int[:]      nodes
	cdef double[:,:] elint
	cdef double      integral = 0.

	cdef np.ndarray[np.double_t,ndim=2] elxyz   = np.ndarray((MAXNODES,ndim),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] elfield = np.zeros((MAXNODES,),dtype=np.double)
	for ielem in range(nelem):
		# Get the values of the field, mask and positions of the element
		elem   = elemList[ielem]
		ngauss = elem.ngauss
		nnod   = elem.nnod
		nodes  = elem.nodes
		# Load the values into vectors
		masked = 1 # True by default (masked=compute)
		for inod in range(nnod):
			masked        = 1 if mask[nodes[inod]] else 0
			elfield[inod] = field[nodes[inod]]
			for idim in range(ndim):
				elxyz[inod,idim] = xyz[nodes[inod],idim]
		# Only compute the integral if the nodes of the element are not masked
		if masked:
			elint     = elem.integrative(elxyz)
			# Accumulate integral
			for ig in range(ngauss): # Loop gauss points
				for inod in range(nnod): # Loop element nodes
					integral += elint[inod,ig]*elfield[inod]
	return integral