#!/usr/bin/env python
#
# pyQvarsi, utils.
#
# Connectivity operations that can be useful for
# other modules, such as computing the cell centers.
#
# Last rev: 25/07/2022
from __future__ import print_function, division

cimport numpy as np
cimport cython

import numpy as np

from ..cr import cr


@cr('meshing.cellCenters')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def cellCenters(double[:,:] xyz,int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> conec(nel,:):  connectivity matrix

	OUT:
		> xyz_cen(nel,3): cell center position
	'''
	cdef int ielem, icon, idim, c, cc, nel = conec.shape[0], ndim = xyz.shape[1], ncon = conec.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] xyz_cen = np.zeros((nel,ndim),dtype = np.double)

	for ielem in range(nel):
		# Set to zero
		for idim in range(ndim):
			xyz_cen[ielem,idim] = 0.
		cc = 0
		# Get the values of the field and the positions of the element
		for icon in range(ncon):
			c = conec[ielem,icon]
			if c < 0: break
			for idim in range(ndim):
				xyz_cen[ielem,idim] += xyz[c,idim]
			cc += 1
		# Average
		for idim in range(ndim):
			xyz_cen[ielem,idim] /= float(cc)
	return xyz_cen