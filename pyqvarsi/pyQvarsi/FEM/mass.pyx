#!/usr/bin/env cython
#
# pyQvarsi, FEM mass.
#
# Small FEM module to compute derivatives and possible other
# simple stuff from Alya output for postprocessing purposes.
#
# Mass matrix computation.
#
# Last rev: 27/12/2020
from __future__ import print_function, division

import numpy as np

cimport numpy as np
cimport cython

from ..   import vmath as math


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def mass_matrix_lumped(double[:,:] xyz,object[:] elemList):
	'''
	Compute lumped diagonal mass matrix for 2D or 3D
	elements.

	IN:
		> xyz(nnod,ndim): positions of the nodes
		> elemList(nel):  list of FEMlib.Element objects

	OUT:
		> vmass(nnod):   lumped mass matrix (open)
	'''
	cdef int		 MAXNODES = 1000
	cdef int         ielem, inod, idim, igp, nnod, ngp, \
					 nelem = len(elemList), nnodtot = xyz.shape[0], ndim = xyz.shape[1]
	cdef object      elem
	cdef int[:]      nodes
	cdef double[:,:] mle

	cdef np.ndarray[np.double_t,ndim=2] elxyz = np.ndarray((MAXNODES,ndim),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] vmass = np.zeros((nnodtot,),dtype=np.double)

	for ielem in range(nelem):
		elem    = elemList[ielem]
		nnod    = elem.nnod
		nodes   = elem.nodes
		ngp     = elem.ngauss
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(ndim):
				elxyz[inod,idim] = xyz[nodes[inod],idim]
		# Compute element mass matrix
		mle = elem.integrative(elxyz)
		# Assemble mass matrix
		for inod in range(nnod):
			for igp in range(ngp):
				vmass[nodes[inod]] += mle[inod,igp]
	return vmass


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def mass_matrix_consistent(xyz,elemList):
	'''
	Compute consistent mass matrix for 2D or 3D
	elements in CSR format.

	IN:
		> xyz(nnod,3):   positions of the nodes
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> cmass(nnod):   consistent mass matrix (open)
	'''
	cdef int		 MAXNODES = 1000
	cdef int         ielem, inod, idim, igp, nnod, ngp, \
					 nelem = len(elemList), nnodtot = xyz.shape[0], ndim = xyz.shape[1]
	cdef object      elem
	cdef int[:]      nodes
	cdef double[:,:] mle

	cdef np.ndarray[np.double_t,ndim=2] elxyz = np.ndarray((MAXNODES,ndim),dtype=np.double)
	cdef object cmass = math.dok_create(nnodtot,nnodtot,dtype=np.double)

	for ielem in range(nelem):
		elem    = elemList[ielem]
		nnod    = elem.nnod
		nodes   = elem.nodes
		ngp     = elem.ngauss
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(ndim):
				elxyz[inod,idim] = xyz[nodes[inod],idim]
		# Compute element mass matrix
		mle = elem.consistent(elxyz)
		# Assemble mass matrix
		for inod in range(nnod):
			for jnod in range(nnod):
				cmass[nodes[inod],nodes[jnod]] += mle[inod,jnod]
	return math.csr_tocsr(cmass)
