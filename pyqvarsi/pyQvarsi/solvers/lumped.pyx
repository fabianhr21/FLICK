#!/usr/bin/env python
#
# pyQvarsi, lumped.
#
# Lumped solver.
#
# Last rev: 20/09/2022
from __future__ import print_function, division

cimport numpy as np
cimport cython

from ..cr import cr


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void solver_lumped_scalar(double[:] A, double[:] b):
	cdef int ii, n = b.shape[0]
	for ii in range(n):
		b[ii] /= A[ii]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void solver_lumped_array(double[:] A, double[:,:] b):
	cdef int ii, idim, n = b.shape[0], ndim = b.shape[1]
	for ii in range(n):
		for idim in range(ndim):
			b[ii,idim] /= A[ii]

@cr('solver.lumped')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
def solver_lumped(double[:] A, np.ndarray b,object commu=None):
	'''
	Solves the system
		b = A*x
	by
		x = b/A
	where A is the diagonal lumped matrix.

	Overwrites b.
	'''
	if len((<object> b).shape) == 1: # Scalar field
		solver_lumped_scalar(A,b)
		# We need to communicate with the boundaries to obtain
		# the full array
		if not commu is None: commu.communicate_scaf(b)
	else:
		solver_lumped_array(A,b)
		# We need to communicate with the boundaries to obtain
		# the full array
		if not commu is None: commu.communicate_arrf(b)
	return b