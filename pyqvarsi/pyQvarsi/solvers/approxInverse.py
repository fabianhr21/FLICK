#!/usr/bin/env python
#
# pyQvarsi, aproximate_inverse.
#
# Aproximate inverse solver.
#
# Last rev: 28/09/2022
from __future__ import print_function, division

import numpy as np

from ..vmath import csr_spmv
from ..cr    import cr


def approxIverse_iter_scalar(A,Al,b,iters,commu):
	'''
	Approximate inverse solver iterations for a scalar field.

	A is a local matrix
	b = b/Al comming from the lumped solver and is a global array
	Al is a global diagonal matrix
	'''
	comm = commu.communicate_scaf if not commu == None else lambda x: x
	dot  = np.dot if isinstance(A,np.ndarray) else csr_spmv
	# Initialize
	r = b.copy() # r = Al*b -> global array
	# Iterate
	for ii in range(iters):
		r -= comm(dot(A,r))/Al # r = r - Al^(-1)*A*r 
		b += r 
	return b

def approxIverse_iter_array(A,Al,b,iters,commu):
	'''
	Approximate inverse solver iterations for a scalar field.

	A is a local matrix
	b = b/Al comming from the lumped solver and is a global array
	Al is a global diagonal matrix	'''
	comm = commu.communicate_scaf if not commu == None else lambda x: x
	dot  = np.dot if isinstance(A,np.ndarray) else csr_spmv
	for idim in range(b.shape[1]):
		# Initialize
		r = b[:,idim].copy() # r = Al*b -> global array
		# Iterate
		for ii in range(iters):
			r -= comm(dot(A,r))/Al # r = r - Al^(-1)*A*r 
			b[:,idim] += r 
	return b


@cr('solvers.approxInv')
def solver_approxInverse(A,Al,b,iters=25,commu=None):
	'''
	Approximate inverse solver for positive-definite
	matrices
		b = A*x
	solved as
		x = inv(A)*b
	and communicating in parallel after performing 
	the spmv.
	'''
	if len(b.shape) == 1: 
		# Scalar field
		approxIverse_iter_scalar(A,Al,b,iters,commu)
	else:
		# Vectorial field
		approxIverse_iter_array(A,Al,b,iters,commu)
	return b