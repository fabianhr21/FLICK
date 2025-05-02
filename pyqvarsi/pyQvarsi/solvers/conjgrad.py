#!/usr/bin/env python
#
# pyQvarsi, conjugate_gradient.
#
# Conjugate gradient solver.
#
# Last rev: 23/09/2022
from __future__ import print_function, division

import numpy as np

from ..vmath import csr_spmv
from ..utils import raiseWarning
from ..cr    import cr


def conjgrad_scalar(A,b,iters,refresh,tol,commu):
	'''
	Conjugate gradient solver for scalar arrays
	'''
	spmv = np.matmul if isinstance(A,np.ndarray) else csr_spmv
	comm = commu.communicate_scaf if not commu == None else lambda x: x
	redu = commu.allreduce        if not commu == None else lambda x,op: x
	# Initialize solver
	x = b.copy()/comm(A.diagonal())
	# Initialize residual
	r = b - comm(spmv(A,x))
	d = r.copy()
	# Error computation
	err  = redu(np.dot(r,r),'nansum')
	stop = tol*redu(np.dot(b,b),'nansum')
	# Start iterations
	for it in range(1,iters+1):
		Ad = comm(spmv(A,d))
		Q1 = err 
		Q2 = redu(np.dot(d,Ad),'nansum')
		# Compute alpha
		alpha = Q1/Q2
		# Update solution
		x += alpha*d
		# Update residual
		if it%refresh == 0:
			r = b - comm(spmv(A,x))
		else:
			r -= alpha*Ad
		# Error
		err = redu(np.dot(r,r),'nansum')
		if err < stop: break
		# Update p
		beta = err/Q1
		d    = r + beta*d
	# Update solution
	b = x.copy()
	# Warning message
	if it == iters: raiseWarning('solver conjgrad maximum iterations reached (error=%.2e)!'%err)
	return b

def conjgrad_vector(A,b,iters,refresh,tol,commu):
	'''
	Conjugate gradient solver for vectorial arrays
	'''
	spmv = np.matmul if isinstance(A,np.ndarray) else csr_spmv
	comm = commu.communicate_scaf if not commu == None else lambda x: x
	redu = commu.allreduce        if not commu == None else lambda x,op: x
	for idim in range(b.shape[1]):
		# Initialize solver
		x = b[:,idim].copy()/comm(A.diagonal())
		# Initialize residual
		r = b[:,idim] - comm(spmv(A,x))
		d = r.copy()
		# Error computation
		err  = redu(np.dot(r,r),'nansum')
		stop = tol*redu(np.dot(b[:,idim],b[:,idim]),'nansum')
		# Start iterations
		for it in range(1,iters+1):
			Ad = comm(spmv(A,d))
			Q1 = err 
			Q2 = redu(np.dot(d,Ad),'nansum')
			# Compute alpha
			alpha = Q1/Q2
			# Update solution
			x += alpha*d
			# Update residual
			if it%refresh == 0:
				r = b[:,idim] - comm(spmv(A,x))
			else:
				r -= alpha*Ad
			# Error
			err  = redu(np.dot(r,r),'nansum')
			if err < stop: break
			# Update p
			beta = err/Q1
			d    = r + beta*d
		# Update solution
		b[:,idim] = x.copy()
	# Warning message
	if it == iters: raiseWarning('solver conjgrad maximum iterations reached (error=%.2e)!'%err)
	return b

@cr('solver.conjgrad')
def solver_conjgrad(A,b,iters=500,refresh=20,tol=1e-8,commu=None,b_global=False):
	'''
	Solve a linar system such as 
		b = A*x
	using the conjugate gradient method 
	and communicating in parallel after performing 
	the spmv.
	'''
	if len(b.shape) > 1:
		if not b_global and commu is not None: commu.communicate_arrf(b)
		b = conjgrad_vector(A,b,iters,refresh,tol,commu)
	else:
		if not b_global and commu is not None: commu.communicate_scaf(b)
		b = conjgrad_scalar(A,b,iters,refresh,tol,commu)
	return b