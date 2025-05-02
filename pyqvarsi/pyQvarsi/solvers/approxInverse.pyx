#!/usr/bin/env python
#
# pyQvarsi, aproximate_inverse.
#
# Aproximate inverse solver.
#
# Last rev: 28/09/2022
from __future__ import print_function, division

cimport numpy as np
cimport cython

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

import numpy as np

from ..utils import raiseWarning
from ..cr    import cr


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void dot_scalar(double* c, double[:,:] A, double* b, int n):
	'''
	Full matrix vector product between A and b. Overwrites b.
	'''
	cdef int ii, jj
	for ii in range(n):
		c[ii] = 0.
		for jj in range(n):
			c[ii] += A[ii,jj]*b[jj]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void spmv_scalar(double* c, int nzdom,int[:] rdom,int[:] cdom,double[:] A,double* b,int n):
	'''
	Sparse matrix vector product between A and b.
	'''
	cdef int ip, jp, iz, rowb, rowe

	for ip in range(n):
		c[ip] = 0.
		# Get CSR section for row ipoin
		rowb = rdom[ip]
		rowe = rdom[ip+1]
		# Loop inside CSR section
		for iz in range(rowb,rowe):
			jp = cdom[iz]        # Col. index
			c[ip] += A[iz]*b[jp] # Dot product


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void approxInverse_scalar_full(double[:,:] A,double[:] Al,double[:] b,int iters):
	'''
	Approximate inverse solver iterations for a scalar field.

	A is a local matrix
	b = b/Al comming from the lumped solver and is a global array
	Al is a global diagonal matrix
	'''
	cdef int ii, it, n = b.shape[0]
	cdef double *c
	cdef double *r

	c = <double*> malloc(n*sizeof(double))
	r = <double*> malloc(n*sizeof(double))

	# Initialize
	memcpy(r,&b[0],n*sizeof(double)) # r = Al*b -> global array
	# Iterate
	for it in range(iters):
		dot_scalar(c,A,r,n) # c = A*r -> local array
		# Update solution
		for ii in range(n):
			r[ii] -= c[ii]/Al[ii] # r = r - Al^(-1)*A*r 
			b[ii] += r[ii]
	# Free memory
	free(c)
	free(r)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void approxInverse_scalar_full_comm(double[:,:] A,double[:] Al,double[:] b,int iters,object commu):
	'''
	Approximate inverse solver iterations for a scalar field.

	A is a local matrix
	b = b/Al comming from the lumped solver and is a global array
	Al is a global diagonal matrix
	'''
	cdef int ii, it, n = b.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] c = np.zeros((n,),np.double)
	cdef double *r

	r = <double*> malloc(n*sizeof(double))

	# Initialize
	memcpy(r,&b[0],n*sizeof(double)) # r = Al*b -> global array
	# Iterate
	for it in range(iters):
		dot_scalar(&c[0],A,r,n)   # c = A*r -> local array
		commu.communicate_scaf(c) # c (local) -> c (global)
		# Update solution
		for ii in range(n):
			r[ii] -= c[ii]/Al[ii] # r = r - Al^(-1)*A*r 
			b[ii] += r[ii]
	# Free memory
	free(r)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void approxInverse_vector_full(double[:,:] A,double[:] Al,double[:,:] b,int iters):
	'''
	Approximate inverse solver iterations for a scalar field.

	A is a local matrix
	b = b/Al comming from the lumped solver and is a global array
	Al is a global diagonal matrix
	'''
	cdef int ii, it, idim, n = b.shape[0], ndim = b.shape[1]
	cdef double *c
	cdef double *r

	c = <double*> malloc(n*sizeof(double))
	r = <double*> malloc(n*sizeof(double))

	for idim in range(ndim):
		# Initialize
		for ii in range(n):
			r[ii] = b[ii,idim] # r = Al*b -> global array
		# Iterate
		for it in range(iters):
			dot_scalar(c,A,r,n) # c = A*r -> local array
			# Update solution
			for ii in range(n):
				r[ii] -= c[ii]/Al[ii] # r = r - Al^(-1)*A*r 
				b[ii,idim] += r[ii]
	# Free memory
	free(c)
	free(r)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void approxInverse_vector_full_comm(double[:,:] A,double[:] Al,double[:,:] b,int iters,object commu):
	'''
	Approximate inverse solver iterations for a scalar field.

	A is a local matrix
	b = b/Al comming from the lumped solver and is a global array
	Al is a global diagonal matrix
	'''
	cdef int ii, it, idim, n = b.shape[0], ndim = b.shape[1]
	cdef np.ndarray[np.double_t,ndim=1] c = np.zeros((n,),np.double)
	cdef double *r

	r = <double*> malloc(n*sizeof(double))

	for idim in range(ndim):
		# Initialize
		for ii in range(n):
			r[ii] = b[ii,idim] # r = Al*b -> global array
		# Iterate
		for it in range(iters):
			dot_scalar(&c[0],A,r,n)   # c = A*r -> local array
			commu.communicate_scaf(c) # c (local) -> c (global)
			# Update solution
			for ii in range(n):
				r[ii] -= c[ii]/Al[ii] # r = r - Al^(-1)*A*r 
				b[ii,idim] += r[ii]
	# Free memory
	free(r)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void approxInverse_scalar_sp(int nzdom,int[:] rdom,int[:] cdom,double[:] A,double[:] Al,double[:] b,int iters):
	'''
	Approximate inverse solver iterations for a scalar field.

	A is a local matrix
	b = b/Al comming from the lumped solver and is a global array
	Al is a global diagonal matrix
	'''
	cdef int ii, it, n = b.shape[0]
	cdef double *c
	cdef double *r

	c = <double*> malloc(n*sizeof(double))
	r = <double*> malloc(n*sizeof(double))

	# Initialize
	memcpy(r,&b[0],n*sizeof(double)) # r = Al*b -> global array
	# Iterate
	for it in range(iters):
		spmv_scalar(c,nzdom,rdom,cdom,A,r,n) # c = A*r -> local array
		# Update solution
		for ii in range(n):
			r[ii] -= c[ii]/Al[ii] # r = r - Al^(-1)*A*r 
			b[ii] += r[ii]
	# Free memory
	free(c)
	free(r)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void approxInverse_scalar_sp_comm(int nzdom,int[:] rdom,int[:] cdom,double[:] A,double[:] Al,double[:] b,int iters,object commu):
	'''
	Approximate inverse solver iterations for a scalar field.

	A is a local matrix
	b = b/Al comming from the lumped solver and is a global array
	Al is a global diagonal matrix
	'''
	cdef int ii, it, n = b.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] c = np.zeros((n,),np.double)
	cdef double *r

	r = <double*> malloc(n*sizeof(double))

	# Initialize
	memcpy(r,&b[0],n*sizeof(double)) # r = Al*b -> global array
	# Iterate
	for it in range(iters):
		spmv_scalar(&c[0],nzdom,rdom,cdom,A,r,n) # c = A*r -> local array
		commu.communicate_scaf(c) # c (local) -> c (global)
		# Update solution
		for ii in range(n):
			r[ii] -= c[ii]/Al[ii] # r = r - Al^(-1)*A*r 
			b[ii] += r[ii]
	# Free memory
	free(r)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void approxInverse_vector_sp(int nzdom,int[:] rdom,int[:] cdom,double[:] A,double[:] Al,double[:,:] b,int iters):
	'''
	Approximate inverse solver iterations for a scalar field.

	A is a local matrix
	b = b/Al comming from the lumped solver and is a global array
	Al is a global diagonal matrix
	'''
	cdef int ii, it, idim, n = b.shape[0], ndim = b.shape[1]
	cdef double *c
	cdef double *r

	c = <double*> malloc(n*sizeof(double))
	r = <double*> malloc(n*sizeof(double))

	for idim in range(ndim):
		# Initialize
		for ii in range(n):
			r[ii] = b[ii,idim] # r = Al*b -> global array
		# Iterate
		for it in range(iters):
			spmv_scalar(c,nzdom,rdom,cdom,A,r,n) # c = A*r -> local array
			# Update solution
			for ii in range(n):
				r[ii] -= c[ii]/Al[ii] # r = r - Al^(-1)*A*r 
				b[ii,idim] += r[ii]
	# Free memory
	free(c)
	free(r)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void approxInverse_vector_sp_comm(int nzdom,int[:] rdom,int[:] cdom,double[:] A,double[:] Al,double[:,:] b,int iters,object commu):
	'''
	Approximate inverse solver iterations for a scalar field.

	A is a local matrix
	b = b/Al comming from the lumped solver and is a global array
	Al is a global diagonal matrix
	'''
	cdef int ii, it, idim, n = b.shape[0], ndim = b.shape[1]
	cdef np.ndarray[np.double_t,ndim=1] c = np.zeros((n,),np.double)
	cdef double *r

	r = <double*> malloc(n*sizeof(double))

	for idim in range(ndim):
		# Initialize
		for ii in range(n):
			r[ii] = b[ii,idim] # r = Al*b -> global array
		# Iterate
		for it in range(iters):
			spmv_scalar(&c[0],nzdom,rdom,cdom,A,r,n) # c = A*r -> local array
			commu.communicate_scaf(c) # c (local) -> c (global)
			# Update solution
			for ii in range(n):
				r[ii] -= c[ii]/Al[ii] # r = r - Al^(-1)*A*r 
				b[ii,idim] += r[ii]
	# Free memory
	free(r)


@cr('solvers.approxInv')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
def solver_approxInverse(object A,double[:] Al,np.ndarray b,int iters=25,object commu=None):
	'''
	Approximate inverse solver for positive-definite
	matrices
		b = A*x
	solved as
		x = inv(A)*b
	and communicating in parallel after performing 
	the spmv.
	'''
	cdef int nzdom
	cdef int[:] rdom, cdom
	cdef double[:] Ad
	# Select which algorithm to use
	if isinstance(A,np.ndarray):
		# Full matrix algorithms
		if len((<object> b).shape) == 1: 
			if commu == None:
				approxInverse_scalar_full(A,Al,b,iters)
			else:
				approxInverse_scalar_full_comm(A,Al,b,iters,commu)
		else:
			# Vectorial field
			if commu == None:
				approxInverse_vector_full(A,Al,b,iters)
			else:
				approxInverse_vector_full_comm(A,Al,b,iters,commu)
	else:
		nzdom = A.nnz
		rdom  = A.indptr
		cdom  = A.indices
		Ad    = A.data
		# Sparse matrix algorithms
		if len((<object> b).shape) == 1: 
			# Scalar field
			if commu == None:
				approxInverse_scalar_sp(nzdom,rdom,cdom,Ad,Al,b,iters)
			else:
				approxInverse_scalar_sp_comm(nzdom,rdom,cdom,Ad,Al,b,iters,commu)
		else:
			# Vectorial field
			if commu == None:
				approxInverse_vector_sp(nzdom,rdom,cdom,Ad,Al,b,iters)
			else:
				approxInverse_vector_sp_comm(nzdom,rdom,cdom,Ad,Al,b,iters,commu)
	return b