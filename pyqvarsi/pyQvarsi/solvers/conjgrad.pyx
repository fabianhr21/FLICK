#!/usr/bin/env python
#
# pyQvarsi, conjugate_gradient.
#
# Conjugate gradient solver.
#
# Last rev: 23/09/2022
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
cdef double dot(double* a, double* b, int n):
	'''
	Full matrix vector product between A and b. Overwrites b.
	'''
	cdef int ii
	cdef double aux = 0.

	for ii in range(n):
		aux += a[ii]*b[ii]
	return aux

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void mv_scalar(double* c, double[:,:] A, double* b, int n):
	'''
	Full matrix vector product between A and b.
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
		rowb = rdom[ip]#+1
		rowe = rdom[ip+1]
		# Loop inside CSR section
		for iz in range(rowb,rowe):
			jp = cdom[iz]        # Col. index
			c[ip] += A[iz]*b[jp] # Dot product


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void conjgrad_scalar_full(double[:,:] A,double[:] b,int iters,int refresh,double tol):
	'''
	Conjugate gradient solver for scalar arrays
	'''
	# Define variables
	cdef int ii, it, n = b.shape[0]
	cdef double err, stop, Q1, Q2, alpha, beta
	cdef double *x
	cdef double *r
	cdef double *d
	cdef double *u

	# Allocate memory
	x = <double*>malloc(n*sizeof(double))
	r = <double*>malloc(n*sizeof(double))
	d = <double*>malloc(n*sizeof(double))
	u = <double*>malloc(n*sizeof(double))

	# Initialize solver using the diagonal of A as a preconditioner
	for ii in range(n):
		x[ii] = b[ii]/A[ii,ii]

	# Initialize residual and direction
	mv_scalar(u,A,x,n)  # u = A*x
	for ii in range(n):
		r[ii] = b[ii] - u[ii] # r = b - A*x
		d[ii] = r[ii]         # d = r

	# Error computation
	err  = dot(r,r,n)
	stop = tol*dot(&b[0],&b[0],n)

	# Start iterations
	for it in range(iters):
		mv_scalar(u,A,d,n)
		Q1 = err
		Q2 = dot(d,&u[0],n)
		# Compute alpha
		alpha = Q1/Q2
		if it%refresh == 0:
			# Update solution
			for ii in range(n):
				x[ii] += alpha*d[ii]
			# Update residual
			mv_scalar(u,A,x,n)
			for ii in range(n):
				r[ii] = b[ii] - u[ii]				
		else:
			# Update solution
			for ii in range(n):
				x[ii] += alpha*d[ii]
				r[ii] -= alpha*u[ii]
		# Error computation
		err = dot(r,r,n)
		if err < stop: break
		# Update direction
		beta = err/Q1
		for ii in range(n):
			d[ii] = r[ii] + beta*d[ii]

	# Update solution
	memcpy(&b[0],x,n*sizeof(double))

	# Free memory
	free(x)
	free(r)
	free(d)
	free(u)

	# Warning message
	if it == iters: raiseWarning('solver conjgrad maximum iterations reached (error=%.2e)!'%err)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
cdef void conjgrad_scalar_full_comm(double[:,:] A,double[:] b,int iters,int refresh,double tol,object commu):
	'''
	Conjugate gradient solver for scalar arrays
	'''
	# Define variables
	cdef int ii, it, n = b.shape[0]
	cdef double err, stop, Q1, Q2, alpha, beta

	cdef np.ndarray[np.double_t,ndim=1] u = np.zeros((n,),np.double)
	cdef double *x
	cdef double *r
	cdef double *d

	# Allocate memory
	x = <double*>malloc(n*sizeof(double))
	r = <double*>malloc(n*sizeof(double))
	d = <double*>malloc(n*sizeof(double))

	# Initialize solver using the diagonal of A as a preconditioner
	for ii in range(n):
		u[ii] = A[ii,ii]
	commu.communicate_scaf(u)
	for ii in range(n):
		x[ii] = b[ii]/u[ii]

	# Initialize residual and direction
	mv_scalar(&u[0],A,x,n) # u containx A*x
	commu.communicate_scaf(u)
	for ii in range(n):
		r[ii] = b[ii] - u[ii] # r = b - A*x
		d[ii] = r[ii]         # d = r

	# Error computation
	err  = dot(r,r,n)
	err  = commu.allreduce(err,op='nansum') 
	stop = tol*dot(&b[0],&b[0],n)
	stop = commu.allreduce(stop,op='nansum') 

	# Start iterations
	for it in range(iters):
		mv_scalar(&u[0],A,d,n)
		commu.communicate_scaf(u)
		Q1 = err
		Q2 = dot(d,&u[0],n)
		Q2 = commu.allreduce(Q2,op='nansum') 
		# Compute alpha
		alpha = Q1/Q2
		if it%refresh == 0:
			# Update solution
			for ii in range(n):
				x[ii] += alpha*d[ii]
			# Update residual
			mv_scalar(&u[0],A,x,n)
			commu.communicate_scaf(u)
			for ii in range(n):
				r[ii] = b[ii] - u[ii]
		else:
			# Update solution
			for ii in range(n):
				x[ii] += alpha*d[ii]		
				r[ii] -= alpha*u[ii]
		# Error computation
		err = dot(r,r,n)
		err = commu.allreduce(err,op='nansum') 
		if err < stop: break
		# Update direction
		beta = err/Q1
		for ii in range(n):
			d[ii] = r[ii] + beta*d[ii]

	# Update solution
	memcpy(&b[0],x,n*sizeof(double))

	# Free memory
	free(x)
	free(r)
	free(d)

	# Warning message
	if it == iters: raiseWarning('solver conjgrad maximum iterations reached (error=%.2e)!'%err)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void conjgrad_vector_full(double[:,:] A,double[:,:] b,int iters,int refresh,double tol):
	'''
	Conjugate gradient solver for scalar arrays
	'''
	# Define variables
	cdef int ii, it, idim, n = b.shape[0], ndim = b.shape[1]
	cdef double err, stop, Q1, Q2, alpha, beta
	cdef double *x
	cdef double *r
	cdef double *d
	cdef double *u

	# Allocate memory
	x = <double*>malloc(n*sizeof(double))
	r = <double*>malloc(n*sizeof(double))
	d = <double*>malloc(n*sizeof(double))
	u = <double*>malloc(n*sizeof(double))

	for idim in range(ndim):
		# Initialize solver using the diagonal of A as a preconditioner
		for ii in range(n):
			x[ii] = b[ii,idim]/A[ii,ii]

		# Initialize residual and direction
		mv_scalar(u,A,x,n) # u containx A*x
		for ii in range(n):
			r[ii] = b[ii,idim] - u[ii] # r = b - A*x
			d[ii] = r[ii]              # d = r
			u[ii] = b[ii,idim]

		# Error computation
		err  = dot(r,r,n)
		stop = tol*dot(u,u,n)

		# Start iterations
		for it in range(iters):
			mv_scalar(u,A,d,n)
			Q1 = err
			Q2 = dot(d,&u[0],n)
			# Compute alpha
			alpha = Q1/Q2
			if it%refresh == 0:
				# Update solution
				for ii in range(n):
					x[ii] += alpha*d[ii]
				# Update residual
				mv_scalar(u,A,x,n)
				for ii in range(n):
					r[ii] = b[ii,idim] - u[ii]				
			else:
				# Update solution
				for ii in range(n):
					x[ii] += alpha*d[ii]
					r[ii] -= alpha*u[ii]
			# Error computation
			err = dot(r,r,n)
			if err < stop: break
			# Update direction
			beta = err/Q1
			for ii in range(n):
				d[ii] = r[ii] + beta*d[ii]

		# Update solution
		for ii in range(n):
			b[ii,idim] = x[ii]

	# Free memory
	free(x)
	free(r)
	free(d)
	free(u)

	# Warning message
	if it == iters: raiseWarning('solver conjgrad maximum iterations reached (error=%.2e)!'%err)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
cdef void conjgrad_vector_full_comm(double[:,:] A,double[:,:] b,int iters,int refresh,double tol,object commu):
	'''
	Conjugate gradient solver for scalar arrays
	'''
	# Define variables
	cdef int ii, it, idim, n = b.shape[0], ndim = b.shape[1]
	cdef double err, stop, Q1, Q2, alpha, beta

	cdef np.ndarray[np.double_t,ndim=1] u = np.zeros((n,),np.double)
	cdef double *x
	cdef double *r
	cdef double *d

	# Allocate memory
	x = <double*>malloc(n*sizeof(double))
	r = <double*>malloc(n*sizeof(double))
	d = <double*>malloc(n*sizeof(double))

	for idim in range(ndim):
		# Initialize solver using the diagonal of A as a preconditioner
		for ii in range(n):
			u[ii] = A[ii,ii]
		commu.communicate_scaf(u)
		for ii in range(n):
			x[ii] = b[ii,idim]/u[ii]

		# Initialize residual and direction
		mv_scalar(&u[0],A,x,n) # u containx A*x
		commu.communicate_scaf(u)
		for ii in range(n):
			r[ii] = b[ii,idim] - u[ii] # r = b - A*x
			d[ii] = r[ii]              # d = r
			u[ii] = b[ii,idim]

		# Error computation
		err  = dot(r,r,n)
		err  = commu.allreduce(err,op='nansum')
		stop = tol*dot(&u[0],&u[0],n)
		stop = commu.allreduce(stop,op='nansum') 

		# Start iterations
		for it in range(iters):
			mv_scalar(&u[0],A,d,n)
			commu.communicate_scaf(u)
			Q1 = err
			Q2 = dot(d,&u[0],n)
			Q2 = commu.allreduce(Q2,op='nansum') 
			# Compute alpha
			alpha = Q1/Q2
			if it%refresh == 0:
				# Update solution
				for ii in range(n):
					x[ii] += alpha*d[ii]
				# Update residual
				mv_scalar(&u[0],A,x,n)
				commu.communicate_scaf(u)
				for ii in range(n):
					r[ii] = b[ii,idim] - u[ii]
			else:
				# Update solution
				for ii in range(n):
					x[ii] += alpha*d[ii]		
					r[ii] -= alpha*u[ii]
			# Error computation
			err = dot(r,r,n)
			err = commu.allreduce(err,op='nansum') 
			if err < stop: break
			# Update direction
			beta = err/Q1
			for ii in range(n):
				d[ii] = r[ii] + beta*d[ii]

		# Update solution
		for ii in range(n):
			b[ii,idim] = x[ii]

	# Free memory
	free(x)
	free(r)
	free(d)

	# Warning message
	if it == iters: raiseWarning('solver conjgrad maximum iterations reached (error=%.2e)!'%err)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void conjgrad_scalar_sp(int nzdom, int[:] rdom, int[:] cdom, double[:] Ad,double[:] b,int iters,int refresh,double tol):
	'''
	Conjugate gradient solver for scalar arrays
	'''
	# Define variables
	cdef int ii, it, iz, rowb, rowe, n = b.shape[0]
	cdef double err, stop, Q1, Q2, alpha, beta
	cdef double *x
	cdef double *r
	cdef double *d
	cdef double *u

	# Allocate memory
	x = <double*>malloc(n*sizeof(double))
	r = <double*>malloc(n*sizeof(double))
	d = <double*>malloc(n*sizeof(double))
	u = <double*>malloc(n*sizeof(double))

	# Initialize solver using the diagonal of A as a preconditioner
	for ii in range(n):
		rowb = rdom[ii]#+1
		rowe = rdom[ii+1]
		for iz in range(rowb,rowe):
			if cdom[iz] == ii: u[ii] = Ad[iz]
		x[ii] = b[ii]/u[ii]

	# Initialize residual and direction
	spmv_scalar(&u[0],nzdom,rdom,cdom,Ad,x,n) # u = A*x
	for ii in range(n):
		r[ii] = b[ii] - u[ii] # r = b - A*x
		d[ii] = r[ii]         # d = r

	# Error computation
	err  = dot(r,r,n)
	stop = tol*dot(&b[0],&b[0],n)

	# Start iterations
	for it in range(iters):
		spmv_scalar(&u[0],nzdom,rdom,cdom,Ad,d,n) # u = A*d
		Q1 = err
		Q2 = dot(d,&u[0],n)
		# Compute alpha
		alpha = Q1/Q2
		if it%refresh == 0:
			# Update solution
			for ii in range(n):
				x[ii] += alpha*d[ii]
			# Update residual
			spmv_scalar(&u[0],nzdom,rdom,cdom,Ad,x,n) # u = A*x
			for ii in range(n):
				r[ii] = b[ii] - u[ii]				
		else:
			# Update solution
			for ii in range(n):
				x[ii] += alpha*d[ii]
				r[ii] -= alpha*u[ii]
		# Error computation
		err = dot(r,r,n)
		if err < stop: break
		# Update direction
		beta = err/Q1
		for ii in range(n):
			d[ii] = r[ii] + beta*d[ii]

	# Update solution
	memcpy(&b[0],x,n*sizeof(double))

	# Free memory
	free(x)
	free(r)
	free(d)
	free(u)

	# Warning message
	if it == iters: raiseWarning('solver conjgrad maximum iterations reached (error=%.2e)!'%err)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
cdef void conjgrad_scalar_sp_comm(int nzdom, int[:] rdom, int[:] cdom, double[:] Ad,double[:] b,int iters,int refresh,double tol,object commu):
	'''
	Conjugate gradient solver for scalar arrays
	'''
	# Define variables
	cdef int ii, it, iz, rowb, rowe, n = b.shape[0]
	cdef double err, stop, Q1, Q2, alpha, beta

	cdef np.ndarray[np.double_t,ndim=1] u = np.zeros((n,),np.double)
	cdef double *x
	cdef double *r
	cdef double *d

	# Allocate memory
	x = <double*>malloc(n*sizeof(double))
	r = <double*>malloc(n*sizeof(double))
	d = <double*>malloc(n*sizeof(double))

	# Initialize solver using the diagonal of A as a preconditioner
	for ii in range(n):
		rowb = rdom[ii]#+1
		rowe = rdom[ii+1]
		for iz in range(rowb,rowe):
			if cdom[iz] == ii: u[ii] = Ad[iz]
	commu.communicate_scaf(u)
	for ii in range(n):
		x[ii] = b[ii]/u[ii]

	# Initialize residual and direction
	spmv_scalar(&u[0],nzdom,rdom,cdom,Ad,x,n) # u = A*x
	commu.communicate_scaf(u)
	for ii in range(n):
		r[ii] = b[ii] - u[ii] # r = b - A*x
		d[ii] = r[ii]         # d = r

	# Error computation
	err  = dot(r,r,n)
	err  = commu.allreduce(err,op='nansum') 
	stop = tol*dot(&b[0],&b[0],n)
	stop = commu.allreduce(stop,op='nansum') 

	# Start iterations
	for it in range(iters):
		spmv_scalar(&u[0],nzdom,rdom,cdom,Ad,d,n) # u = A*x
		commu.communicate_scaf(u)
		Q1 = err
		Q2 = dot(d,&u[0],n)
		Q2 = commu.allreduce(Q2,op='nansum') 
		# Compute alpha
		alpha = Q1/Q2
		if it%refresh == 0:
			# Update solution
			for ii in range(n):
				x[ii] += alpha*d[ii]
			# Update residual
			spmv_scalar(&u[0],nzdom,rdom,cdom,Ad,x,n) # u = A*x
			commu.communicate_scaf(u)
			for ii in range(n):
				r[ii] = b[ii] - u[ii]
		else:
			# Update solution
			for ii in range(n):
				x[ii] += alpha*d[ii]		
				r[ii] -= alpha*u[ii]
		# Error computation
		err = dot(r,r,n)
		err = commu.allreduce(err,op='nansum') 
		if err < stop: break
		# Update direction
		beta = err/Q1
		for ii in range(n):
			d[ii] = r[ii] + beta*d[ii]

	# Update solution
	memcpy(&b[0],x,n*sizeof(double))

	# Free memory
	free(x)
	free(r)
	free(d)

	# Warning message
	if it == iters: raiseWarning('solver conjgrad maximum iterations reached (error=%.2e)!'%err)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn offq negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
cdef void conjgrad_vector_sp(int nzdom, int[:] rdom, int[:] cdom, double[:] Ad,double[:,:] b,int iters,int refresh,double tol):
	'''
	Conjugate gradient solver for scalar arrays
	'''
	# Define variables
	cdef int ii, it, ixdim, iz, rowb, rowe, n = b.shape[0], ndim = b.shape[1]
	cdef double err, stop, Q1, Q2, alpha, beta
	cdef double *x
	cdef double *r
	cdef double *d
	cdef double *u

	# Allocate memory
	x = <double*>malloc(n*sizeof(double))
	r = <double*>malloc(n*sizeof(double))
	d = <double*>malloc(n*sizeof(double))
	u = <double*>malloc(n*sizeof(double))

	for idim in range(ndim):
		# Initialize solver using the diagonal of A as a preconditioner
		for ii in range(n):
			rowb = rdom[ii]#+1
			rowe = rdom[ii+1]
			for iz in range(rowb,rowe):
				if cdom[iz] == ii: u[ii] = Ad[iz]
			x[ii] = b[ii,idim]/u[ii]

		# Initialize residual and direction
		spmv_scalar(&u[0],nzdom,rdom,cdom,Ad,x,n) # u = A*x
		for ii in range(n):
			r[ii] = b[ii,idim] - u[ii] # r = b - A*x
			d[ii] = r[ii]              # d = r
			u[ii] = b[ii,idim]

		# Error computation
		err  = dot(r,r,n)
		stop = tol*dot(u,u,n)

		# Start iterations
		for it in range(iters):
			spmv_scalar(&u[0],nzdom,rdom,cdom,Ad,d,n) # u = A*x
			Q1 = err
			Q2 = dot(d,&u[0],n)
			# Compute alpha
			alpha = Q1/Q2
			# Compute alpha
			alpha = Q1/Q2
			if it%refresh == 0:
				# Update solution
				for ii in range(n):
					x[ii] += alpha*d[ii]
				# Update residual
				spmv_scalar(&u[0],nzdom,rdom,cdom,Ad,x,n) # u = A*x
				for ii in range(n):
					r[ii] = b[ii,idim] - u[ii]				
			else:
				# Update solution
				for ii in range(n):
					x[ii] += alpha*d[ii]
					r[ii] -= alpha*u[ii]
			# Error computation
			err = dot(r,r,n)
			if err < stop: break
			# Update direction
			beta = err/Q1
			for ii in range(n):
				d[ii] = r[ii] + beta*d[ii]

		# Update solution
		for ii in range(n):
			b[ii,idim] = x[ii]

	# Free memory
	free(x)
	free(r)
	free(d)
	free(u)

	# Warning message
	if it == iters: raiseWarning('solver conjgrad maximum iterations reached (error=%.2e)!'%err)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
cdef void conjgrad_vector_sp_comm(int nzdom, int[:] rdom, int[:] cdom, double[:] Ad,double[:,:] b,int iters,int refresh,double tol,object commu):
	'''
	Conjugate gradient solver for scalar arrays
	'''
	# Define variables
	cdef int ii, it, ixdim, iz, rowb, rowe, n = b.shape[0], ndim = b.shape[1]
	cdef double err, stop, Q1, Q2, alpha, beta

	cdef np.ndarray[np.double_t,ndim=1] u = np.zeros((n,),np.double)
	cdef double *x
	cdef double *r
	cdef double *d

	# Allocate memory
	x = <double*>malloc(n*sizeof(double))
	r = <double*>malloc(n*sizeof(double))
	d = <double*>malloc(n*sizeof(double))

	for idim in range(ndim):
		# Initialize solver using the diagonal of A as a preconditioner
		for ii in range(n):
			Q1   = 1.
			rowb = rdom[ii]#+1
			rowe = rdom[ii+1]
			for iz in range(rowb,rowe):
				if cdom[iz] == ii: u[ii] = Ad[iz]
		commu.communicate_scaf(u)
		for ii in range(n):
			x[ii] = b[ii,idim]/u[ii]
		
		# Initialize residual and direction
		spmv_scalar(&u[0],nzdom,rdom,cdom,Ad,x,n) # u = A*x
		commu.communicate_scaf(u)
		for ii in range(n):
			r[ii] = b[ii,idim] - u[ii] # r = b - A*x
			d[ii] = r[ii]              # d = r
			u[ii] = b[ii,idim]

		# Error computation
		err  = dot(r,r,n)
		err  = commu.allreduce(err,op='nansum')
		stop = tol*dot(&u[0],&u[0],n)
		stop = commu.allreduce(stop,op='nansum')

		# Start iterations
		for it in range(iters):
			spmv_scalar(&u[0],nzdom,rdom,cdom,Ad,d,n) # u = A*d
			commu.communicate_scaf(u)
			Q1 = err
			Q2 = dot(d,&u[0],n)
			Q2 = commu.allreduce(Q2,op='nansum')
			# Compute alpha
			alpha = Q1/Q2
			if it%refresh == 0:
				# Update solution
				for ii in range(n):
					x[ii] += alpha*d[ii]
				# Update residual
				spmv_scalar(&u[0],nzdom,rdom,cdom,Ad,x,n) # u = A*x
				commu.communicate_scaf(u)
				for ii in range(n):
					r[ii] = b[ii,idim] - u[ii]
			else:
				# Update solution
				for ii in range(n):
					x[ii] += alpha*d[ii]		
					r[ii] -= alpha*u[ii]
			# Error computation
			err = dot(r,r,n)
			err = commu.allreduce(err,op='nansum')
			if err < stop: break
			# Update direction
			beta = err/Q1
			for ii in range(n):
				d[ii] = r[ii] + beta*d[ii]

		# Update solution
		for ii in range(n):
			b[ii,idim] = x[ii]

	# Free memory
	free(x)
	free(r)
	free(d)

	# Warning message
	if it == iters: raiseWarning('solver conjgrad maximum iterations reached (error=%.2e)!'%err)

@cr('solver.conjgrad')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
def solver_conjgrad(object A,np.ndarray b,int iters=500,int refresh=20,double tol=1e-8,object commu=None,int b_global=False):
	'''
	Solve a linar system such as 
		b = A*x
	using the conjugate gradient method 
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
				conjgrad_scalar_full(A,b,iters,refresh,tol)
			else:
				if not b_global: commu.communicate_scaf(b)
				conjgrad_scalar_full_comm(A,b,iters,refresh,tol,commu)
		else:
			# Vectorial field
			if commu == None:
				conjgrad_vector_full(A,b,iters,refresh,tol)
			else:
				if not b_global: commu.communicate_arrf(b)
				conjgrad_vector_full_comm(A,b,iters,refresh,tol,commu)
	else:
		nzdom = A.nnz
		rdom  = A.indptr
		cdom  = A.indices
		Ad    = A.data
		# Sparse matrix algorithms
		if len((<object> b).shape) == 1: 
			# Scalar field
			if commu == None:
				conjgrad_scalar_sp(nzdom,rdom,cdom,Ad,b,iters,refresh,tol)
			else:
				if not b_global: commu.communicate_scaf(b)
				conjgrad_scalar_sp_comm(nzdom,rdom,cdom,Ad,b,iters,refresh,tol,commu)
		else:
			# Vectorial field
			if commu == None:
				conjgrad_vector_sp(nzdom,rdom,cdom,Ad,b,iters,refresh,tol)
			else:
				if not b_global: commu.communicate_arrf(b)
				conjgrad_vector_sp_comm(nzdom,rdom,cdom,Ad,b,iters,refresh,tol,commu)
	return b