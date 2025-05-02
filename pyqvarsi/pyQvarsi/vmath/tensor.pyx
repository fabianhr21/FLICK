#!/usr/bin/env cython
#
# pyQvarsi, MATH tensor.
#
# Module to compute mathematical operations between
# scalar, vectorial and tensor arrays.
#
# Tensor operations (3x3 matrices).
#
# Last rev: 28/12/2020
# cython: legacy_implicit_noexcept=True
from __future__ import print_function, division

import numpy as np

cimport numpy as np
cimport cython

from libc.math                  cimport sqrt, fabs, sin, cos
from libc.stdlib                cimport malloc, free, qsort
from libc.string                cimport memcpy
from scipy.linalg.cython_lapack cimport dgeev, dgees


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef int compare(const void* a, const void* b) nogil:
	cdef double *aa = <double*>a
	cdef double *bb = <double*>b
	if (aa[0]==bb[0]):  return 0
	if (aa[0] < bb[0]): return -1
	return 1

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef bint sortfun(double* ER, double* EI):
	return fabs(EI[0]) > 0


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def identity(double[:,:] A):
	'''
	Identity tensor of the same size of A.
	'''
	cdef int ii,i,j,n = A.shape[0], m = int(sqrt(A.shape[1]))
	cdef np.ndarray[np.double_t,ndim=2] C = np.ndarray((n,A.shape[1]),dtype=np.double)
	for ii in range(n):
		for i in range(m):
			for j in range(m):
				C[ii,m*i+j] = 1. if i == j else 0.
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def transpose(double[:,:] A):
	'''
	Transposes the array tensor A into A^t.
	'''
	cdef int ii, jj, kk, n = A.shape[0], m = A.shape[1], m2 = <int>sqrt(m)
	cdef np.ndarray[np.double_t,ndim=2] C = np.ndarray((n,m),dtype=np.double)
	for ii in range(n):
		for jj in range(m2):
			for kk in range(m2):
				C[ii,m2*jj+kk] = A[ii,m2*kk+jj]
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def trace(double[:,:] A):
	'''
	Computes the trace of a tensor array A
	'''
	cdef int ii,n = A.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] C = np.ndarray((n,),dtype=np.double)
	for ii in range(n):
		C[ii] = A[ii,0] + A[ii,4] + A[ii,8]
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def det(double[:,:] A):
	'''
	Computes the determinant of a tensor array A
	'''
	cdef int ii,n = A.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] C = np.ndarray((n,),dtype=np.double)
	for ii in range(n):
		C[ii] = A[ii,0]*(A[ii,4]*A[ii,8]-A[ii,5]*A[ii,7]) + \
				A[ii,1]*(A[ii,5]*A[ii,6]-A[ii,3]*A[ii,8]) + \
				A[ii,2]*(A[ii,3]*A[ii,7]-A[ii,4]*A[ii,6])
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
@cython.nonecheck(False)
def inverse(double[:,:] A):
	'''
	Computes the inverse of a tensor array A
	'''
	cdef int ii,i,j,k,n = A.shape[0],mm = A.shape[1], m = int(sqrt(mm))
	cdef double d
	cdef np.ndarray[np.double_t,ndim=2] C   = np.ndarray((n,mm),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] aux = np.ndarray((m,2*m),dtype=np.double)
	# Loop the whole points
	for ii in range(n):
		# First extract the submatrix that we are interested to work with
		for i in range(m):
			for j in range(2*m):
				aux[i,j] = 0. # Reset the f**** matrix
				if j < m:
					aux[i,j] = A[ii,m*i+j]
				if j == (i+m):
					aux[i,j] = 1.
		# Reduce to diagonal matrix
		for i in range(m):
			for j in range(m):
				if not j == i:
					d = aux[j,i]/aux[i,i]
					for k in range(2*m):
						aux[j,k] -= aux[i,k]*d
		# Reduce to unit matrix
		for i in range(m):
			d = aux[i,i]
			for j in range(2*m):
				aux[i,j] /= d
		# Copy to output
		for i in range(m):
			for j in range(m,2*m):
				C[ii,m*i+(j-m)] = aux[i,j]
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def matmul(double[:,:] A,double[:,:] B):
	'''
	Computes the matrix multiplication of two tensors
	A, B of the same shape (3x3)
	'''
	cdef int ii,i,j,k,n = A.shape[0]
	cdef np.ndarray[np.double_t,ndim=2] C = np.ndarray((n,9),dtype=np.double)
	for ii in range(n):
		for i in range(3):
			for j in range(3):
				C[ii,3*i+j] = 0
				for k in range(3):
					C[ii,3*i+j] += A[ii,3*i+k]*B[ii,j+3*k]
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def doubleDot(double[:,:] A,double[:,:] B):
	'''
	Computes the double dot product (A:B or A_ijB_ij) 
	between two tensors.
	'''
	cdef int ii,n = A.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] c = np.ndarray((n,),dtype=np.double)
	for ii in range(n):
		c[ii] = A[ii,0]*B[ii,0] + A[ii,1]*B[ii,1] + A[ii,2]*B[ii,2] +\
				A[ii,3]*B[ii,3] + A[ii,4]*B[ii,4] + A[ii,5]*B[ii,5] +\
				A[ii,6]*B[ii,6] + A[ii,7]*B[ii,7] + A[ii,8]*B[ii,8]
	return c

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def tripleDot(double[:,:] A,double[:,:] B, double[:,:] C):
	'''
	Computes AijBjkCki
	'''
	cdef int ii,i,j,k,n = A.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] c = np.ndarray((n,),dtype=np.double)
	for ii in range(n):
		c[ii] = 0.
		for i in range(3):
			for j in range(3):
				for k in range(3):
					c[ii] += A[ii,3*i+j]*B[ii,3*j+k]*C[ii,3*k+i]
	return c

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def quatrupleDot(double[:,:] A,double[:,:] B, double[:,:] C, double[:,:] D):
	'''
	Computes AijBjkCklDli
	'''
	cdef int ii,i,j,k,l,n = A.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] c = np.ndarray((n,),dtype=np.double)
	for ii in range(n):
		c[ii] = 0.
		for i in range(3):
			for j in range(3):
				for k in range(3):
					for l in range(3):
						c[ii] += A[ii,3*i+j]*B[ii,3*j+k]*C[ii,3*k+l]*D[ii,3*l+i]
	return c

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def scaTensProd(double[:] k,double[:,:] A):
	'''
	Computes the product of a scalar times a tensor.
	'''
	cdef int ii,jj,n = A.shape[0],m = A.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] C = np.ndarray((n,m),dtype=np.double)
	for ii in range(n):
		for jj in range(m):
			C[ii,jj] = k[ii]*A[ii,jj]
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def tensVecProd(double[:,:] A,double[:,:] b):
	'''
	Computes the product of a tensor times a vector.
	'''
	cdef int ii,jj,kk,n = A.shape[0], m = b.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] c = np.ndarray((n,m),dtype=np.double)
	for ii in range(n):
		for jj in range(m):
			c[ii,jj] = 0.
			for kk in range(m):
				c[ii,jj] += A[ii,m*jj+kk]*b[ii,kk]
	return c

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def tensNorm(double[:,:] A):
	'''
	Computes the L2 norm of a tensor.
	'''
	cdef int ii,n = A.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] c = np.ndarray((n,),dtype=np.double)
	for ii in range(n):
		c[ii] = sqrt(A[ii,0]*A[ii,0] + A[ii,1]*A[ii,1] + A[ii,2]*A[ii,2] +\
					 A[ii,3]*A[ii,3] + A[ii,4]*A[ii,4] + A[ii,5]*A[ii,5] +\
					 A[ii,6]*A[ii,6] + A[ii,7]*A[ii,7] + A[ii,8]*A[ii,8])
	return c

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def eigenvalues(double[:,:] A):
	'''
	Computes the eigenvalues of A and returns them
	so that l1 > l2 > l3
	'''
	cdef int ii, n = A.shape[0], m = <int>sqrt(A.shape[1])
	cdef np.ndarray[np.double_t,ndim=2] out = np.ndarray((n,m),dtype=np.double)

	# Parameters for Lapack's *geev
	cdef char *jobVL = 'N' # The left eigenvector u(j) of A satisfies: u(j)**H * A = lambda(j) * u(j)**H. 'N' to not compute.
	cdef char *jobVR = 'N' # The right eigenvector v(j) of A satisfies: A * v(j) = lambda(j) * v(j). 'V' to compute.
	cdef double *B         # a (m,m) matrix
	cdef double *WR        # eigenvalues, real part
	cdef double *WI        # eigenvalues, imaginary part
	cdef double *VL        # eigenvectors, left part
	cdef double *VR        # eigenvectors, right part
	cdef double *work
	cdef int ldA = m, ldVL = 1, ldVR = m, lwork = 5*m, info

	B    = <double*>malloc(m*m*sizeof(double))
	WR   = <double*>malloc(m*sizeof(double))
	WI   = <double*>malloc(m*sizeof(double))
	VL   = <double*>malloc(1*sizeof(double))
	VR   = <double*>malloc(1*sizeof(double))
	work = <double*>malloc(lwork*sizeof(double))

	for ii in range(n):
		# Copy to B
		memcpy(B,&A[ii,0],m*m*sizeof(double))
		# Compute eigenvalues using dgeev
		dgeev(jobVL,jobVR,&m,B,&ldA,WR,WI,VL,&ldVL,VR,&ldVR,work,&lwork,&info)
		# Sort and store the eigenvalues
		qsort(WR,m,sizeof(double),compare)
		# Copy arrays to output
		memcpy(&out[ii,0],WR,m*sizeof(double))

	free(B)
	free(WR)
	free(WI)
	free(VL)
	free(VR)
	free(work)
	return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def schur(double[:,:] A):
	'''
	Computes the schur decomposition of A
	'''
	cdef int ii, n = A.shape[0], m = <int>sqrt(A.shape[1])
	cdef np.ndarray[np.double_t,ndim=2] S = np.ndarray((n,m*m),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] Q = np.ndarray((n,m*m),dtype=np.double)

	cdef char *jobVS = 'V' # Compute schur vectors
	cdef char *sort  = 'S'
	cdef double *B         # a (m,m) matrix
	cdef double *WR        # eigenvalues, real part
	cdef double *WI        # eigenvalues, imaginary part
	cdef double *VS        # orthogonal matrix Z of Schur vectors.
	cdef double *work
	cdef bint    *bwork
	cdef int ldA = m, sdim, ldVS = m, lwork = 5*m, info

	B     = <double*>malloc(ldA*m*sizeof(double))
	WR    = <double*>malloc(m*sizeof(double))
	WI    = <double*>malloc(m*sizeof(double))
	VS    = <double*>malloc(ldVS*m*sizeof(double))
	work  = <double*>malloc(lwork*sizeof(double))
	bwork = <bint*>malloc(m*sizeof(bint))

	for ii in range(n):
		# Copy to B
		memcpy(B,&A[ii,0],m*m*sizeof(double))
		# Compute schur decomposition
		dgees(jobVS,sort,sortfun,&m,B,&ldA,&sdim,WR,WI,VS,&ldVS,work,&lwork,bwork,&info)
		# Store output values
		memcpy(&S[ii,0],B,m*m*sizeof(double))
		memcpy(&Q[ii,0],VS,m*m*sizeof(double))

	free(B)
	free(WR)
	free(WI)
	free(VS)
	free(work)
	free(bwork)
	return S, Q

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def tensRotate(double[:,:] A,double gamma,double beta,double alpha):
	'''
	Rotate a tensorial array given some angles and a center.
	'''
	cdef int ii, i, j, k, npoints = A.shape[0]
	cdef double pi = np.pi
	cdef np.ndarray[np.double_t,ndim=2] R    = np.ndarray((3,3),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] out  = np.ndarray((npoints,9),dtype=np.double)
	
	# Convert to radians
	alpha = pi*alpha/180.0
	beta  = pi*beta/180.0	
	gamma = pi*gamma/180.0

	# Define rotation matrix
	R[0,0] = cos(alpha)*cos(beta)
	R[1,0] = cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma)
	R[2,0] = cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma)
	R[0,1] = sin(alpha)*cos(beta)
	R[1,1] = sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma)
	R[2,1] = sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma)
	R[0,2] = -sin(beta)
	R[1,2] = cos(beta)*sin(gamma)
	R[2,2] = cos(beta)*cos(gamma)

	# Rotate - compute R*A
	for ii in range(npoints):
		for i in range(3):
			for j in range(3):
				out[ii,3*i+j] = 0
				for k in range(3):
					out[ii,3*i+j] += R[i,k]*A[ii,j+3*k]

	return out