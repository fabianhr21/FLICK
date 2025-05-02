#!/usr/bin/env python
#
# pyQvarsi, MATH tensor.
#
# Module to compute mathematical operations between
# scalar, vectorial and tensor arrays.
#
# Tensor operations (3x3 matrices).
#
# Last rev: 18/11/2020
from __future__ import print_function, division

import numpy as np, scipy


def identity(A):
	'''
	Identity tensor of the same size of A.
	'''
	I = np.zeros(A.shape,dtype=A.dtype)
	n = int(np.sqrt(A.shape[1]))
	for ii in range(n):
		I[:,n*ii+ii] = 1.
	return I

def transpose(A):
	'''
	Transposes the array tensor A into A^t.
	'''
	A_t = np.zeros(A.shape,dtype=A.dtype)
	if A_t.shape[0] == 4:
		A_t[:,0] = A[:,0]
		A_t[:,1] = A[:,3]
		A_t[:,2] = A[:,2]
		A_t[:,3] = A[:,1]
	if A_t.shape[1] == 9:
		A_t[:,0] = A[:,0]
		A_t[:,1] = A[:,3]
		A_t[:,2] = A[:,6]
		A_t[:,3] = A[:,1]
		A_t[:,4] = A[:,4]
		A_t[:,5] = A[:,7]
		A_t[:,6] = A[:,2]
		A_t[:,7] = A[:,5]
		A_t[:,8] = A[:,8]
	return A_t

def trace(A):
	'''
	Computes the trace of a tensor array A
	'''
	return A[:,0] + A[:,4] + A[:,8]

def det(A):
	'''
	Computes the determinant of a tensor array A
	'''
	return A[:,0]*(A[:,4]*A[:,8]-A[:,5]*A[:,7]) + \
	       A[:,1]*(A[:,5]*A[:,6]-A[:,3]*A[:,8]) + \
	       A[:,2]*(A[:,3]*A[:,7]-A[:,4]*A[:,6])

def inverse(A):
	'''
	Computes the inverse of a tensor array A,
	A is assumed not to be singular
	'''
	nn,mm = A.shape[0],A.shape[1]
	m = int(np.sqrt(mm))
	C = np.zeros((nn,2*mm),dtype=A.dtype)
	# Set up the matrix
	C[:,:mm] = A
	for ii in range(m):
		for jj in range(2*m):
			if jj == (ii+m):
				ind = m*ii+(jj-m)+mm if jj>=m else m*ii+jj
				C[:,ind] = 1.
	# Reducing to diagonal matrix
	for ii in range(m):
		for jj in range(m):
			if not jj == ii:
				ind1 = m*jj+(ii-m)+mm if ii>=m else m*jj+ii
				ind2 = m*ii+(ii-m)+mm if ii>=m else m*ii+ii
				aux = C[:,ind1]/C[:,ind2]
				for kk in range(2*m):
					ind1 = m*jj+(kk-m)+mm if kk>=m else m*jj+kk
					ind2 = m*ii+(kk-m)+mm if kk>=m else m*ii+kk
					C[:,ind1] -= C[:,ind2]*aux
 	# Reducing to unit matrix
	for ii in range(m):
		aux = C[:,m*ii+ii].copy()
		for jj in range(2*m):
			ind = m*ii+(jj-m)+mm if jj>=m else m*ii+jj
			C[:,ind] /= aux
	return C[:,mm:]

def matmul(A,B):
	'''
	Computes the matrix multiplication of two tensors
	A, B of the same shape (3x3)
	'''
	C = np.zeros(A.shape,dtype=A.dtype)
	for i in range(3):
		for j in range(3):
			C[:,3*i+j] = 0
			for k in range(3):
				C[:,3*i+j] += A[:,3*i+k]*B[:,j+3*k]
	return C

def doubleDot(A,B):
	'''
	Computes the double dot product (A:B or A_ijB_ij) 
	between two tensors.
	'''
	return np.sum(A*B,axis=1)

def tripleDot(A,B,C):
	'''
	Computes AijBjkCki
	'''
	c = np.zeros((A.shape[0],),dtype=A.dtype)
	for i in range(3):
		for j in range(3):
			for k in range(3):
				c[:] += A[:,3*i+j]*B[:,3*j+k]*C[:,3*k+i]
	return c

def quatrupleDot(A,B,C,D):
	'''
	Computes AijBjkCklDli
	'''
	c = np.zeros((A.shape[0],),dtype=A.dtype)
	for i in range(3):
		for j in range(3):
			for k in range(3):
				for l in range(3):
					c[:] += A[:,3*i+j]*B[:,3*j+k]*C[:,3*k+l]*D[:,3*l+i]
	return c

def scaTensProd(k,A):
	'''
	Computes the product of a scalar times a tensor.
	'''
	C = np.zeros(A.shape,dtype=A.dtype)
	for ii in range(C.shape[1]):
		C[:,ii] = k*A[:,ii]
	return C

def tensVecProd(A,b):
	'''
	Computes the product of a tensor times a vector.
	'''
	c = np.zeros(b.shape,dtype=b.dtype)
	for ii in range(c.shape[1]):
		for jj in range(c.shape[1]):
			c[:,ii] += A[:,c.shape[1]*ii+jj]*b[:,jj]
	return c

def tensNorm(A):
	'''
	Computes the L2 norm of a tensor.
	'''
	return np.sqrt(doubleDot(A,A))

def eigenvalues(A):
	'''
	Computes the eigenvalues of A and returns them
	so that l1 > l2 > l3
	'''
	n, m = A.shape[0], int(np.sqrt(A.shape[1]))
	e = np.zeros((n,m),dtype=np.double)
	for ii in range(n):
		R = np.reshape(A[ii,:],(m,m))
		# Compute eigenvalues and eigenvectors
		e[ii,:] = np.sort(np.linalg.eigvalsh(R)) # Sort eigenvalues so that l1 > l2 > l3
	return e

def schur(A):
	'''
	Computes the schur decomposition of A
	'''
	n, m = A.shape[0], int(np.sqrt(A.shape[1]))
	S = np.zeros((n,m*m),dtype=np.double)
	Q = np.zeros((n,m*m),dtype=np.double)
	sortfun = lambda xr,xi: (abs(xi) > 0.0)
	for ii in range(n):
		R = np.reshape(A[ii,:],(m,m))
		# Compute schur decomposition
		ss, qq, sdim = scipy.linalg.schur(R,sort=sortfun)
		S[ii,:] = ss.reshape((m*m,))
		Q[ii,:] = qq.reshape((m*m,))
	return S, Q

def tensRotate(A,gamma,beta,alpha):
	'''
	Rotate a vectorial array given some angles and a center.
	'''
	# Convert to radians
	alpha = np.deg2rad(alpha)
	beta  = np.deg2rad(beta)
	gamma = np.deg2rad(gamma)

	# Define rotation matrix
	R = np.ndarray((A.shape[0],9),dtype=np.double)
	R[:,0] = np.cos(alpha)*np.cos(beta)
	R[:,3] = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
	R[:,6] = np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)
	R[:,1] = np.sin(alpha)*np.cos(beta)
	R[:,4] = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
	R[:,7] = np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)
	R[:,2] = -np.sin(beta)
	R[:,5] = np.cos(beta)*np.sin(gamma)
	R[:,8] = np.cos(beta)*np.cos(gamma)

	# Rotate
	return matmul(R,A)