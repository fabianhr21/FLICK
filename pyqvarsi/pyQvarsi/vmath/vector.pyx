#!/usr/bin/env cython
#
# pyQvarsi, MATH vector.
#
# Module to compute mathematical operations between
# scalar, vectorial and tensor arrays.
#
# Vectorial operations (3,) vectors.
#
# Last rev: 28/12/2020
from __future__ import print_function, division

import numpy as np

cimport numpy as np
cimport cython
from libc.math cimport sqrt, sin, cos


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def dot(double[:,:] a, double[:,:] b):
	'''
	Computes the dot product between two vector arrays.
	'''
	cdef int ii,n = a.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] c = np.ndarray((n,),dtype=np.double)
	for ii in range(n):
		c[ii] = a[ii,0]*b[ii,0] + a[ii,1]*b[ii,1] + a[ii,2]*b[ii,2]
	return c

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def cross(double[:,:] a, double[:,:] b):
	'''
	Computes the cross product between two vector arrays.
	'''
	cdef int ii,n = a.shape[0]
	cdef np.ndarray[np.double_t,ndim=2] c = np.ndarray((n,3),dtype=np.double)
	for ii in range(n):
		c[ii,0] = a[ii,1]*b[ii,2] - a[ii,2]*b[ii,1]
		c[ii,1] = a[ii,2]*b[ii,0] - a[ii,0]*b[ii,2]
		c[ii,2] = a[ii,0]*b[ii,1] - a[ii,1]*b[ii,0]
	return c

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def outer(double[:,:] a, double[:,:] b):
	'''
	Computes the outer product between two vector arrays
	'''
	cdef int ii,n = a.shape[0]
	cdef np.ndarray[np.double_t,ndim=2] C = np.ndarray((n,9),dtype=np.double)
	for ii in range(n):
		C[ii,0] = a[ii,0]*b[ii,0]
		C[ii,1] = a[ii,0]*b[ii,1]
		C[ii,2] = a[ii,0]*b[ii,2]
		C[ii,3] = a[ii,1]*b[ii,0]
		C[ii,4] = a[ii,1]*b[ii,1]
		C[ii,5] = a[ii,1]*b[ii,2]
		C[ii,6] = a[ii,2]*b[ii,0]
		C[ii,7] = a[ii,2]*b[ii,1]
		C[ii,8] = a[ii,2]*b[ii,2]
	return C

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def scaVecProd(double[:] k, double[:,:] a):
	'''
	Computes the product of a scalar times a vector.
	'''
	cdef int ii,jj,n = a.shape[0],m = a.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] c = np.ndarray((n,m),dtype=np.double)
	for ii in range(n):
		for jj in range(m):
			c[ii,jj] = k[ii]*a[ii,jj]
	return c

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def vecTensProd(double[:,:] a, double[:,:] B):
	'''
	Computes the product of a vector times a tensor. 
	'''
	cdef int ii,jj,kk,n = a.shape[0],m = B.shape[1], l = a.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] c = np.ndarray((n,m),dtype=np.double)
	for ii in range(n):
		for jj in range(m):
			c[ii,jj] = 0.
			for kk in range(m):
				c[ii,jj] += a[ii,kk]*B[ii,m*kk+jj]
	return c

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def vecNorm(double[:,:] a):
	'''
	Computes the dot product between two vector arrays.
	'''
	cdef int ii,n = a.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] c = np.ndarray((n,),dtype=np.double)
	for ii in range(n):
		c[ii] = sqrt(a[ii,0]*a[ii,0] + a[ii,1]*a[ii,1] + a[ii,2]*a[ii,2])
	return c

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def vecRotate(double[:,:] a,double gamma,double beta,double alpha,double[:] center):
	'''
	Rotate a vectorial array given some angles and a center.
	'''
	cdef int ii, npoints = a.shape[0]
	cdef double pi = np.pi
	cdef np.ndarray[np.double_t,ndim=1] aux1 = np.ndarray((3,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] aux2 = np.ndarray((3,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] R    = np.ndarray((3,3),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] out  = np.ndarray((npoints,3),dtype=np.double)
	
	# Convert to radians
	alpha = pi*alpha/180.0
	beta  = pi*beta/180.0	
	gamma = pi*gamma/180.0

	# Define rotation matrix
	R[0,0] = cos(alpha)*cos(beta)
	R[0,1] = sin(alpha)*cos(beta)
	R[0,2] = -sin(beta)
	R[1,0] = cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma)
	R[1,1] = sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma)
	R[1,2] = cos(beta)*sin(gamma)
	R[2,0] = cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma)
	R[2,1] = sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma)
	R[2,2] = cos(beta)*cos(gamma)

	for ii in range(npoints):
		# Subtract center
		aux1[0] = a[ii,0] - center[0]
		aux1[1] = a[ii,1] - center[1]
		aux1[2] = a[ii,2] - center[2]
		# Dot product
		aux2[0] = R[0,0]*aux1[0] + R[1,0]*aux1[1] + R[2,0]*aux1[2]
		aux2[1] = R[0,1]*aux1[0] + R[1,1]*aux1[1] + R[2,1]*aux1[2]
		aux2[2] = R[0,2]*aux1[0] + R[1,2]*aux1[1] + R[2,2]*aux1[2]
		# Add center
		aux1[0] = aux2[0] + center[0]
		aux1[1] = aux2[1] + center[1]
		aux1[2] = aux2[2] + center[2]		
		# Save output vector
		out[ii,0] = aux1[0]
		out[ii,1] = aux1[1]
		out[ii,2] = aux1[2]

	return out
