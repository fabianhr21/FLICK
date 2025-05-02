#!/usr/bin/env cython
#
# pyQvarsi, MATH utils.
#
# Module to compute mathematical operations between
# scalar, vectorial and tensor arrays.
#
# Useful utilities.
#
# Last rev: 28/12/2020
from __future__ import print_function, division

import numpy as np

cimport numpy as np
cimport cython
from libc.math cimport sqrt


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def linopScaf(double a,double[:] scaf1,double b,double[:] scaf2):
	'''
	Linear operations between two scalar fields
	'''
	cdef int ii, n = scaf1.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] c = np.zeros((n,),dtype=np.double)
	for ii in range(n):
		c[ii] = a*scaf1[ii] + b*scaf2[ii]
	return c

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def linopArrf(double a,double[:,:] arrf1,double b,double[:,:] arrf2):
	'''
	Linear operations between two array fields
	'''
	cdef int ii, jj, n = arrf1.shape[0], m = arrf1.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] c = np.zeros((n,m),dtype=np.double)
	for ii in range(n):
		for jj in range(m):
			c[ii,jj] = a*arrf1[ii,jj] + b*arrf2[ii,jj]
	return c

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def maxVal(double[:] a,double b):
	'''
	Maximum between an array a and a value b
	'''
	cdef int ii, n = a.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((n,),dtype=np.double)
	for ii in range(n):
		out[ii] = max(a[ii],b)
	return out;

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def minVal(double[:] a,double b):
	'''
	Minimum between an array a and a value b
	'''
	cdef int ii, n = a.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((n,),dtype=np.double)
	for ii in range(n):
		out[ii] = min(a[ii],b)
	return out;

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def maxArr(double[:] a,double[:] b):
	'''
	Element-wise maximum between two arrays
	'''
	cdef int ii, n = a.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((n,),dtype=np.double)
	for ii in range(n):
		out[ii] = max(a[ii],b[ii])
	return out;

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def minArr(double[:] a,double[:] b):
	'''
	Element-wise minimum between two arrays
	'''
	cdef int ii, n = a.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((n,),dtype=np.double)
	for ii in range(n):
		out[ii] = min(a[ii],b[ii])
	return out;

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def deltaKronecker(int i,int j):
	'''
	Returns the Kronecker delta, 1 if i == j else 0.
	'''
	return 1. if i == j else 0.

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def alternateTensor(int i,int j,int k):
	'''
	Returns the alternating tensor:
		e_123 = e_231 = e_312 = 1
		e_321 = e_213 = e_132 = -1
	for the rest is 0 
	'''
	if (i,j,k) == (1,2,3) or (i,j,k) == (2,3,1) or (i,j,k) == (3,1,2):
		return 1.
	if (i,j,k) == (3,2,1) or (i,j,k) == (2,1,3) or (i,j,k) == (1,3,2):
		return -1.
	return 0.

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def reorder1to2(double[:,:] xyz1,double[:,:] xyz2):
	'''
	Find the indices that reorder 1 to be equal to 2
	'''
	cdef int ii, n1 = xyz1.shape[0], jj, n2 = xyz2.shape[0], idx, start, end, found
	cdef int offset = 2500
	cdef double d, d2, epsi = 1e-6
	cdef np.ndarray[np.int32_t,ndim=1] order = np.zeros((n1,),dtype=np.int32)
	# Loop each point of xyz1
	for ii in range(n1):
		found = False
		idx   = 0 if ii == 0 else order[ii-1]
		# Do a local search close to idx
		start = max(0,idx-offset)
		end   = min(idx+offset,n2)
		for jj in range(start,end):
			# Compute the difference
			d2 = (xyz1[ii,0] - xyz2[jj,0])*(xyz1[ii,0] - xyz2[jj,0]) + \
				 (xyz1[ii,1] - xyz2[jj,1])*(xyz1[ii,1] - xyz2[jj,1]) + \
				 (xyz1[ii,2] - xyz2[jj,2])*(xyz1[ii,2] - xyz2[jj,2])
			# Compute square root
			d = sqrt(d2)
			# Find the value
			if d < epsi: 
				idx = jj
				found = True
				break
		# Search at the beginning
		if not found:
			start = 0
			end   = idx-offset
			for jj in range(start,end):
				# Compute the difference
				d2 = (xyz1[ii,0] - xyz2[jj,0])*(xyz1[ii,0] - xyz2[jj,0]) + \
					 (xyz1[ii,1] - xyz2[jj,1])*(xyz1[ii,1] - xyz2[jj,1]) + \
					 (xyz1[ii,2] - xyz2[jj,2])*(xyz1[ii,2] - xyz2[jj,2])
				# Compute square root
				d = sqrt(d2)
				# Find the value
				if d < epsi: 
					idx = jj
					found = True
					break
		# Search at the end
		if not found:
			start = idx+offset
			end   = n2
			for jj in range(start,end):
				# Compute the difference
				d2 = (xyz1[ii,0] - xyz2[jj,0])*(xyz1[ii,0] - xyz2[jj,0]) + \
					 (xyz1[ii,1] - xyz2[jj,1])*(xyz1[ii,1] - xyz2[jj,1]) + \
					 (xyz1[ii,2] - xyz2[jj,2])*(xyz1[ii,2] - xyz2[jj,2])
				# Compute square root
				d = sqrt(d2)
				# Find the value
				if d < epsi: 
					idx = jj
					found = True
					break	
		# Set the value at order
		order[ii] = idx
	return order