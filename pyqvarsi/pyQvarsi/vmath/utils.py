#!/usr/bin/env python
#
# pyQvarsi, MATH utils.
#
# Module to compute mathematical operations between
# scalar, vectorial and tensor arrays.
#
# Useful utilities.
#
# Last rev: 04/11/2020
from __future__ import print_function, division

import numpy as np


def linopScaf(a,scaf1,b,scaf2):
	'''
	Linear operations between two scalar fields
	'''
	return a*scaf1 + b*scaf2

def linopArrf(a,arrf1,b,arrf2):
	'''
	Linear operations between two array fields
	'''
	return a*arrf1 + b*arrf2

def maxVal(a,b):
	'''
	Maximum between an array a and a value b
	'''
	out = a.copy()
	out[out<b] = b
	return out;

def minVal(a,b):
	'''
	Minimum between an array a and a value b
	'''
	out = a.copy()
	out[out>b] = b
	return out;

def maxArr(a,b):
	'''
	Element-wise maximum between two arrays
	'''
	return np.maximum(a,b)

def minArr(a,b):
	'''
	Element-wise minimum between two arrays
	'''
	return np.minimum(a,b)

def deltaKronecker(i,j):
	'''
	Returns the Kronecker delta, 1 if i == j else 0.
	'''
	return 1. if i == j else 0.

def alternateTensor(i,j,k):
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

def reorder1to2(xyz1,xyz2):
	'''
	Find the indices that reorder 1 to be equal to 2
	'''
	order = np.zeros((xyz1.shape[0],),dtype=np.int32)
	# Loop each point of xyz1
	for ii in range(xyz1.shape[0]):
		# Compute the difference
		d2 = (xyz1[ii,0] - xyz2[:,0])*(xyz1[ii,0] - xyz2[:,0]) + \
			 (xyz1[ii,1] - xyz2[:,1])*(xyz1[ii,1] - xyz2[:,1]) + \
			 (xyz1[ii,2] - xyz2[:,2])*(xyz1[ii,2] - xyz2[:,2])
		# Find which index corresponds to mesh2
		order[ii] = np.argmin(d2)
	return order