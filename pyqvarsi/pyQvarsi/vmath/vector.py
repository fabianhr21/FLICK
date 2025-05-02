#!/usr/bin/env python
#
# pyQvarsi, MATH vector.
#
# Module to compute mathematical operations between
# scalar, vectorial and tensor arrays.
#
# Vectorial operations (3,) vectors.
#
# Last rev: 04/11/2020
from __future__ import print_function, division

import numpy as np


def dot(a,b):
	'''
	Computes the dot product between two vector arrays.
	'''
	return np.sum(a*b,axis=1)

def cross(a,b):
	'''
	Computes the cross product between two vector arrays.
	'''
	c = np.zeros((a.shape[0],3),dtype=a.dtype)
	c[:,0] = a[:,1]*b[:,2] - a[:,2]*b[:,1]
	c[:,1] = a[:,2]*b[:,0] - a[:,0]*b[:,2]
	c[:,2] = a[:,0]*b[:,1] - a[:,1]*b[:,0]
	return c

def outer(a,b):
	'''
	Computes the outer product between two vector arrays
	'''
	C = np.zeros((a.shape[0],9),dtype=a.dtype)
	C[:,0] = a[:,0]*b[:,0]
	C[:,1] = a[:,0]*b[:,1]
	C[:,2] = a[:,0]*b[:,2]
	C[:,3] = a[:,1]*b[:,0]
	C[:,4] = a[:,1]*b[:,1]
	C[:,5] = a[:,1]*b[:,2]
	C[:,6] = a[:,2]*b[:,0]
	C[:,7] = a[:,2]*b[:,1]
	C[:,8] = a[:,2]*b[:,2]
	return C

def scaVecProd(k,a):
	'''
	Computes the product of a scalar times a vector.
	'''
	c = np.zeros(a.shape,dtype=a.dtype)
	for ii in range(c.shape[1]):
		c[:,ii] = k*a[:,ii]
	return c

def vecTensProd(a,B):
	'''
	Computes the product of a vector times a tensor. 
	'''
	c = np.zeros(a.shape,dtype=a.dtype)
	l = c.shape[1]
	for ii in range(l):
		for jj in range(l):
			c[:,ii] +=  a[:,jj]*B[:,l*jj + ii]
	return c

def vecNorm(a):
	'''
	Computes the norm of a vector a
	'''
	return np.sqrt(dot(a,a))

def vecRotate(a,gamma,beta,alpha,center):
	'''
	Rotate a vectorial array given some angles and a center.
	'''
	# Convert to radians
	alpha = np.deg2rad(alpha)
	beta  = np.deg2rad(beta)
	gamma = np.deg2rad(gamma)

	# Define rotation matrix
	R = np.ndarray((a.shape[0],9),dtype=np.double)
	R[:,0] = np.cos(alpha)*np.cos(beta)
	R[:,1] = np.sin(alpha)*np.cos(beta)
	R[:,2] = -np.sin(beta)
	R[:,3] = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
	R[:,4] = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
	R[:,5] = np.cos(beta)*np.sin(gamma)
	R[:,6] = np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)
	R[:,7] = np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)
	R[:,8] = np.cos(beta)*np.cos(gamma)

	# Define center rotation
	rotc = np.ndarray((a.shape[0],3),dtype=np.double)
	rotc[:,0] = center[0]
	rotc[:,1] = center[1]
	rotc[:,2] = center[2]

	# Rotate
	out  = a.copy() - rotc
	out  = vecTensProd(out,R)
	out += rotc

	return out