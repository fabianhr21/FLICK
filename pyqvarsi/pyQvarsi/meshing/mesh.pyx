#!/usr/bin/env cython
#
# pyQvarsi, utils.
#
# Meshing utility routines.
#
# Last rev: 10/06/2021
from __future__ import print_function, division

cimport numpy as np
cimport cython

from libc.math   cimport sqrt, tanh

import numpy as np

from ..utils.common import raiseError
from ..cr           import cr


@cython.cdivision(True)    # turn off zero division check
cdef void advance_direction(double *p, double *v, double d, double H, int conc, double f):
	cdef int ii
	if conc == 0:
		for ii in range(3): p[ii] += d*v[ii]
	else:
		for ii in range(3): p[ii] += H/2.*(1 + ( tanh( f*(2*d*v[ii]/H-1) )/tanh(f) ))

@cr('meshing.planeMesh')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def planeMesh(double[:] p1,double[:] p2,double[:] p4,int n1,int n2,int conc=0,double f=0.2):
	'''
	3D mesh plane, useful for slices.

	4-------3
	|		|
	|		|
	1-------2

	Can concentrate in axis 1 or 2. 
	A value of 3 concentrates in both 1 and 2.
	A value of 0 keeps the mesh uniform.
	'''
	cdef int idx, ii, jj, nt = n1*n2, cx, cy, ne = (n1-1)*(n2-1)
	cdef double L12, L14, dx, dy
	cdef np.ndarray[np.double_t,ndim=1] p   = np.ndarray((3,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] v12 = np.ndarray((3,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] v14 = np.ndarray((3,),dtype=np.double)

	cdef np.ndarray[np.double_t,ndim=2] coord = np.zeros((nt,3),dtype=np.double)
	cdef np.ndarray[np.int32_t,ndim=1]  lninv = np.zeros((nt,),dtype=np.int32)
	cdef np.ndarray[np.int32_t ,ndim=2] lnods = np.zeros((ne,4),dtype=np.int32)
	cdef np.ndarray[np.int32_t ,ndim=1] ltype = np.zeros((ne,),dtype=np.int32)
	cdef np.ndarray[np.int32_t,ndim=1]  leinv = np.zeros((ne,),dtype=np.int32)

	if conc > 3: raiseError('Invalid concentration parameter %d!'%conc)
	
	# Director unitary vectors
	for ii in range(3):
		v12[ii] = p2[ii] - p1[ii]
		v14[ii] = p4[ii] - p1[ii]
	L12 = sqrt(v12[0]*v12[0]+v12[1]*v12[1]+v12[2]*v12[2])
	L14 = sqrt(v14[0]*v14[0]+v14[1]*v14[1]+v14[2]*v14[2])
	for ii in range(3):
		v12[ii] /= L12
		v14[ii] /= L14

	# Generate the points
	idx = 0
	dx  = 1./(n1-1.)
	dy  = 1./(n2-1.)
	cx  = 0 if conc != 1 and conc != 3 else 1
	cy  = 0 if conc != 2 and conc != 3 else 1
	for jj in range(n2):
		# Start at p1
		p[0] = p1[0]
		p[1] = p1[1]
		p[2] = p1[2]
		# Advance
		advance_direction(&p[0],&v14[0],jj*dy*L14,L14,cy,f)
		for ii in range(n1):
			coord[idx,0] = p[0]
			coord[idx,1] = p[1]
			coord[idx,2] = p[2]
			advance_direction(&coord[idx,0],&v12[0],ii*dx*L12,L12,cy,f)
			lninv[idx] = idx
			idx       += 1

	# Create mesh arrays (elements)
	idx = 0
	for jj in range(n2-1):
		for ii in range(n1-1):
			lnods[idx,0] = 0    + ii + n1*jj
			lnods[idx,1] = 1    + ii + n1*jj
			lnods[idx,2] = n1+1 + ii + n1*jj
			lnods[idx,3] = n1   + ii + n1*jj
			ltype[idx]     = 12 # QUA04
			leinv[idx] = idx
			idx += 1

	return coord,lnods,ltype,lninv,leinv


@cr('meshing.cubeMesh')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def cubeMesh(double[:] p1,double[:] p2,double[:] p4,double[:] p5,int n1,int n2,int n3,
	int conc=0,double f=0.2):
	'''
	3D mesh cube, useful for volumes.

	  8-------7
	 /|      /|
	4-------3 |
	| 5-----|-6
	|/      |/
	1-------2

	Can concentrate in axis 1, 2 or 3. 
	A value of 4 concentrates in both 1 and 2.
	A value of 0 keeps the mesh uniform.
	'''
	cdef int idx, ii, jj, kk, nt = n1*n2*n3, cx, cy, cz, ne = (n1-1)*(n2-1)*(n3-1)
	cdef double L12, L14, L15, dx, dy, dz
	cdef np.ndarray[np.double_t,ndim=1] aux = np.ndarray((3,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] p   = np.ndarray((3,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] v12 = np.ndarray((3,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] v14 = np.ndarray((3,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] v15 = np.ndarray((3,),dtype=np.double)

	cdef np.ndarray[np.double_t,ndim=2] coord = np.zeros((nt,3),dtype=np.double)
	cdef np.ndarray[np.int32_t,ndim=1]  lninv = np.zeros((nt,),dtype=np.int32)
	cdef np.ndarray[np.int32_t ,ndim=2] lnods = np.zeros((ne,8),dtype=np.int32)
	cdef np.ndarray[np.int32_t ,ndim=1] ltype = np.zeros((ne,),dtype=np.int32)
	cdef np.ndarray[np.int32_t,ndim=1]  leinv = np.zeros((ne,),dtype=np.int32)

	if conc > 4: raiseError('Invalid concentration parameter %d!',conc)
	
	# Director unitary vectors
	for ii in range(3):
		v12[ii] = p2[ii] - p1[ii]
		v14[ii] = p4[ii] - p1[ii]
		v15[ii] = p5[ii] - p1[ii]
	L12 = sqrt(v12[0]*v12[0]+v12[1]*v12[1]+v12[2]*v12[2])
	L14 = sqrt(v14[0]*v14[0]+v14[1]*v14[1]+v14[2]*v14[2])
	L15 = sqrt(v15[0]*v15[0]+v15[1]*v15[1]+v15[2]*v15[2])
	for ii in range(3):
		v12[ii] /= L12
		v14[ii] /= L14
		v15[ii] /= L15

	# Create the mesh arrays
	idx = 0
	dx  = 1./(n1-1.)
	dy  = 1./(n2-1.)
	dz  = 1./(n3-1.)
	cx  = 0 if conc != 1 and conc != 4 else 1
	cy  = 0 if conc != 2 and conc != 4 else 1
	cz  = 0 if conc != 3 and conc != 4 else 1
	for kk in range(n3):
		# Start at p1
		p[0] = p1[0]
		p[1] = p1[1]
		p[2] = p1[2]
		# Advance
		advance_direction(&p[0],&v15[0],kk*dz*L15,L15,cz,f)
		for jj in range(n2):
			aux[0] = 0
			aux[1] = 0
			aux[2] = 0
			advance_direction(&aux[0],&v14[0],jj*dy*L14,L14,cy,f)
			for ii in range(n1):
				coord[idx,0] = p[0] + aux[0]
				coord[idx,1] = p[1] + aux[1]
				coord[idx,2] = p[2] + aux[2]
				advance_direction(&coord[idx,0],&v12[0],ii*dx*L12,L12,cy,f)
				lninv[idx] = idx
				idx       += 1

	# Create mesh arrays (elements)
	idx = 0
	for kk in range(n3-1):
		for jj in range(n2-1):
			for ii in range(n1-1):
				lnods[idx,0] = 0          + ii + n1*jj + n1*n2*kk
				lnods[idx,1] = 1          + ii + n1*jj + n1*n2*kk
				lnods[idx,2] = n1+1       + ii + n1*jj + n1*n2*kk
				lnods[idx,3] = n1         + ii + n1*jj + n1*n2*kk
				lnods[idx,4] = n1*n2      + ii + n1*jj + n1*n2*kk
				lnods[idx,5] = n1*n2+1    + ii + n1*jj + n1*n2*kk
				lnods[idx,6] = n1*n2+n1+1 + ii + n1*jj + n1*n2*kk
				lnods[idx,7] = n1*n2+n1   + ii + n1*jj + n1*n2*kk
				ltype[idx]   = 37 # HEX08
				leinv[idx]   = idx
				idx += 1
	
	return coord,lnods,ltype,lninv, leinv