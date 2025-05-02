#!/usr/bin/env cython
#
# pyQvarsi, FEM grad.
#
# Small FEM module to compute derivatives and possible other
# simple stuff from Alya output for postprocessing purposes.
#
# FEM gradient according to Alya.
#
# Last rev: 30/09/2020
from __future__ import print_function, division

import numpy as np

cimport numpy as np
cimport cython

from ..utils.common import raiseError


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradScaf2D(double[:,:] xyz,double[:] field,object[:] elemList):
	'''
	Compute the gradient of a 2D scalar field given a list 
	of elements (internal function).

	Assemble of the gradients is not done here.

	IN:
		> xyz(nnod,2):   positions of the nodes
		> field(nnod,):  scalar field
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> gradient(nnod,2): gradient of scalar field
	'''
	cdef int           MAXNODES = 100
	cdef int           igauss, ielem, inod, nnod, ngauss, nelem = len(elemList), nnodtot = field.shape[0]
	cdef double        xfact
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:]   shapef
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.zeros((nnodtot,2),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,2),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] elgrad   = np.ndarray((2,),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,) ,dtype=np.double)
	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
		shapef  = elem.shape
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(2):
				elxyz[inod,idim] = xyz[nodes[inod],idim]
			elfield[inod] = field[nodes[inod]]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			elgrad[0] = 0.
			elgrad[1] = 0.
			# Compute element gradients
			for inod in range(nnod):
				elgrad[0] += deri[0,inod,igauss]*elfield[inod] # df/dx
				elgrad[1] += deri[1,inod,igauss]*elfield[inod] # df/dy
			# Assemble gradients
			for inod in range(nnod):
				xfact = vol[igauss] * shapef[inod,igauss]
				gradient[nodes[inod],0] += xfact * elgrad[0]
				gradient[nodes[inod],1] += xfact * elgrad[1]
	return gradient

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradScaf3D(double[:,:] xyz, double[:] field,object[:] elemList):
	'''
	Compute the gradient of a 3D scalar field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> field(nnod,):  scalar field
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> gradient(nnod,3): gradient of scalar field
	'''
	cdef int           MAXNODES = 1000
	cdef int           igauss, ielem, inod, nnod, ngauss, nelem = len(elemList), nnodtot = field.shape[0]
	cdef double        xfact
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:]   shapef
	cdef double[:,:,:] deri
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.zeros((nnodtot,3),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,3),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] elgrad   = np.ndarray((3,),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,) ,dtype=np.double)
	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
		shapef  = elem.shape
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(3):
				elxyz[inod,idim] = xyz[nodes[inod],idim]
			elfield[inod] = field[nodes[inod]]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			elgrad[0] = 0.
			elgrad[1] = 0.
			elgrad[2] = 0.
			# Compute element gradients
			for inod in range(nnod):
				elgrad[0] += deri[0,inod,igauss]*elfield[inod] # df/dx
				elgrad[1] += deri[1,inod,igauss]*elfield[inod] # df/dy
				elgrad[2] += deri[2,inod,igauss]*elfield[inod] # df/dz
			# Assemble gradients
			for inod in range(nnod):
				xfact = vol[igauss] * shapef[inod,igauss]
				gradient[nodes[inod],0] += xfact * elgrad[0]
				gradient[nodes[inod],1] += xfact * elgrad[1]
				gradient[nodes[inod],2] += xfact * elgrad[2]
	return gradient

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradVecf2D(double[:,:] xyz, double[:,:] field,object[:] elemList):
	'''
	Compute the gradient of a 2D vectorial field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,2):   positions of the nodes
		> field(nnod,2): vectorial field
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> gradient(nnod,4): gradient of vectorial field
	'''
	cdef int           MAXNODES = 100
	cdef int           igauss, ielem, inod, idim, nnod, ngauss, nelem = len(elemList), nnodtot = field.shape[0]
	cdef double        xfact
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:]   shapef
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.zeros((nnodtot,4),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,2),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,2),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] elgrad   = np.ndarray((4,),dtype=np.double)

	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
		shapef  = elem.shape
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(2):
				elxyz[inod,idim]   = xyz[nodes[inod],idim]
				elfield[inod,idim] = field[nodes[inod],idim]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			elgrad[0] = 0.
			elgrad[1] = 0.
			elgrad[2] = 0.
			elgrad[3] = 0.
			# Compute element gradients
			for inod in range(nnod):
				elgrad[0] += deri[0,inod,igauss]*elfield[inod,0] # du/dx
				elgrad[1] += deri[1,inod,igauss]*elfield[inod,0] # du/dy
				elgrad[2] += deri[0,inod,igauss]*elfield[inod,1] # dv/dx
				elgrad[3] += deri[1,inod,igauss]*elfield[inod,1] # dv/dy
			# Assemble gradients
			for inod in range(nnod):
				xfact = vol[igauss] * shapef[inod,igauss]
				gradient[nodes[inod],0] += xfact * elgrad[0]
				gradient[nodes[inod],1] += xfact * elgrad[1]
				gradient[nodes[inod],2] += xfact * elgrad[2]
				gradient[nodes[inod],3] += xfact * elgrad[3]
	return gradient

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradVecf3D(double[:,:] xyz, double[:,:] field,object[:] elemList):
	'''
	Compute the gradient of a vectorial field given a list of elements
	(internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> field(nnod,3): vectorial field
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> gradient(nnod,9): gradient of vectorial field
	'''
	cdef int           MAXNODES = 1000
	cdef int           igauss, ielem, inod, idim, nnod, ngauss, nelem = len(elemList), nnodtot = field.shape[0]
	cdef double        xfact
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:]   shapef
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.zeros((nnodtot,9),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,3),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,3),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] elgrad   = np.ndarray((9,),dtype=np.double)

	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
		shapef  = elem.shape
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(3):
				elxyz[inod,idim]   = xyz[nodes[inod],idim]
				elfield[inod,idim] = field[nodes[inod],idim]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			elgrad[0] = 0.
			elgrad[1] = 0.
			elgrad[2] = 0.
			elgrad[3] = 0.
			elgrad[4] = 0.
			elgrad[5] = 0.
			elgrad[6] = 0.
			elgrad[7] = 0.
			elgrad[8] = 0.
			# Compute element gradients
			for inod in range(nnod):
				elgrad[0] += deri[0,inod,igauss]*elfield[inod,0] # du/dx
				elgrad[1] += deri[1,inod,igauss]*elfield[inod,0] # du/dy
				elgrad[2] += deri[2,inod,igauss]*elfield[inod,0] # du/dz
				elgrad[3] += deri[0,inod,igauss]*elfield[inod,1] # dv/dx
				elgrad[4] += deri[1,inod,igauss]*elfield[inod,1] # dv/dy
				elgrad[5] += deri[2,inod,igauss]*elfield[inod,1] # dv/dz
				elgrad[6] += deri[0,inod,igauss]*elfield[inod,2] # dw/dx
				elgrad[7] += deri[1,inod,igauss]*elfield[inod,2] # dw/dy
				elgrad[8] += deri[2,inod,igauss]*elfield[inod,2] # dw/dz
			# Assemble gradients
			for inod in range(nnod):
				xfact = vol[igauss] * shapef[inod,igauss]
				gradient[nodes[inod],0] += xfact * elgrad[0]
				gradient[nodes[inod],1] += xfact * elgrad[1]
				gradient[nodes[inod],2] += xfact * elgrad[2]
				gradient[nodes[inod],3] += xfact * elgrad[3]
				gradient[nodes[inod],4] += xfact * elgrad[4]
				gradient[nodes[inod],5] += xfact * elgrad[5]
				gradient[nodes[inod],6] += xfact * elgrad[6]
				gradient[nodes[inod],7] += xfact * elgrad[7]
				gradient[nodes[inod],8] += xfact * elgrad[8]
	return gradient	

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradTenf2D(double[:,:] xyz, double[:,:] field,object[:] elemList):
	'''
	Compute the gradient of a 2D tensorial field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,2):   positions of the nodes
		> field(nnod,4): tensorial field
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> gradient(nnod,8): gradient of tensorial field
	'''
	cdef int           MAXNODES = 100
	cdef int           igauss, ielem, inod, idim, nnod, ngauss, nelem = len(elemList), nnodtot = field.shape[0]
	cdef double        xfact
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:]   shapef
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.zeros((nnodtot,8),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,3),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,4),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] elgrad   = np.ndarray((8,),dtype=np.double)

	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
		shapef  = elem.shape
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(2):
				elxyz[inod,idim]   = xyz[nodes[inod],idim]
			for idim in range(4):	
				elfield[inod,idim] = field[nodes[inod],idim]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			elgrad[0]  = 0.
			elgrad[1]  = 0.
			elgrad[2]  = 0.
			elgrad[3]  = 0.
			elgrad[4]  = 0.
			elgrad[5]  = 0.
			elgrad[6]  = 0.
			elgrad[7]  = 0.
			# Compute element gradients
			for inod in range(nnod):
				elgrad[0] += deri[0,inod,igauss]*elfield[inod,0] # da_11/dx
				elgrad[1] += deri[1,inod,igauss]*elfield[inod,0] # da_11/dy
				elgrad[2] += deri[0,inod,igauss]*elfield[inod,1] # da_12/dx
				elgrad[3] += deri[1,inod,igauss]*elfield[inod,1] # da_12/dy
				elgrad[4] += deri[0,inod,igauss]*elfield[inod,2] # da_21/dx
				elgrad[5] += deri[1,inod,igauss]*elfield[inod,2] # da_21/dy
				elgrad[6] += deri[0,inod,igauss]*elfield[inod,3] # da_22/dx
				elgrad[7] += deri[1,inod,igauss]*elfield[inod,3] # da_22/dy
			# Assemble gradients
			for inod in range(nnod):
				xfact = vol[igauss] * shapef[inod,igauss]
				gradient[nodes[inod],0]  += xfact * elgrad[0]
				gradient[nodes[inod],1]  += xfact * elgrad[1]
				gradient[nodes[inod],2]  += xfact * elgrad[2]
				gradient[nodes[inod],3]  += xfact * elgrad[3]
				gradient[nodes[inod],4]  += xfact * elgrad[4]
				gradient[nodes[inod],5]  += xfact * elgrad[5]
				gradient[nodes[inod],6]  += xfact * elgrad[6]
				gradient[nodes[inod],7]  += xfact * elgrad[7]
	return gradient

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradTenf3D(double[:,:] xyz, double[:,:] field,object[:] elemList):
	'''
	Compute the gradient of a tensorial field given a list of elements
	(internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> field(nnod,9): tensorial field
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> gradient(nnod,27): gradient of tensorial field
	'''
	cdef int           MAXNODES = 1000
	cdef int           igauss, ielem, inod, idim, nnod, ngauss, nelem = len(elemList), nnodtot = field.shape[0]
	cdef double        xfact
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:]   shapef
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.zeros((nnodtot,27),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,3),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,9),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] elgrad   = np.ndarray((27,),dtype=np.double)

	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
		shapef  = elem.shape
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(3):
				elxyz[inod,idim]   = xyz[nodes[inod],idim]
			for idim in range(9):	
				elfield[inod,idim] = field[nodes[inod],idim]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			elgrad[0]  = 0.
			elgrad[1]  = 0.
			elgrad[2]  = 0.
			elgrad[3]  = 0.
			elgrad[4]  = 0.
			elgrad[5]  = 0.
			elgrad[6]  = 0.
			elgrad[7]  = 0.
			elgrad[8]  = 0.
			elgrad[9]  = 0.
			elgrad[10] = 0.
			elgrad[11] = 0.
			elgrad[12] = 0.
			elgrad[13] = 0.
			elgrad[14] = 0.
			elgrad[15] = 0.
			elgrad[16] = 0.
			elgrad[17] = 0.
			elgrad[18] = 0.
			elgrad[19] = 0.
			elgrad[20] = 0.
			elgrad[21] = 0.
			elgrad[22] = 0.
			elgrad[23] = 0.
			elgrad[24] = 0.
			elgrad[25] = 0.
			elgrad[26] = 0.
			# Compute element gradients
			for inod in range(nnod):
				elgrad[0 ] += deri[0,inod,igauss]*elfield[inod,0] # da_11/dx
				elgrad[1 ] += deri[1,inod,igauss]*elfield[inod,0] # da_11/dy
				elgrad[2 ] += deri[2,inod,igauss]*elfield[inod,0] # da_11/dz
				elgrad[3 ] += deri[0,inod,igauss]*elfield[inod,1] # da_12/dx
				elgrad[4 ] += deri[1,inod,igauss]*elfield[inod,1] # da_12/dy
				elgrad[5 ] += deri[2,inod,igauss]*elfield[inod,1] # da_12/dz
				elgrad[6 ] += deri[0,inod,igauss]*elfield[inod,2] # da_13/dx
				elgrad[7 ] += deri[1,inod,igauss]*elfield[inod,2] # da_13/dy
				elgrad[8 ] += deri[2,inod,igauss]*elfield[inod,2] # da_13/dz
				elgrad[9 ] += deri[0,inod,igauss]*elfield[inod,3] # da_21/dx
				elgrad[10] += deri[1,inod,igauss]*elfield[inod,3] # da_21/dy
				elgrad[11] += deri[2,inod,igauss]*elfield[inod,3] # da_21/dz
				elgrad[12] += deri[0,inod,igauss]*elfield[inod,4] # da_22/dx
				elgrad[13] += deri[1,inod,igauss]*elfield[inod,4] # da_22/dy
				elgrad[14] += deri[2,inod,igauss]*elfield[inod,4] # da_22/dz
				elgrad[15] += deri[0,inod,igauss]*elfield[inod,5] # da_23/dx
				elgrad[16] += deri[1,inod,igauss]*elfield[inod,5] # da_23/dy
				elgrad[17] += deri[2,inod,igauss]*elfield[inod,5] # da_23/dz
				elgrad[18] += deri[0,inod,igauss]*elfield[inod,6] # da_31/dx
				elgrad[19] += deri[1,inod,igauss]*elfield[inod,6] # da_31/dy
				elgrad[20] += deri[2,inod,igauss]*elfield[inod,6] # da_31/dz
				elgrad[21] += deri[0,inod,igauss]*elfield[inod,7] # da_32/dx
				elgrad[22] += deri[1,inod,igauss]*elfield[inod,7] # da_32/dy
				elgrad[23] += deri[2,inod,igauss]*elfield[inod,7] # da_32/dz
				elgrad[24] += deri[0,inod,igauss]*elfield[inod,8] # da_33/dx
				elgrad[25] += deri[1,inod,igauss]*elfield[inod,8] # da_33/dy
				elgrad[26] += deri[2,inod,igauss]*elfield[inod,8] # da_33/dz
			# Assemble gradients
			for inod in range(nnod):
				xfact = vol[igauss] * shapef[inod,igauss]
				gradient[nodes[inod],0 ] += xfact * elgrad[0]
				gradient[nodes[inod],1 ] += xfact * elgrad[1]
				gradient[nodes[inod],2 ] += xfact * elgrad[2]
				gradient[nodes[inod],3 ] += xfact * elgrad[3]
				gradient[nodes[inod],4 ] += xfact * elgrad[4]
				gradient[nodes[inod],5 ] += xfact * elgrad[5]
				gradient[nodes[inod],6 ] += xfact * elgrad[6]
				gradient[nodes[inod],7 ] += xfact * elgrad[7]
				gradient[nodes[inod],8 ] += xfact * elgrad[8]
				gradient[nodes[inod],9 ] += xfact * elgrad[9]
				gradient[nodes[inod],10] += xfact * elgrad[10]
				gradient[nodes[inod],11] += xfact * elgrad[11]
				gradient[nodes[inod],12] += xfact * elgrad[12]
				gradient[nodes[inod],13] += xfact * elgrad[13]
				gradient[nodes[inod],14] += xfact * elgrad[14]
				gradient[nodes[inod],15] += xfact * elgrad[15]
				gradient[nodes[inod],16] += xfact * elgrad[16]
				gradient[nodes[inod],17] += xfact * elgrad[17]
				gradient[nodes[inod],18] += xfact * elgrad[18]
				gradient[nodes[inod],19] += xfact * elgrad[19]
				gradient[nodes[inod],20] += xfact * elgrad[20]
				gradient[nodes[inod],21] += xfact * elgrad[21]
				gradient[nodes[inod],22] += xfact * elgrad[22]
				gradient[nodes[inod],23] += xfact * elgrad[23]
				gradient[nodes[inod],24] += xfact * elgrad[24]
				gradient[nodes[inod],25] += xfact * elgrad[25]
				gradient[nodes[inod],26] += xfact * elgrad[26]
	return gradient

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradGen2D(double[:,:] xyz, double[:,:] field,object[:] elemList):
	'''
	Compute the gradient of a 2D generic field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,2):   positions of the nodes
		> field(nnod,n): field
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> gradient(nnod,2*n): gradient of field
	'''
	cdef int           MAXNODES = 100
	cdef int           igauss, ielem, inod, idim1, idim2, igrad, nnod, ngauss, \
					   nelem = len(elemList), nnodtot = field.shape[0], ndim = field.shape[1]
	cdef double        xfact
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:]   shapef
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.zeros((nnodtot,2*ndim),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,2),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,ndim),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] elgrad   = np.ndarray((2*ndim,),dtype=np.double)

	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
		shapef  = elem.shape
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim1 in range(2):
				elxyz[inod,idim1]   = xyz[nodes[inod],idim1]
			for idim1 in range(ndim):
				elfield[inod,idim1] = field[nodes[inod],idim1]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			for igrad in range(2*ndim):
				elgrad[igrad] = 0.
			# Compute element gradients
			igrad  = 0
			for idim2 in range(ndim):
				for idim1 in range(2):
					# Compute element gradients
					for inod in range(nnod):
						elgrad[igrad] += deri[idim1,inod,igauss]*elfield[inod,idim2] 
					igrad += 1	
			# Assemble gradients
			for inod in range(nnod):
				xfact = vol[igauss] * shapef[inod,igauss]
				for igrad in range(2*ndim):
					gradient[nodes[inod],igrad] += xfact * elgrad[igrad]
	return gradient

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradGen3D(double[:,:] xyz, double[:,:] field,object[:] elemList):
	'''
	Compute the gradient of a 3D generic field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> field(nnod,n): field
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> gradient(nnod,3*n): gradient of field
	'''
	cdef int           MAXNODES = 1000
	cdef int           igauss, ielem, inod, idim1, idim2, igrad, nnod, ngauss, \
					   nelem = len(elemList), nnodtot = field.shape[0], ndim = field.shape[1]
	cdef double        xfact
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:]   shapef
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.zeros((nnodtot,3*ndim),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,3),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,ndim),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] elgrad   = np.ndarray((3*ndim,),dtype=np.double)

	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
		shapef  = elem.shape
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim1 in range(3):
				elxyz[inod,idim1]   = xyz[nodes[inod],idim1]
			for idim1 in range(ndim):
				elfield[inod,idim1] = field[nodes[inod],idim1]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			for igrad in range(3*ndim):
				elgrad[igrad] = 0.
			# Compute element gradients
			igrad  = 0
			for idim2 in range(ndim):
				for idim1 in range(3):
					# Compute element gradients
					for inod in range(nnod):
						elgrad[igrad] += deri[idim1,inod,igauss]*elfield[inod,idim2] 
					igrad += 1	
			# Assemble gradients
			for inod in range(nnod):
				xfact = vol[igauss] * shapef[inod,igauss]
				for igrad in range(3*ndim):
					gradient[nodes[inod],igrad] += xfact * elgrad[igrad]
	return gradient


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
def gradient2D(double[:,:] xyz, np.ndarray field,object[:] elemList):
	'''
	Compute the gradient of a 2D scalar or vectorial 
	field given a list of elements.

	IN:
		> xyz(nnod,2):       positions of the nodes
		> field(nnod,ndim):  scalar or vectorial field
		> elemList(nel):     list of FEMlib.Element objects
	
	OUT:
		> gradient(nnod,2*ndim): gradient of field
	'''
	cdef int nnod = field.shape[0]
	cdef np.ndarray[np.double_t,ndim=2] gradient
	# Select which gradient to implement
	if len((<object> field).shape) == 1: # Scalar field
		gradient = _gradScaf2D(xyz,field,elemList)
	elif field.shape[1] == 2: # Vectorial field
		gradient = _gradVecf2D(xyz,field,elemList)
	elif field.shape[1] == 4: # Tensorial field
		gradient = _gradTenf2D(xyz,field,elemList)
	else:
		gradient = _gradGen2D(xyz,field,elemList)

	if gradient.size == 0:
		raiseError('Oops! That should never have happened')

	return gradient


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
def gradient3D(double[:,:] xyz,np.ndarray field,object[:] elemList):
	'''
	Compute the gradient of a 3D scalar or vectorial 
	field given a list of elements.

	IN:
		> xyz(nnod,3):       positions of the nodes
		> field(nnod,ndim):  scalar or vectorial field
		> elemList(nel):     list of FEMlib.Element objects

	OUT:
		> gradient(nnod,3*ndim): gradient of field
	'''
	cdef int nnod = field.shape[0]
	cdef np.ndarray[np.double_t,ndim=2] gradient
	# Select which gradient to implement
	if len((<object> field).shape) == 1: # Scalar field
		gradient = _gradScaf3D(xyz,field,elemList)
	elif field.shape[1] == 3: # Vectorial field
		gradient = _gradVecf3D(xyz,field,elemList)
	elif field.shape[1] == 9: # Tensorial field
		gradient = _gradTenf3D(xyz,field,elemList)
	else:
		gradient = _gradGen3D(xyz,field,elemList)

	if gradient.size == 0:
		raiseError('Oops! That should never have happened')

	return gradient