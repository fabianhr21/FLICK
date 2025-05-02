#!/usr/bin/env cython
#
# pyQvarsi, FEM grad.
#
# Small FEM module to compute derivatives and possible other
# simple stuff from Alya output for postprocessing purposes.
#
# FEM gradient according to Alya.
#
# Gradients are returned at the Gauss Points instead of the nodes
#
# Last rev: 29/04/2022
from __future__ import print_function, division

import numpy as np

cimport numpy as np
cimport cython

from ..utils.common import raiseError


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradScaf2D(double[:,:] xyz,double[:] field,object[:] elemList,int ngaussT):
	'''
	Compute the gradient of a 2D scalar field given a list 
	of elements (internal function).

	Assemble of the gradients is not done here.

	IN:
		> xyz(nnod,2):   positions of the nodes
		> field(nnod,):  scalar field
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       total number of Gauss points

	OUT:
		> gradient(ngaussT,2): gradient of scalar field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, nnod, ngauss, nelem = len(elemList)
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.ndarray((ngaussT,2),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,2),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] elfield  = np.ndarray((MAXNODES,) ,dtype=np.double)
	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(2):
				elxyz[inod,idim] = xyz[nodes[inod],idim]
			elfield[inod] = field[nodes[inod]]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			gradient[igaussT,0] = 0.
			gradient[igaussT,1] = 0.
			# Compute element gradients
			for inod in range(nnod):
				gradient[igaussT,0] += deri[0,inod,igauss]*elfield[inod] # df/dx
				gradient[igaussT,1] += deri[1,inod,igauss]*elfield[inod] # df/dy
			igaussT += 1
	return gradient

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradScaf3D(double[:,:] xyz, double[:] field,object[:] elemList,int ngaussT):
	'''
	Compute the gradient of a 3D scalar field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> field(nnod,):  scalar field
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       total number of Gauss points

	OUT:
		> gradient(ngaussT,3): gradient of scalar field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, nnod, ngauss, nelem = len(elemList)
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.ndarray((ngaussT,3),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,3),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] elfield  = np.ndarray((MAXNODES,) ,dtype=np.double)
	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(3):
				elxyz[inod,idim] = xyz[nodes[inod],idim]
			elfield[inod] = field[nodes[inod]]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			gradient[igaussT,0] = 0.
			gradient[igaussT,1] = 0.
			gradient[igaussT,2] = 0.
			# Compute element gradients
			for inod in range(nnod):
				gradient[igaussT,0] += deri[0,inod,igauss]*elfield[inod] # df/dx
				gradient[igaussT,1] += deri[1,inod,igauss]*elfield[inod] # df/dy
				gradient[igaussT,2] += deri[2,inod,igauss]*elfield[inod] # df/dz
			igaussT += 1
	return gradient

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradVecf2D(double[:,:] xyz, double[:,:] field,object[:] elemList,int ngaussT):
	'''
	Compute the gradient of a 2D vectorial field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,2):   positions of the nodes
		> field(nnod,2): vectorial field
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       total number of Gauss points

	OUT:
		> gradient(ngaussT,4): gradient of vectorial field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, idim, nnod, ngauss, nelem = len(elemList)
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.ndarray((ngaussT,4),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,2),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,2),dtype=np.double)
	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
		for inod in range(nnod):
			for idim in range(2):
				elxyz[inod,idim]   = xyz[nodes[inod],idim]
				elfield[inod,idim] = field[nodes[inod],idim]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			gradient[igaussT,0] = 0.
			gradient[igaussT,1] = 0.
			gradient[igaussT,2] = 0.
			gradient[igaussT,3] = 0.
			# Compute element gradients
			for inod in range(nnod):
				gradient[igaussT,0] += deri[0,inod,igauss]*elfield[inod,0] # du/dx
				gradient[igaussT,1] += deri[1,inod,igauss]*elfield[inod,0] # du/dy
				gradient[igaussT,2] += deri[0,inod,igauss]*elfield[inod,1] # dv/dx
				gradient[igaussT,3] += deri[1,inod,igauss]*elfield[inod,1] # dv/dy
			igaussT += 1
	return gradient

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradVecf3D(double[:,:] xyz, double[:,:] field,object[:] elemList,int ngaussT):
	'''
	Compute the gradient of a vectorial field given a list of elements
	(internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> field(nnod,3): vectorial field
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       total number of Gauss points

	OUT:
		> gradient(ngaussT,9): gradient of vectorial field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, idim, nnod, ngauss, nelem = len(elemList)
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.ndarray((ngaussT,9),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,3),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,3),dtype=np.double)
	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(3):
				elxyz[inod,idim]   = xyz[nodes[inod],idim]
				elfield[inod,idim] = field[nodes[inod],idim]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			gradient[igaussT,0] = 0.
			gradient[igaussT,1] = 0.
			gradient[igaussT,2] = 0.
			gradient[igaussT,3] = 0.
			gradient[igaussT,4] = 0.
			gradient[igaussT,5] = 0.
			gradient[igaussT,6] = 0.
			gradient[igaussT,7] = 0.
			gradient[igaussT,8] = 0.
			# Compute element gradients
			for inod in range(nnod):
				gradient[igaussT,0] += deri[0,inod,igauss]*elfield[inod,0] # du/dx
				gradient[igaussT,1] += deri[1,inod,igauss]*elfield[inod,0] # du/dy
				gradient[igaussT,2] += deri[2,inod,igauss]*elfield[inod,0] # du/dz
				gradient[igaussT,3] += deri[0,inod,igauss]*elfield[inod,1] # dv/dx
				gradient[igaussT,4] += deri[1,inod,igauss]*elfield[inod,1] # dv/dy
				gradient[igaussT,5] += deri[2,inod,igauss]*elfield[inod,1] # dv/dz
				gradient[igaussT,6] += deri[0,inod,igauss]*elfield[inod,2] # dw/dx
				gradient[igaussT,7] += deri[1,inod,igauss]*elfield[inod,2] # dw/dy
				gradient[igaussT,8] += deri[2,inod,igauss]*elfield[inod,2] # dw/dz
			igaussT += 1
	return gradient	

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradTenf2D(double[:,:] xyz, double[:,:] field,object[:] elemList,int ngaussT):
	'''
	Compute the gradient of a 2D tensorial field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,2):   positions of the nodes
		> field(nnod,4): tensorial field
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       total number of Gauss points

	OUT:
		> gradient(ngaussT,8): gradient of tensorial field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, idim, nnod, ngauss, nelem = len(elemList), nnodtot = field.shape[0]
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.ndarray((ngaussT,8),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,3),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,4),dtype=np.double)
	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
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
			gradient[igaussT,0]  = 0.
			gradient[igaussT,1]  = 0.
			gradient[igaussT,2]  = 0.
			gradient[igaussT,3]  = 0.
			gradient[igaussT,4]  = 0.
			gradient[igaussT,5]  = 0.
			gradient[igaussT,6]  = 0.
			gradient[igaussT,7]  = 0.
			# Compute element gradients
			for inod in range(nnod):
				gradient[igaussT,0] += deri[0,inod,igauss]*elfield[inod,0] # da_11/dx
				gradient[igaussT,1] += deri[1,inod,igauss]*elfield[inod,0] # da_11/dy
				gradient[igaussT,2] += deri[0,inod,igauss]*elfield[inod,1] # da_12/dx
				gradient[igaussT,3] += deri[1,inod,igauss]*elfield[inod,1] # da_12/dy
				gradient[igaussT,4] += deri[0,inod,igauss]*elfield[inod,2] # da_21/dx
				gradient[igaussT,5] += deri[1,inod,igauss]*elfield[inod,2] # da_21/dy
				gradient[igaussT,6] += deri[0,inod,igauss]*elfield[inod,3] # da_22/dx
				gradient[igaussT,7] += deri[1,inod,igauss]*elfield[inod,3] # da_22/dy
			igaussT += 1
	return gradient

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradTenf3D(double[:,:] xyz, double[:,:] field,object[:] elemList,int ngaussT):
	'''
	Compute the gradient of a tensorial field given a list of elements
	(internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> field(nnod,9): tensorial field
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       total number of Gauss points

	OUT:
		> gradient(ngaussT,27): gradient of tensorial field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, idim, nnod, ngauss, nelem = len(elemList)
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.ndarray((ngaussT,27),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,3),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,9),dtype=np.double)
	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
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
			gradient[igaussT,0]  = 0.
			gradient[igaussT,1]  = 0.
			gradient[igaussT,2]  = 0.
			gradient[igaussT,3]  = 0.
			gradient[igaussT,4]  = 0.
			gradient[igaussT,5]  = 0.
			gradient[igaussT,6]  = 0.
			gradient[igaussT,7]  = 0.
			gradient[igaussT,8]  = 0.
			gradient[igaussT,9]  = 0.
			gradient[igaussT,10] = 0.
			gradient[igaussT,11] = 0.
			gradient[igaussT,12] = 0.
			gradient[igaussT,13] = 0.
			gradient[igaussT,14] = 0.
			gradient[igaussT,15] = 0.
			gradient[igaussT,16] = 0.
			gradient[igaussT,17] = 0.
			gradient[igaussT,18] = 0.
			gradient[igaussT,19] = 0.
			gradient[igaussT,20] = 0.
			gradient[igaussT,21] = 0.
			gradient[igaussT,22] = 0.
			gradient[igaussT,23] = 0.
			gradient[igaussT,24] = 0.
			gradient[igaussT,25] = 0.
			gradient[igaussT,26] = 0.
			# Compute element gradients
			for inod in range(nnod):
				gradient[igaussT,0 ] += deri[0,inod,igauss]*elfield[inod,0] # da_11/dx
				gradient[igaussT,1 ] += deri[1,inod,igauss]*elfield[inod,0] # da_11/dy
				gradient[igaussT,2 ] += deri[2,inod,igauss]*elfield[inod,0] # da_11/dz
				gradient[igaussT,3 ] += deri[0,inod,igauss]*elfield[inod,1] # da_12/dx
				gradient[igaussT,4 ] += deri[1,inod,igauss]*elfield[inod,1] # da_12/dy
				gradient[igaussT,5 ] += deri[2,inod,igauss]*elfield[inod,1] # da_12/dz
				gradient[igaussT,6 ] += deri[0,inod,igauss]*elfield[inod,2] # da_13/dx
				gradient[igaussT,7 ] += deri[1,inod,igauss]*elfield[inod,2] # da_13/dy
				gradient[igaussT,8 ] += deri[2,inod,igauss]*elfield[inod,2] # da_13/dz
				gradient[igaussT,9 ] += deri[0,inod,igauss]*elfield[inod,3] # da_21/dx
				gradient[igaussT,10] += deri[1,inod,igauss]*elfield[inod,3] # da_21/dy
				gradient[igaussT,11] += deri[2,inod,igauss]*elfield[inod,3] # da_21/dz
				gradient[igaussT,12] += deri[0,inod,igauss]*elfield[inod,4] # da_22/dx
				gradient[igaussT,13] += deri[1,inod,igauss]*elfield[inod,4] # da_22/dy
				gradient[igaussT,14] += deri[2,inod,igauss]*elfield[inod,4] # da_22/dz
				gradient[igaussT,15] += deri[0,inod,igauss]*elfield[inod,5] # da_23/dx
				gradient[igaussT,16] += deri[1,inod,igauss]*elfield[inod,5] # da_23/dy
				gradient[igaussT,17] += deri[2,inod,igauss]*elfield[inod,5] # da_23/dz
				gradient[igaussT,18] += deri[0,inod,igauss]*elfield[inod,6] # da_31/dx
				gradient[igaussT,19] += deri[1,inod,igauss]*elfield[inod,6] # da_31/dy
				gradient[igaussT,20] += deri[2,inod,igauss]*elfield[inod,6] # da_31/dz
				gradient[igaussT,21] += deri[0,inod,igauss]*elfield[inod,7] # da_32/dx
				gradient[igaussT,22] += deri[1,inod,igauss]*elfield[inod,7] # da_32/dy
				gradient[igaussT,23] += deri[2,inod,igauss]*elfield[inod,7] # da_32/dz
				gradient[igaussT,24] += deri[0,inod,igauss]*elfield[inod,8] # da_33/dx
				gradient[igaussT,25] += deri[1,inod,igauss]*elfield[inod,8] # da_33/dy
				gradient[igaussT,26] += deri[2,inod,igauss]*elfield[inod,8] # da_33/dz
			igaussT += 1
	return gradient

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradGen2D(double[:,:] xyz, double[:,:] field,object[:] elemList,int ngaussT):
	'''
	Compute the gradient of a 2D generic field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,2):   positions of the nodes
		> field(nnod,n): field
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       total number of Gauss points

	OUT:
		> gradient(ngaussT,2*n): gradient of field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, idim1, idim2, igrad, nnod, ngauss, \
					   nelem = len(elemList), ndim = field.shape[1]
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.ndarray((ngaussT,2*ndim),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,2),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,ndim),dtype=np.double)
	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
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
				gradient[igaussT,igrad] = 0.
			# Compute element gradients
			igrad  = 0
			for idim2 in range(ndim):
				for idim1 in range(2):
					# Compute element gradients
					for inod in range(nnod):
						gradient[igaussT,igrad] += deri[idim1,inod,igauss]*elfield[inod,idim2] 
					igrad += 1	
			igaussT += 1
	return gradient

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _gradGen3D(double[:,:] xyz, double[:,:] field,object[:] elemList,int ngaussT):
	'''
	Compute the gradient of a 3D generic field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> field(nnod,n): field
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       total number of Gauss points

	OUT:
		> gradient(ngaussT,3*n): gradient of field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, idim1, idim2, igrad, nnod, ngauss, \
					   nelem = len(elemList), ndim = field.shape[1]
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] gradient = np.ndarray((ngaussT,3*ndim),dtype=np.double), \
										elxyz    = np.ndarray((MAXNODES,3),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,ndim),dtype=np.double)
	# Open rule
	for ielem in range(nelem):
		elem    = elemList[ielem]
		ngauss  = elem.ngauss
		nnod    = elem.nnod
		nodes   = elem.nodes
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
				gradient[igaussT,igrad] = 0.
			# Compute element gradients
			igrad  = 0
			for idim2 in range(ndim):
				for idim1 in range(3):
					# Compute element gradients
					for inod in range(nnod):
						gradient[igaussT,igrad] += deri[idim1,inod,igauss]*elfield[inod,idim2] 
					igrad += 1	
			igaussT += 1
	return gradient


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
def gradient2Dgp(double[:,:] xyz, np.ndarray field,object[:] elemList,int ngaussT):
	'''
	Compute the gradient of a 2D scalar or vectorial 
	field given a list of elements.

	IN:
		> xyz(nnod,2):       positions of the nodes
		> field(nnod,ndim):  scalar or vectorial field
		> elemList(nel):     list of FEMlib.Element objects
		> ngaussT:           total number of Gauss points
	
	OUT:
		> gradient(ngaussT,2*ndim): gradient of field
	'''
	cdef int nnod = field.shape[0]
	cdef np.ndarray[np.double_t,ndim=2] gradient
	# Select which gradient to implement
	if len((<object> field).shape) == 1: # Scalar field
		gradient = _gradScaf2D(xyz,field,elemList,ngaussT)
	elif field.shape[1] == 2: # Vectorial field
		gradient = _gradVecf2D(xyz,field,elemList,ngaussT)
	elif field.shape[1] == 4: # Tensorial field
		gradient = _gradTenf2D(xyz,field,elemList,ngaussT)
	else:
		gradient = _gradGen2D(xyz,field,elemList,ngaussT)

	if gradient.size == 0:
		raiseError('Oops! That should never have happened')
	return gradient

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
def gradient3Dgp(double[:,:] xyz,np.ndarray field,object[:] elemList,int ngaussT):
	'''
	Compute the gradient of a 3D scalar or vectorial 
	field given a list of elements.

	IN:
		> xyz(nnod,3):       positions of the nodes
		> field(nnod,ndim):  scalar or vectorial field
		> elemList(nel):     list of FEMlib.Element objects
		> ngaussT:           total number of Gauss points

	OUT:
		> gradient(ngaussT,3*ndim): gradient of field
	'''
	cdef int nnod = field.shape[0]
	cdef np.ndarray[np.double_t,ndim=2] gradient
	# Select which gradient to implement
	if len((<object> field).shape) == 1: # Scalar field
		gradient = _gradScaf3D(xyz,field,elemList,ngaussT)
	elif field.shape[1] == 3: # Vectorial field
		gradient = _gradVecf3D(xyz,field,elemList,ngaussT)
	elif field.shape[1] == 9: # Tensorial field
		gradient = _gradTenf3D(xyz,field,elemList,ngaussT)
	else:
		gradient = _gradGen3D(xyz,field,elemList,ngaussT)

	if gradient.size == 0:
		raiseError('Oops! That should never have happened')
	return gradient