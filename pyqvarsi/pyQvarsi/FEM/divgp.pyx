#!/usr/bin/env cython
#
# pyQvarsi, FEM div.
#
# Small FEM module to compute derivatives and possible other
# simple stuff from Alya output for postprocessing purposes.
#
# FEM divergence according to Alya.
#
# Divergece are returned at the Gauss Points instead of the nodes
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
cdef np.ndarray[np.double_t,ndim=1] _divVecf2D(double[:,:] xyz,double[:,:] field,object[:] elemList,int ngaussT):
	'''
	Compute the divergence of a 2D scalar field given a list 
	of elements (internal function).

	Assemble of the divergence is not done here.

	IN:
		> xyz(nnod,2):   positions of the nodes
		> field(nnod,2): vectorial field
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       total number of Gauss points

	OUT:
		> div(ngaussT):  divergence of scalar field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, idim, nnod, ngauss, nelem = len(elemList)
	cdef double        dudx, dvdy
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol#, mle
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] elxyz    = np.ndarray((MAXNODES,2),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,2),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] divergence = np.ndarray((ngaussT,),dtype=np.double)

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
				elfield[inod,idim] = field[nodes[inod],idim]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			dudx = 0.
			dvdy = 0.
			# Compute element gradients
			for inod in range(nnod):
				dudx += deri[0,inod,igauss]*elfield[inod,0] # du/dx
				dvdy += deri[1,inod,igauss]*elfield[inod,1] # dv/dy
			# Divergence
			divergence[igaussT] = dudx + dvdy
			igaussT += 1
	return divergence

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=1] _divVecf3D(double[:,:] xyz,double[:,:] field,object[:] elemList,int ngaussT):
	'''
	Compute the divergence of a vectorial field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> field(nnod,3): vectorial field
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       total number of Gauss points

	OUT:
		> divergence(ngaussT): divergence of vectorial field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, idim, nnod, ngauss, nelem = len(elemList)
	cdef double        dudx, dvdy, dwdz
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol#, mle
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] elxyz    = np.ndarray((MAXNODES,3),dtype=np.double), \
										elfield  = np.ndarray((MAXNODES,3),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] divergence = np.ndarray((ngaussT,),dtype=np.double)

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
			dudx = 0.
			dvdy = 0.
			dwdz = 0.
			# Compute element gradients
			for inod in range(nnod):
				dudx += deri[0,inod,igauss]*elfield[inod,0] # du/dx
				dvdy += deri[1,inod,igauss]*elfield[inod,1] # dv/dy
				dwdz += deri[2,inod,igauss]*elfield[inod,2] # dw/dz
			# Divergence
			divergence[igaussT] = dudx + dvdy + dwdz
			igaussT += 1
	return divergence

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _divTenf2D(double[:,:] xyz,double[:,:] field,object[:] elemList,int ngaussT):
	'''
	Compute the divergence of a 2D tensorial field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,2):   positions of the nodes
		> field(nnod,4): tensorial field
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       total number of Gauss points

	OUT:
		> divergence(ngaussT,2): divergence of tensorial field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, idim, nnod, ngauss, nelem = len(elemList)
	cdef double        da11dx, da21dx, da12dy, da22dy
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol#, mle
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] elxyz      = np.ndarray((MAXNODES,2),dtype=np.double), \
										elfield    = np.ndarray((MAXNODES,4),dtype=np.double), \
										divergence = np.ndarray((ngaussT,2),dtype=np.double)

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
			da11dx = 0.
			da21dx = 0.
			da12dy = 0.
			da22dy = 0.
			# Compute element gradients
			for inod in range(nnod):
				da11dx += deri[0,inod,igauss]*elfield[inod,0] # da_11/dx
				da21dx += deri[0,inod,igauss]*elfield[inod,2] # da_21/dx
				da12dy += deri[1,inod,igauss]*elfield[inod,1] # da_12/dy
				da22dy += deri[1,inod,igauss]*elfield[inod,3] # da_22/dy
			# Divergence
			divergence[igaussT,0] = da11dx + da12dy
			divergence[igaussT,1] = da21dx + da22dy
			igaussT += 1
	return divergence

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _divTenf3D(double[:,:] xyz,double[:,:] field,object[:] elemList,int ngaussT):
	'''
	Compute the divergence of a tensorial field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> field(nnod,9): tensorial field
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       total number of Gauss points

	OUT:
		> divergence(ngaussT,3): divergence of tensorial field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, idim, nnod, ngauss, nelem = len(elemList)
	cdef double        da11dx, da21dx, da31dx, da12dy, da22dy, da32dy, da13dz, da23dz, da33dz
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol#, mle
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] elxyz      = np.ndarray((MAXNODES,3),dtype=np.double), \
										elfield    = np.ndarray((MAXNODES,9),dtype=np.double), \
										divergence = np.ndarray((ngaussT,3),dtype=np.double)

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
			da11dx = 0.
			da21dx = 0.
			da31dx = 0.
			da12dy = 0.
			da22dy = 0.
			da32dy = 0.
			da13dz = 0.
			da23dz = 0.
			da33dz = 0.
			# Compute element gradients
			for inod in range(nnod):
				da11dx += deri[0,inod,igauss]*elfield[inod,0] # da_11/dx
				da21dx += deri[0,inod,igauss]*elfield[inod,3] # da_21/dx
				da31dx += deri[0,inod,igauss]*elfield[inod,6] # da_31/dx
				da12dy += deri[1,inod,igauss]*elfield[inod,1] # da_12/dy
				da22dy += deri[1,inod,igauss]*elfield[inod,4] # da_22/dy
				da32dy += deri[1,inod,igauss]*elfield[inod,7] # da_32/dy
				da13dz += deri[2,inod,igauss]*elfield[inod,2] # da_13/dz
				da23dz += deri[2,inod,igauss]*elfield[inod,5] # da_23/dz
				da33dz += deri[2,inod,igauss]*elfield[inod,8] # da_33/dz
			# Divergence
			divergence[igaussT,0] = da11dx + da12dy + da13dz
			divergence[igaussT,1] = da21dx + da22dy + da23dz
			divergence[igaussT,2] = da31dx + da32dy + da33dz
			igaussT += 1
	return divergence

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _divGen2D(double[:,:] xyz,double[:,:] field,object[:] elemList,int ngaussT):
	'''
	Compute the divergence of a 2D generic field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,2):     positions of the nodes
		> field(nnod,n):   tensorial field
		> elemList(nel,n): list of FEMlib.Element objects
		> ngaussT:         total number of Gauss points

	OUT:
		> divergence(ngaussT,n/2): divergence of generic field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, idim, ifield, idiv, nnod, ngauss, \
					   nelem = len(elemList), ndim = field.shape[1]
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol#, mle
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] divergence = np.ndarray((ngaussT,ndim//2),dtype=np.double), \
										elxyz      = np.ndarray((MAXNODES,2),dtype=np.double), \
										elfield    = np.ndarray((MAXNODES,ndim),dtype=np.double)

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
			for idim in range(ndim):
				elfield[inod,idim] = field[nodes[inod],idim]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			for idiv in range(ndim//2):
				divergence[igaussT,idiv] = 0
				for idim in range(2):
					ifield = idim + 2*idiv
					for inod in range(nnod):
						divergence[igaussT,idiv] += deri[idim,inod,igauss]*elfield[inod,ifield] 	
			igaussT += 1
	return divergence

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire functio
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] _divGen3D(double[:,:] xyz,double[:,:] field,object[:] elemList,int ngaussT):
	'''
	Compute the divergence of a 3D generic field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,3):     positions of the nodes
		> field(nnod,n):   tensorial field
		> elemList(nel,n): list of FEMlib.Element objects
		> ngaussT:         total number of Gauss points

	OUT:
		> divergence(ngaussT,n/3): divergence of generic field
	'''
	cdef int           MAXNODES = 100, igaussT = 0
	cdef int           igauss, ielem, inod, idim, ifield, idiv, nnod, ngauss, \
					   nelem = len(elemList), ndim = field.shape[1]
	cdef object        elem
	cdef int[:]        nodes
	cdef double[:]     vol#, mle
	cdef double[:,:,:] deri
	
	cdef np.ndarray[np.double_t,ndim=2] divergence = np.ndarray((ngaussT,ndim//3),dtype=np.double), \
										elxyz      = np.ndarray((MAXNODES,3),dtype=np.double), \
										elfield    = np.ndarray((MAXNODES,ndim),dtype=np.double)

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
			for idim in range(ndim):
				elfield[inod,idim] = field[nodes[inod],idim]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(ngauss):
			for idiv in range(ndim//3):
				divergence[igaussT,idiv] = 0
				for idim in range(3):
					ifield = idim + 3*idiv
					for inod in range(nnod):
						divergence[igaussT,idiv] += deri[idim,inod,igauss]*elfield[inod,ifield]
			igaussT += 1
	return divergence


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def divergence2Dgp(double[:,:] xyz, np.ndarray field,object[:] elemList,int ngaussT):
	'''
	Compute the divergene of a 2D scalar or vectorial 
	field given a list of elements.

	IN:
		> xyz(nnod,2):       positions of the nodes
		> field(nnod,ndim):  scalar or vectorial field
		> elemList(nel):     list of FEMlib.Element objects
		> ngaussT:           total number of Gauss points
	
	OUT:
		> divergence(ngaussT,ndim/2): divergence of field
	'''
	cdef np.ndarray divergence
	# Select which gradient to implement
	if len((<object> field).shape) == 1: # Scalar field
		raiseError('Divergence of scalar field not allowed!!')
	elif field.shape[1] == 2: # Vectorial field
		divergence = _divVecf2D(xyz,field,elemList,ngaussT)
	elif field.shape[1] == 4: # Tensorial field
		divergence = _divTenf2D(xyz,field,elemList,ngaussT)
	else:
		divergence = _divGen2D(xyz,field,elemList,ngaussT)

	if divergence.size == 0:
		raiseError('Oops! That should never have happened')

	return divergence

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def divergence3Dgp(double[:,:] xyz, np.ndarray field,object[:] elemList,int ngaussT):
	'''
	Compute the divergence of a 3D scalar or vectorial 
	field given a list of elements.

	IN:
		> xyz(nnod,3):       positions of the nodes
		> field(nnod,ndim):  scalar or vectorial field
		> elemList(nel):     list of FEMlib.Element objects
		> ngaussT:           total number of Gauss points

	OUT:
		> divergence(ngaussT,ndim/3): divergence of field
	'''
	cdef np.ndarray divergence
	# Select which gradient to implement
	if len((<object> field).shape) == 1: # Scalar field
		raiseError('Divergence of scalar field not allowed!!')
	elif field.shape[1] == 3: # Vectorial field
		divergence = _divVecf3D(xyz,field,elemList,ngaussT)
	elif field.shape[1] == 9: # Tensorial field
		divergence = _divTenf3D(xyz,field,elemList,ngaussT)
	else:
		divergence = _divGen3D(xyz,field,elemList,ngaussT)

	if divergence.size == 0:
		raiseError('Oops! That should never have happened')

	return divergence