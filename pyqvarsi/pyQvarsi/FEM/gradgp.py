#!/usr/bin/env python
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
# Last rev: 28/04/2022
from __future__ import print_function, division

import numpy as np

from ..utils.common import raiseError


def _gradScaf2D(xyz,field,elemList,ngaussT):
	'''
	Compute the gradient of a 2D scalar field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,2):   positions of the nodes
		> field(nnod,):  scalar field
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       total number of Gauss points

	OUT:
		> gradient(ngaussT,2): gradient of scalar field
	'''
	igaussT  = 0
	gradient = np.zeros((ngaussT,2))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			# Compute element gradients
			gradient[igaussT,0] = np.dot(deri[0,:,igauss],elfield[:]) # df/dx
			gradient[igaussT,1] = np.dot(deri[1,:,igauss],elfield[:]) # df/dy
			igaussT += 1
	return gradient

def _gradScaf3D(xyz,field,elemList,ngaussT):
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
	igaussT  = 0
	gradient = np.zeros((ngaussT,3))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			# Compute element gradients
			gradient[igaussT,0] = np.dot(deri[0,:,igauss],elfield[:]) # df/dx
			gradient[igaussT,1] = np.dot(deri[1,:,igauss],elfield[:]) # df/dy
			gradient[igaussT,2] = np.dot(deri[2,:,igauss],elfield[:]) # df/dz
			igaussT += 1
	return gradient


def _gradVecf2D(xyz,field,elemList,ngaussT):
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
	igaussT  = 0
	gradient = np.zeros((ngaussT,4))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			# Compute element gradients
			gradient[igaussT,0] = np.dot(deri[0,:,igauss],elfield[:,0]) # du/dx
			gradient[igaussT,1] = np.dot(deri[1,:,igauss],elfield[:,0]) # du/dy
			gradient[igaussT,2] = np.dot(deri[0,:,igauss],elfield[:,1]) # dv/dx
			gradient[igaussT,3] = np.dot(deri[1,:,igauss],elfield[:,1]) # dv/dy
			igaussT += 1
	return gradient

def _gradVecf3D(xyz,field,elemList,ngaussT):
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
	igaussT  = 0
	gradient = np.zeros((ngaussT,9))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			# Compute element gradients
			gradient[igaussT,0] = np.dot(deri[0,:,igauss],elfield[:,0]) # du/dx
			gradient[igaussT,1] = np.dot(deri[1,:,igauss],elfield[:,0]) # du/dy
			gradient[igaussT,2] = np.dot(deri[2,:,igauss],elfield[:,0]) # du/dz
			gradient[igaussT,3] = np.dot(deri[0,:,igauss],elfield[:,1]) # dv/dx
			gradient[igaussT,4] = np.dot(deri[1,:,igauss],elfield[:,1]) # dv/dy
			gradient[igaussT,5] = np.dot(deri[2,:,igauss],elfield[:,1]) # dv/dz
			gradient[igaussT,6] = np.dot(deri[0,:,igauss],elfield[:,2]) # dw/dx
			gradient[igaussT,7] = np.dot(deri[1,:,igauss],elfield[:,2]) # dw/dy
			gradient[igaussT,8] = np.dot(deri[2,:,igauss],elfield[:,2]) # dw/dz
			igaussT += 1
	return gradient	


def _gradTenf2D(xyz,field,elemList,ngaussT):
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
	igaussT  = 0
	gradient = np.zeros((ngaussT,8))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			# Compute element gradients
			gradient[igaussT,0] = np.dot(deri[0,:,igauss],elfield[:,0]) # da_11/dx
			gradient[igaussT,1] = np.dot(deri[1,:,igauss],elfield[:,0]) # da_11/dy
			gradient[igaussT,2] = np.dot(deri[0,:,igauss],elfield[:,1]) # da_12/dx
			gradient[igaussT,3] = np.dot(deri[1,:,igauss],elfield[:,1]) # da_12/dy
			gradient[igaussT,4] = np.dot(deri[0,:,igauss],elfield[:,2]) # da_21/dx
			gradient[igaussT,5] = np.dot(deri[1,:,igauss],elfield[:,2]) # da_21/dy
			gradient[igaussT,6] = np.dot(deri[0,:,igauss],elfield[:,3]) # da_22/dx
			gradient[igaussT,7] = np.dot(deri[1,:,igauss],elfield[:,3]) # da_22/dy
			igaussT += 1
	return gradient

def _gradTenf3D(xyz,field,elemList,ngaussT):
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
	igaussT  = 0
	gradient = np.zeros((ngaussT,27))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			# Compute element gradients
			gradient[igaussT,0 ] = np.dot(deri[0,:,igauss],elfield[:,0]) # da_11/dx
			gradient[igaussT,1 ] = np.dot(deri[1,:,igauss],elfield[:,0]) # da_11/dy
			gradient[igaussT,2 ] = np.dot(deri[2,:,igauss],elfield[:,0]) # da_11/dz
			gradient[igaussT,3 ] = np.dot(deri[0,:,igauss],elfield[:,1]) # da_12/dx
			gradient[igaussT,4 ] = np.dot(deri[1,:,igauss],elfield[:,1]) # da_12/dy
			gradient[igaussT,5 ] = np.dot(deri[2,:,igauss],elfield[:,1]) # da_12/dz
			gradient[igaussT,6 ] = np.dot(deri[0,:,igauss],elfield[:,2]) # da_13/dx
			gradient[igaussT,7 ] = np.dot(deri[1,:,igauss],elfield[:,2]) # da_13/dy
			gradient[igaussT,8 ] = np.dot(deri[2,:,igauss],elfield[:,2]) # da_13/dz
			gradient[igaussT,9 ] = np.dot(deri[0,:,igauss],elfield[:,3]) # da_21/dx
			gradient[igaussT,10] = np.dot(deri[1,:,igauss],elfield[:,3]) # da_21/dy
			gradient[igaussT,11] = np.dot(deri[2,:,igauss],elfield[:,3]) # da_21/dz
			gradient[igaussT,12] = np.dot(deri[0,:,igauss],elfield[:,4]) # da_22/dx
			gradient[igaussT,13] = np.dot(deri[1,:,igauss],elfield[:,4]) # da_22/dy
			gradient[igaussT,14] = np.dot(deri[2,:,igauss],elfield[:,4]) # da_22/dz
			gradient[igaussT,15] = np.dot(deri[0,:,igauss],elfield[:,5]) # da_23/dx
			gradient[igaussT,16] = np.dot(deri[1,:,igauss],elfield[:,5]) # da_23/dy
			gradient[igaussT,17] = np.dot(deri[2,:,igauss],elfield[:,5]) # da_23/dz
			gradient[igaussT,18] = np.dot(deri[0,:,igauss],elfield[:,6]) # da_31/dx
			gradient[igaussT,19] = np.dot(deri[1,:,igauss],elfield[:,6]) # da_31/dy
			gradient[igaussT,20] = np.dot(deri[2,:,igauss],elfield[:,6]) # da_31/dz
			gradient[igaussT,21] = np.dot(deri[0,:,igauss],elfield[:,7]) # da_32/dx
			gradient[igaussT,22] = np.dot(deri[1,:,igauss],elfield[:,7]) # da_32/dy
			gradient[igaussT,23] = np.dot(deri[2,:,igauss],elfield[:,7]) # da_32/dz
			gradient[igaussT,24] = np.dot(deri[0,:,igauss],elfield[:,8]) # da_33/dx
			gradient[igaussT,25] = np.dot(deri[1,:,igauss],elfield[:,8]) # da_33/dy
			gradient[igaussT,26] = np.dot(deri[2,:,igauss],elfield[:,8]) # da_33/dz
			igaussT += 1
	return gradient


def _gradGen2D(xyz,field,elemList,ngaussT):
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
	igaussT  = 0
	gradient = np.zeros((ngaussT,2*field.shape[1]))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			for ifield in range(field.shape[1]):
				for idim in range (2):
					# Compute element gradients
					gradient[igaussT,2*ifield+idim] = np.dot(deri[idim,:,igauss],elfield[:,ifield]) 
			igaussT += 1
	return gradient

def _gradGen3D(xyz,field,elemList,ngaussT):
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
	igaussT  = 0
	gradient = np.zeros((ngaussT,3*field.shape[1]))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			for ifield in range(field.shape[1]):
				for idim in range (3):
					# Compute element gradients
					gradient[igaussT,3*ifield+idim] = np.dot(deri[idim,:,igauss],elfield[:,ifield]) 
			igaussT += 1
	return gradient


def gradient2Dgp(xyz,field,elemList,ngaussT):
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
	gradient = np.array([])
	# Select which gradient to implement
	if len(field.shape) == 1: # Scalar field
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


def gradient3Dgp(xyz,field,elemList,ngaussT):
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
	gradient = np.array([])
	# Select which gradient to implement
	if len(field.shape) == 1: # Scalar field
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