#!/usr/bin/env python
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

from ..utils.common import raiseError


def _gradScaf2D(xyz,field,elemList):
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
	gradient = np.zeros((field.shape[0],2),dtype=np.double)
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			elgrad = np.zeros((2,),dtype=np.double)
			# Compute element gradients
			elgrad[0] = np.dot(deri[0,:,igauss],elfield[:]) # df/dx
			elgrad[1] = np.dot(deri[1,:,igauss],elfield[:]) # df/dy
			# Assemble gradients
			xfact = vol[igauss]*elem.shape[:,igauss]
			gradient[elem.nodes,0] += xfact * elgrad[0] 
			gradient[elem.nodes,1] += xfact * elgrad[1] 
	return gradient

def _gradScaf3D(xyz,field,elemList):
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
	gradient = np.zeros((field.shape[0],3))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			elgrad = np.zeros((3,))
			# Compute element gradients
			elgrad[0] = np.dot(deri[0,:,igauss],elfield[:]) # df/dx
			elgrad[1] = np.dot(deri[1,:,igauss],elfield[:]) # df/dy
			elgrad[2] = np.dot(deri[2,:,igauss],elfield[:]) # df/dz
			# Assemble gradients
			xfact = vol[igauss] * elem.shape[:,igauss]
			gradient[elem.nodes,0] += xfact * elgrad[0] 
			gradient[elem.nodes,1] += xfact * elgrad[1] 
			gradient[elem.nodes,2] += xfact * elgrad[2]
	return gradient


def _gradVecf2D(xyz,field,elemList):
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
	gradient = np.zeros((field.shape[0],4))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			elgrad = np.zeros((4,))
			# Compute element gradients
			elgrad[0] = np.dot(deri[0,:,igauss],elfield[:,0]) # du/dx
			elgrad[1] = np.dot(deri[1,:,igauss],elfield[:,0]) # du/dy
			elgrad[2] = np.dot(deri[0,:,igauss],elfield[:,1]) # dv/dx
			elgrad[3] = np.dot(deri[1,:,igauss],elfield[:,1]) # dv/dy
			# Assemble gradients
			xfact = vol[igauss] * elem.shape[:,igauss]
			gradient[elem.nodes,0] += xfact * elgrad[0]
			gradient[elem.nodes,1] += xfact * elgrad[1]
			gradient[elem.nodes,2] += xfact * elgrad[2]
			gradient[elem.nodes,3] += xfact * elgrad[3]
	return gradient

def _gradVecf3D(xyz,field,elemList):
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
	gradient = np.zeros((field.shape[0],9))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			elgrad = np.zeros((9,))
			# Compute element gradients
			elgrad[0] = np.dot(deri[0,:,igauss],elfield[:,0]) # du/dx
			elgrad[1] = np.dot(deri[1,:,igauss],elfield[:,0]) # du/dy
			elgrad[2] = np.dot(deri[2,:,igauss],elfield[:,0]) # du/dz
			elgrad[3] = np.dot(deri[0,:,igauss],elfield[:,1]) # dv/dx
			elgrad[4] = np.dot(deri[1,:,igauss],elfield[:,1]) # dv/dy
			elgrad[5] = np.dot(deri[2,:,igauss],elfield[:,1]) # dv/dz
			elgrad[6] = np.dot(deri[0,:,igauss],elfield[:,2]) # dw/dx
			elgrad[7] = np.dot(deri[1,:,igauss],elfield[:,2]) # dw/dy
			elgrad[8] = np.dot(deri[2,:,igauss],elfield[:,2]) # dw/dz
			# Assemble gradients
			xfact = vol[igauss] * elem.shape[:,igauss]
			gradient[elem.nodes,0] += xfact * elgrad[0]
			gradient[elem.nodes,1] += xfact * elgrad[1]
			gradient[elem.nodes,2] += xfact * elgrad[2]
			gradient[elem.nodes,3] += xfact * elgrad[3]
			gradient[elem.nodes,4] += xfact * elgrad[4]
			gradient[elem.nodes,5] += xfact * elgrad[5]
			gradient[elem.nodes,6] += xfact * elgrad[6]
			gradient[elem.nodes,7] += xfact * elgrad[7]
			gradient[elem.nodes,8] += xfact * elgrad[8]
	return gradient	


def _gradTenf2D(xyz,field,elemList):
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
	gradient = np.zeros((field.shape[0],8))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			elgrad = np.zeros((8,))
			# Compute element gradients
			elgrad[0] = np.dot(deri[0,:,igauss],elfield[:,0]) # da_11/dx
			elgrad[1] = np.dot(deri[1,:,igauss],elfield[:,0]) # da_11/dy
			elgrad[2] = np.dot(deri[0,:,igauss],elfield[:,1]) # da_12/dx
			elgrad[3] = np.dot(deri[1,:,igauss],elfield[:,1]) # da_12/dy
			elgrad[4] = np.dot(deri[0,:,igauss],elfield[:,2]) # da_21/dx
			elgrad[5] = np.dot(deri[1,:,igauss],elfield[:,2]) # da_21/dy
			elgrad[6] = np.dot(deri[0,:,igauss],elfield[:,3]) # da_22/dx
			elgrad[7] = np.dot(deri[1,:,igauss],elfield[:,3]) # da_22/dy

			# Assemble gradients
			xfact = vol[igauss] * elem.shape[:,igauss]
			gradient[elem.nodes,0] += xfact * elgrad[0]
			gradient[elem.nodes,1] += xfact * elgrad[1]
			gradient[elem.nodes,2] += xfact * elgrad[2]
			gradient[elem.nodes,3] += xfact * elgrad[3]
			gradient[elem.nodes,4] += xfact * elgrad[4]
			gradient[elem.nodes,5] += xfact * elgrad[5]
			gradient[elem.nodes,6] += xfact * elgrad[6]
			gradient[elem.nodes,7] += xfact * elgrad[7]
	return gradient

def _gradTenf3D(xyz,field,elemList):
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
	gradient = np.zeros((field.shape[0],27))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			elgrad = np.zeros((27,))
			# Compute element gradients
			elgrad[0 ] = np.dot(deri[0,:,igauss],elfield[:,0]) # da_11/dx
			elgrad[1 ] = np.dot(deri[1,:,igauss],elfield[:,0]) # da_11/dy
			elgrad[2 ] = np.dot(deri[2,:,igauss],elfield[:,0]) # da_11/dz
			elgrad[3 ] = np.dot(deri[0,:,igauss],elfield[:,1]) # da_12/dx
			elgrad[4 ] = np.dot(deri[1,:,igauss],elfield[:,1]) # da_12/dy
			elgrad[5 ] = np.dot(deri[2,:,igauss],elfield[:,1]) # da_12/dz
			elgrad[6 ] = np.dot(deri[0,:,igauss],elfield[:,2]) # da_13/dx
			elgrad[7 ] = np.dot(deri[1,:,igauss],elfield[:,2]) # da_13/dy
			elgrad[8 ] = np.dot(deri[2,:,igauss],elfield[:,2]) # da_13/dz
			elgrad[9 ] = np.dot(deri[0,:,igauss],elfield[:,3]) # da_21/dx
			elgrad[10] = np.dot(deri[1,:,igauss],elfield[:,3]) # da_21/dy
			elgrad[11] = np.dot(deri[2,:,igauss],elfield[:,3]) # da_21/dz
			elgrad[12] = np.dot(deri[0,:,igauss],elfield[:,4]) # da_22/dx
			elgrad[13] = np.dot(deri[1,:,igauss],elfield[:,4]) # da_22/dy
			elgrad[14] = np.dot(deri[2,:,igauss],elfield[:,4]) # da_22/dz
			elgrad[15] = np.dot(deri[0,:,igauss],elfield[:,5]) # da_23/dx
			elgrad[16] = np.dot(deri[1,:,igauss],elfield[:,5]) # da_23/dy
			elgrad[17] = np.dot(deri[2,:,igauss],elfield[:,5]) # da_23/dz
			elgrad[18] = np.dot(deri[0,:,igauss],elfield[:,6]) # da_31/dx
			elgrad[19] = np.dot(deri[1,:,igauss],elfield[:,6]) # da_31/dy
			elgrad[20] = np.dot(deri[2,:,igauss],elfield[:,6]) # da_31/dz
			elgrad[21] = np.dot(deri[0,:,igauss],elfield[:,7]) # da_32/dx
			elgrad[22] = np.dot(deri[1,:,igauss],elfield[:,7]) # da_32/dy
			elgrad[23] = np.dot(deri[2,:,igauss],elfield[:,7]) # da_32/dz
			elgrad[24] = np.dot(deri[0,:,igauss],elfield[:,8]) # da_33/dx
			elgrad[25] = np.dot(deri[1,:,igauss],elfield[:,8]) # da_33/dy
			elgrad[26] = np.dot(deri[2,:,igauss],elfield[:,8]) # da_33/dz
			# Assemble gradients
			xfact = vol[igauss] * elem.shape[:,igauss]
			gradient[elem.nodes,0 ] += xfact * elgrad[0]
			gradient[elem.nodes,1 ] += xfact * elgrad[1]
			gradient[elem.nodes,2 ] += xfact * elgrad[2]
			gradient[elem.nodes,3 ] += xfact * elgrad[3]
			gradient[elem.nodes,4 ] += xfact * elgrad[4]
			gradient[elem.nodes,5 ] += xfact * elgrad[5]
			gradient[elem.nodes,6 ] += xfact * elgrad[6]
			gradient[elem.nodes,7 ] += xfact * elgrad[7]
			gradient[elem.nodes,8 ] += xfact * elgrad[8]
			gradient[elem.nodes,9 ] += xfact * elgrad[9]
			gradient[elem.nodes,10] += xfact * elgrad[10]
			gradient[elem.nodes,11] += xfact * elgrad[11]
			gradient[elem.nodes,12] += xfact * elgrad[12]
			gradient[elem.nodes,13] += xfact * elgrad[13]
			gradient[elem.nodes,14] += xfact * elgrad[14]
			gradient[elem.nodes,15] += xfact * elgrad[15]
			gradient[elem.nodes,16] += xfact * elgrad[16]
			gradient[elem.nodes,17] += xfact * elgrad[17]
			gradient[elem.nodes,18] += xfact * elgrad[18]
			gradient[elem.nodes,19] += xfact * elgrad[19]
			gradient[elem.nodes,20] += xfact * elgrad[20]
			gradient[elem.nodes,21] += xfact * elgrad[21]
			gradient[elem.nodes,22] += xfact * elgrad[22]
			gradient[elem.nodes,23] += xfact * elgrad[23]
			gradient[elem.nodes,24] += xfact * elgrad[24]
			gradient[elem.nodes,25] += xfact * elgrad[25]
			gradient[elem.nodes,26] += xfact * elgrad[26]
	return gradient


def _gradGen2D(xyz,field,elemList):
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
	gradient = np.zeros((field.shape[0],2*field.shape[1]))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			elgrad = np.zeros((2*field.shape[1],))
			igrad  = 0
			for ifield in range(field.shape[1]):
				for idim in range (2):
					# Compute element gradients
					elgrad[igrad] = np.dot(deri[idim,:,igauss],elfield[:,ifield]) 
					igrad += 1

			# Assemble gradients
			xfact = vol[igauss] * elem.shape[:,igauss]
			for igrad in range(2*field.shape[1]):
				gradient[elem.nodes,igrad] += xfact * elgrad[igrad]
	return gradient

def _gradGen3D(xyz,field,elemList):
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
	gradient = np.zeros((field.shape[0],3*field.shape[1]))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			elgrad = np.zeros((3*field.shape[1],))
			igrad  = 0
			for ifield in range(field.shape[1]):
				for idim in range (3):
					# Compute element gradients
					elgrad[igrad] = np.dot(deri[idim,:,igauss],elfield[:,ifield]) 
					igrad += 1

			# Assemble gradients
			xfact = vol[igauss] * elem.shape[:,igauss]
			for igrad in range(3*field.shape[1]):
				gradient[elem.nodes,igrad] += xfact * elgrad[igrad]
	return gradient


def gradient2D(xyz,field,elemList):
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
	gradient = np.array([])
	# Select which gradient to implement
	if len(field.shape) == 1: # Scalar field
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


def gradient3D(xyz,field,elemList):
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
	gradient = np.array([])
	# Select which gradient to implement
	if len(field.shape) == 1: # Scalar field
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