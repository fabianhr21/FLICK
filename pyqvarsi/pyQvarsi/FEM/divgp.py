#!/usr/bin/env python
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

from ..utils.common import raiseError


def _divVecf2D(xyz,field,elemList,ngaussT):
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
	igaussT = 0
	divergence = np.zeros((ngaussT,))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			# Compute needed element derivatives
			dudx = np.dot(deri[0,:,igauss],elfield[:,0]) # du/dx
			dvdy = np.dot(deri[1,:,igauss],elfield[:,1]) # dv/dy
			# Divergence
			divergence[igaussT] = dudx + dvdy
			igaussT += 1
	return divergence

def _divVecf3D(xyz,field,elemList,ngaussT):
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
	igaussT = 0
	divergence = np.zeros((ngaussT,))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			# Compute needed element derivatives
			dudx = np.dot(deri[0,:,igauss],elfield[:,0]) # du/dx
			dvdy = np.dot(deri[1,:,igauss],elfield[:,1]) # dv/dy
			dwdz = np.dot(deri[2,:,igauss],elfield[:,2]) # dw/dz
			# Divergence
			divergence[igaussT] = dudx + dvdy + dwdz
			igaussT += 1
	return divergence


def _divTenf2D(xyz,field,elemList,ngaussT):
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
	igaussT = 0
	divergence = np.zeros((ngaussT,2))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			# Compute needed element derivatives
			da11dx = np.dot(deri[0,:,igauss],elfield[:,0]) # da_11/dx
			da21dx = np.dot(deri[0,:,igauss],elfield[:,2]) # da_21/dx
			da12dy = np.dot(deri[1,:,igauss],elfield[:,1]) # da_12/dy
			da22dy = np.dot(deri[1,:,igauss],elfield[:,3]) # da_22/dy
			# Divergence
			divergence[igaussT,0] = da11dx + da12dy
			divergence[igaussT,1] = da21dx + da22dy
			igaussT += 1
	return divergence

def _divTenf3D(xyz,field,elemList,ngaussT):
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
	igaussT = 0
	divergence = np.zeros((ngaussT,3))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			# Compute needed element derivatives
			da11dx = np.dot(deri[0,:,igauss],elfield[:,0]) # da_11/dx
			da21dx = np.dot(deri[0,:,igauss],elfield[:,3]) # da_21/dx
			da31dx = np.dot(deri[0,:,igauss],elfield[:,6]) # da_31/dx
			da12dy = np.dot(deri[1,:,igauss],elfield[:,1]) # da_12/dy
			da22dy = np.dot(deri[1,:,igauss],elfield[:,4]) # da_22/dy
			da32dy = np.dot(deri[1,:,igauss],elfield[:,7]) # da_32/dy
			da13dz = np.dot(deri[2,:,igauss],elfield[:,2]) # da_13/dz
			da23dz = np.dot(deri[2,:,igauss],elfield[:,5]) # da_23/dz
			da33dz = np.dot(deri[2,:,igauss],elfield[:,8]) # da_33/dz
			# Divergence
			divergence[igaussT,0] = da11dx + da12dy + da13dz
			divergence[igaussT,1] = da21dx + da22dy + da23dz
			divergence[igaussT,2] = da31dx + da32dy + da33dz
			igaussT += 1
	return divergence


def _divGen2D(xyz,field,elemList,ngaussT):
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
	igaussT = 0
	divergence = np.zeros((ngaussT,field.shape[1]//2))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			for idiv in range(field.shape[1]//2):
				divergence[igaussT,idiv] = 0
				for idim in range (2):
					ifield = idim + 2*idiv
					divergence[igaussT,idiv] += np.dot(deri[idim,:,igauss],elfield[:,ifield]) 
			igaussT += 1
	return divergence

def _divGen3D(xyz,field,elemList,ngaussT):
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
	igaussT = 0
	divergence = np.zeros((ngaussT,field.shape[1]//3))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			eldiv = np.zeros((field.shape[1]//3,)) # integer division
			for idiv in range(field.shape[1]//3):
				divergence[igaussT,idiv] = 0
				for idim in range (3):
					ifield = idim + 3*idiv
					divergence[igaussT,idiv] += np.dot(deri[idim,:,igauss],elfield[:,ifield]) 
			igaussT += 1
	return divergence


def divergence2Dgp(xyz,field,elemList,ngaussT):
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
	divergence = np.array([])
	# Select which divergence to implement
	if len(field.shape) == 1: # Scalar field
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


def divergence3Dgp(xyz,field,elemList,ngaussT):
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
	divergence = np.array([])
	# Select which divergence to implement
	if len(field.shape) == 1: # Scalar field
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