#!/usr/bin/env python
#
# pyQvarsi, FEM div.
#
# Small FEM module to compute derivatives and possible other
# simple stuff from Alya output for postprocessing purposes.
#
# FEM divergence according to Alya.
#
# Last rev: 16/11/2020
from __future__ import print_function, division

import numpy as np

from ..utils.common import raiseError


def _divVecf2D(xyz,field,elemList):
	'''
	Compute the divergence of a 2D scalar field given a list 
	of elements (internal function).

	Assemble of the divergence is not done here.

	IN:
		> xyz(nnod,2):   positions of the nodes
		> field(nnod,2): vectorial field
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> div(nnod):     divergence of scalar field
	'''
	divergence = np.zeros((field.shape[0],))
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
			eldiv = dudx + dvdy
			# Assemble
			xfact = vol[igauss] * elem.shape[:,igauss]
			divergence[elem.nodes] += xfact * eldiv
	return divergence

def _divVecf3D(xyz,field,elemList):
	'''
	Compute the divergence of a vectorial field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> field(nnod,3): vectorial field
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> divergence(nnod): divergence of vectorial field
	'''
	divergence = np.zeros((field.shape[0],))
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
			eldiv = dudx + dvdy + dwdz
			# Assemble
			xfact = vol[igauss] * elem.shape[:,igauss]
			divergence[elem.nodes] += xfact * eldiv
	return divergence


def _divTenf2D(xyz,field,elemList):
	'''
	Compute the divergence of a 2D tensorial field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,2):   positions of the nodes
		> field(nnod,4): tensorial field
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> divergence(nnod,2): divergence of tensorial field
	'''
	divergence = np.zeros((field.shape[0],2))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			eldiv = np.zeros((2,))
			# Compute needed element derivatives
			da11dx = np.dot(deri[0,:,igauss],elfield[:,0]) # da_11/dx
			da21dx = np.dot(deri[0,:,igauss],elfield[:,2]) # da_21/dx
			da12dy = np.dot(deri[1,:,igauss],elfield[:,1]) # da_12/dy
			da22dy = np.dot(deri[1,:,igauss],elfield[:,3]) # da_22/dy
			# Divergence
			eldiv[0] = da11dx + da12dy
			eldiv[1] = da21dx + da22dy
			# Assemble
			xfact = vol[igauss] * elem.shape[:,igauss]
			divergence[elem.nodes,0] += xfact * eldiv[0]
			divergence[elem.nodes,1] += xfact * eldiv[1]
	return divergence

def _divTenf3D(xyz,field,elemList):
	'''
	Compute the divergence of a tensorial field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> field(nnod,9): tensorial field
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> divergence(nnod,3): divergence of tensorial field
	'''
	divergence = np.zeros((field.shape[0],3))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			eldiv = np.zeros((3,))
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
			eldiv[0] = da11dx + da12dy + da13dz
			eldiv[1] = da21dx + da22dy + da23dz
			eldiv[2] = da31dx + da32dy + da33dz
			# Assemble
			xfact = vol[igauss] * elem.shape[:,igauss]
			divergence[elem.nodes,0]  += xfact * eldiv[0]
			divergence[elem.nodes,1]  += xfact * eldiv[1]
			divergence[elem.nodes,2]  += xfact * eldiv[2]
	return divergence


def _divGen2D(xyz,field,elemList):
	'''
	Compute the divergence of a 2D generic field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,2):     positions of the nodes
		> field(nnod,n):   tensorial field
		> elemList(nel,n): list of FEMlib.Element objects

	OUT:
		> divergence(nnod,n/2): divergence of generic field
	'''
	divergence = np.zeros((field.shape[0],field.shape[1]//2))
	# Open rule
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		# Compute element derivatives per each Gauss point
		deri, vol = elem.derivative(elxyz)
		# Loop the Gauss points
		for igauss in range(elem.ngauss):
			eldiv = np.zeros((field.shape[1]//2,)) # integer division
			for idiv in range(field.shape[1]//2):
				eldiv[idiv] = 0
				for idim in range (2):
					ifield = idim + 2*idiv
					eldiv[idiv] += np.dot(deri[idim,:,igauss],elfield[:,ifield]) 
			# Assemble
			xfact = vol[igauss] * elem.shape[:,igauss]
			for idiv in range(field.shape[1]//2):
				divergence[elem.nodes,idiv] += xfact * eldiv[idiv]
	return divergence

def _divGen3D(xyz,field,elemList):
	'''
	Compute the divergence of a 3D generic field given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,3):     positions of the nodes
		> field(nnod,n):   tensorial field
		> elemList(nel,n): list of FEMlib.Element objects

	OUT:
		> divergence(nnod,n/3): divergence of generic field
	'''
	divergence = np.zeros((field.shape[0],field.shape[1]//3))
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
				eldiv[idiv] = 0
				for idim in range (3):
					ifield = idim + 3*idiv
					eldiv[idiv] += np.dot(deri[idim,:,igauss],elfield[:,ifield]) 
			# Assemble
			xfact = vol[igauss] * elem.shape[:,igauss]
			for idiv in range(field.shape[1]//3):
				divergence[elem.nodes,idiv] += xfact * eldiv[idiv]
	return divergence


def divergence2D(xyz,field,elemList):
	'''
	Compute the divergene of a 2D scalar or vectorial 
	field given a list of elements.

	IN:
		> xyz(nnod,2):       positions of the nodes
		> field(nnod,ndim):  scalar or vectorial field
		> elemList(nel):     list of FEMlib.Element objects
	
	OUT:
		> divergence(nnod,ndim/2): divergence of field
	'''
	divergence = np.array([])
	# Select which divergence to implement
	if len(field.shape) == 1: # Scalar field
		raiseError('Divergence of scalar field not allowed!!')
	elif field.shape[1] == 2: # Vectorial field
		divergence = _divVecf2D(xyz,field,elemList)
	elif field.shape[1] == 4: # Tensorial field
		divergence = _divTenf2D(xyz,field,elemList)
	else:
		divergence = _divGen2D(xyz,field,elemList)

	if divergence.size == 0:
		raiseError('Oops! That should never have happened')

	return divergence


def divergence3D(xyz,field,elemList):
	'''
	Compute the divergence of a 3D scalar or vectorial 
	field given a list of elements.

	IN:
		> xyz(nnod,3):       positions of the nodes
		> field(nnod,ndim):  scalar or vectorial field
		> elemList(nel):     list of FEMlib.Element objects

	OUT:
		> divergence(nnod,ndim/3): divergence of field
	'''
	divergence = np.array([])
	# Select which divergence to implement
	if len(field.shape) == 1: # Scalar field
		raiseError('Divergence of scalar field not allowed!!')
	elif field.shape[1] == 3: # Vectorial field
		divergence = _divVecf3D(xyz,field,elemList)
	elif field.shape[1] == 9: # Tensorial field
		divergence = _divTenf3D(xyz,field,elemList)
	else:
		divergence = _divGen3D(xyz,field,elemList)

	if divergence.size == 0:
		raiseError('Oops! That should never have happened')

	return divergence