#!/usr/bin/env python
#
# pyQvarsi, FEM smooth.
#
# Small FEM module to compute derivatives and possible other
# simple stuff from Alya output for postprocessing purposes.
#
# Utilities.
#
# Last rev: 12/11/2021
from __future__ import print_function, division

import numpy as np

from .lib           import LinearTriangle
from ..utils.common import raiseError


def cellCenters(xyz,elemList):
	'''
	Compute the cell centers given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> xyz_cen(nel,3): cell center position
	'''
	xyz_cen = np.zeros((len(elemList),xyz.shape[1]),np.double)
	# Open rule
	for ielem,elem in enumerate(elemList):
		# Get the values of the field and the positions of the element
		elxyz = xyz[elem.nodes]
		xyz_cen[ielem,:] = elem.centroid(elxyz)
	return xyz_cen


def nodes2Gauss(field,elemList,ngaussT):
	'''
	Compute the position of the Gauss points given a list 
	of elements (internal function).

	IN:
		> field(nnod,ndim): field at the nodes
		> elemList(nel):    list of FEMlib.Element objects
		> ngaussT:          number of Gauss points

	OUT:
		> field_gp(ngaussT,ndim): field at the Gauss points
	'''
	igaussT  = 0
	field_gp = np.zeros((ngaussT,field.shape[1]) if len(field.shape) > 1 else (ngaussT,),np.double)
	# Open rule
	for ielem,elem in enumerate(elemList):
		# Get the values of the field and the positions of the element
		elfield = field[elem.nodes]
		field_gp[igaussT:elem.ngauss+igaussT] = elem.nodes2gp(elfield)
		igaussT += elem.ngauss
	return field_gp


def gauss2Nodes(field_gp,xyz,elemList):
	'''
	Compute the position of the nodal field given a list 
	of elements (internal function).

	Errors in approximation seem to be due to the lumped mass
	matrix, especially on the boundaries.

	IN:
		> field_gp(ngaussT,ndim): field at the Gauss points
		> xyz(nnod,3):      positions of the nodes
		> elemList(nel):    list of FEMlib.Element objects

	OUT:
		> field(nnod,ndim): field at the nodes
	'''
	igaussT = 0
	if len(field_gp.shape) > 1:
		field = np.zeros((xyz.shape[0],field_gp.shape[1]),np.double)
		# Open rule
		for ielem,elem in enumerate(elemList):
			elxyz  = xyz[elem.nodes]
			_, vol = elem.derivative(elxyz)
			# Get the values of the field and the positions of the element
			elfield_gp = field_gp[igaussT:elem.ngauss+igaussT]
			for igauss in range(elem.ngauss):
				# Assemble
				xfact = vol[igauss]*elem.shape[:,igauss] # (nnod,)
				for idim in range(field_gp.shape[1]):
					field[elem.nodes,idim] += xfact*elfield_gp[igauss,idim] # (nnod,)*(nnod,ndim)
				igaussT += 1
	else:
		field = np.zeros((xyz.shape[0],),np.double)
		# Open rule
		for ielem,elem in enumerate(elemList):
			elxyz  = xyz[elem.nodes]
			_, vol = elem.derivative(elxyz)
			# Get the values of the field and the positions of the element
			elfield_gp = field_gp[igaussT:elem.ngauss+igaussT]
			for igauss in range(elem.ngauss):
				# Assemble
				xfact = vol[igauss]*elem.shape[:,igauss] # (nnod,)
				field[elem.nodes] += xfact*elfield_gp[igauss] # (nnod,)*(nnod,)
				igaussT += 1
	return field


def nodes_per_element(elemList):
	'''
	Get the maximum number of nodes per element
	'''
	return np.max([len(e) for e in elemList])


def connectivity(elemList):
	'''
	Recover the connectivity array from the element list
	'''
	nelnod = nodes_per_element(elemList)
	lnods  = -np.ones((len(elemList),nelnod),np.int32) 
	ltype  = np.zeros((len(elemList),),np.int32)
	for i,e in enumerate(elemList):
		lnods[i,:len(e)] = e.nodes
		ltype[i,]        = e.type
	return lnods, ltype


def quad2tri(elem, ngauss=-1):
	'''
	Given a list of 2D elements (tri or tri + quad),
	it converts the quad elements to tri elements.
	'''
	elem_all_tri = []
	for i, e in enumerate(elem):
		if e.type == 10: #TRI03
			elem_all_tri.append(e)
		elif e.type == 12: #QUAD04 (split in triangles)
			tri1 = [e.nodes[0], e.nodes[1], e.nodes[2]]
			tri2 = [e.nodes[2], e.nodes[3], e.nodes[0]]
			elem_all_tri.append(LinearTriangle(tri1) if ngauss < 0 else LinearTriangle(tri1,ngauss))
			elem_all_tri.append(LinearTriangle(tri2) if ngauss < 0 else LinearTriangle(tri2,ngauss))
		else:
			raiseError('Element type not recognised.')
	return np.array(elem_all_tri, dtype=object)