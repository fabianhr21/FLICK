#!/usr/bin/env cython
#
# pyQvarsi, FEM smooth.
#
# Small FEM module to compute derivatives and possible other
# simple stuff from Alya output for postprocessing purposes.
#
# Field smoothing computation.
#
# Last rev: 31/08/2021
from __future__ import print_function, division

import numpy as np

cimport numpy as np
cimport cython

from libc.string cimport memcpy, memset

from .lib           import LinearTriangle
from ..utils.common import raiseError


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def cellCenters(double[:,:] xyz, object[:] elemList):
	'''
	Compute the cell centers of an element given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,2):   positions of the nodes
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> xyz_cen(nel,2): cell centers
	'''
	cdef int       MAXNODES = 1000
	cdef int       ielem, inod, idim, nnod, nelem = len(elemList), ndim = xyz.shape[1]
	cdef int[:]    nodes
	cdef double[:] elxyz_cen

	cdef np.ndarray[np.double_t,ndim=2] elxyz   = np.ndarray((MAXNODES,ndim),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] xyz_cen = np.ndarray((nelem,ndim),dtype=np.double)

	for ielem in range(nelem):
		elem  = elemList[ielem]
		nnod  = elem.nnod
		nodes = elem.nodes
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(ndim):
				elxyz[inod,idim] = xyz[nodes[inod],idim]
		# Compute cell centers
		elxyz_cen = elem.centroid(elxyz)
		for idim in range(ndim):
			xyz_cen[ielem,idim] = elxyz_cen[idim]
	return xyz_cen


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=1] _nodes2Gauss_scaf(double[:] field, object[:] elemList, int ngaussT):
	'''
	Compute the position of the Gauss points given a list 
	of elements (internal function).

	SCALAR fields only

	IN:
		> field(nnod,):  field at the nodes
		> elemList(nel): list of FEMlib.Element objects
		> ngaussT:       number of Gauss points

	OUT:
		> field_gp(ngaussT,): field at the Gauss points
	'''
	cdef int       MAXNODES = 100, igaussT = 0
	cdef int       ielem, inod, idim, igauss, nnod, ngauss, nelem = len(elemList)
	cdef object    elem
	cdef int[:]    nodes
	cdef double[:] elfield_gp

	cdef np.ndarray[np.double_t,ndim=1] elfield  = np.ndarray((MAXNODES,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] field_gp = np.ndarray((ngaussT,),dtype=np.double)

	for ielem in range(nelem):
		elem   = elemList[ielem]
		nnod   = elem.nnod
		nodes  = elem.nodes
		ngauss = elem.ngauss
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			elfield[inod] = field[nodes[inod]]
		# Project to Gauss points
		elfield_gp = elem.nodes2gp(elfield)
		for igauss in range(ngauss):
			field_gp[igaussT] = elfield_gp[igauss]
			igaussT += 1
	return field_gp

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _nodes2Gauss_arrf(double[:,:] field, object[:] elemList, int ngaussT):
	'''
	Compute the position of the Gauss points given a list 
	of elements (internal function).

	ARRAY fields only

	IN:
		> field(nnod,ndim): field at the nodes
		> elemList(nel):    list of FEMlib.Element objects
		> ngaussT:          number of Gauss points

	OUT:
		> field_gp(ngaussT,ndim): field at the Gauss points
	'''
	cdef int         MAXNODES = 100, igaussT = 0
	cdef int         ielem, inod, idim, igauss, nnod, ngauss, nelem = len(elemList), ndim = field.shape[1]
	cdef object      elem
	cdef int[:]      nodes
	cdef double[:,:] elfield_gp

	cdef np.ndarray[np.double_t,ndim=2] elfield  = np.ndarray((MAXNODES,ndim),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] field_gp = np.ndarray((ngaussT,ndim),dtype=np.double)

	for ielem in range(nelem):
		elem   = elemList[ielem]
		nnod   = elem.nnod
		nodes  = elem.nodes
		ngauss = elem.ngauss
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(ndim):
				elfield[inod,idim] = field[nodes[inod],idim]
		# Project to Gauss points
		elfield_gp = elem.nodes2gp(elfield)
		for igauss in range(ngauss):
			for idim in range(ndim):
				field_gp[igaussT,idim] = elfield_gp[igauss,idim]
			igaussT += 1
	return field_gp

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def nodes2Gauss(np.ndarray field, object[:] elemList, int ngaussT):
	'''
	Compute the position of the Gauss points given a list 
	of elements (internal function).

	IN:
		> field(nnod,ndim): field at the nodes
		> elemList(nel):    list of FEMlib.Element objects
		> ngaussT:          number of Gauss points

	OUT:
		> field_gp(ngaussT*nel,ndim): field at the Gauss points
	'''
	return _nodes2Gauss_scaf(field,elemList,ngaussT) if len((<object> field).shape) == 1 else _nodes2Gauss_arrf(field,elemList,ngaussT) 


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=1] _gauss2Nodes_scaf(double[:] field_gp, double[:,:] xyz, object[:] elemList):
	'''
	Compute the position of the nodal field given a list 
	of elements (internal function).

	SCALAR fields only

	IN:
		> field_gp(ngaussT,ndim): field at the Gauss points
		> elemList(nel):          list of FEMlib.Element objects

	OUT:
		> field(nnod,ndim): field at the nodes
	'''
	cdef int         MAXNODES = 100, igaussT = 0
	cdef int         ielem, inod, idim, igauss, nnod, ngauss, nelem = len(elemList), ndim = xyz.shape[1]
	cdef double      elfield_gp
	cdef object      elem
	cdef int[:]      nodes
	cdef double[:]   vol
	cdef double[:,:] shapef

	cdef np.ndarray[np.double_t,ndim=2] elxyz      = np.ndarray((MAXNODES,ndim),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] field      = np.zeros((xyz.shape[0],),dtype=np.double)

	for ielem in range(nelem):
		elem   = elemList[ielem]
		nnod   = elem.nnod
		nodes  = elem.nodes
		ngauss = elem.ngauss
		shapef = elem.shape
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(ndim):
				elxyz[inod,idim] = xyz[nodes[inod],idim]
		# Compute element derivatives per each Gauss point
		_, vol = elem.derivative(elxyz)
		# Get the values at the element Gauss points
		for igauss in range(ngauss):
			elfield_gp = field_gp[igaussT]
			# Project to nodes
			for inod in range(nnod):
				field[nodes[inod]] += elfield_gp*vol[igauss]*shapef[inod,igauss]
			igaussT += 1
	return field

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _gauss2Nodes_arrf(double[:,:] field_gp, double[:,:] xyz, object[:] elemList):
	'''
	Compute the position of the Gauss points given a list 
	of elements (internal function).

	ARRAY fields only

	IN:
		> field(nnod,ndim): field at the nodes
		> elemList(nel):    list of FEMlib.Element objects

	OUT:
		> field_gp(ngauss*nel,ndim): field at the Gauss points
	'''
	cdef int         MAXNODES = 100, igaussT = 0
	cdef int         ielem, inod, idim, igauss, nnod, ngauss, nelem = len(elemList), narr = field_gp.shape[1], ndim = xyz.shape[1]
	cdef object      elem
	cdef int[:]      nodes
	cdef double[:]   vol
	cdef double[:,:] shapef

	cdef np.ndarray[np.double_t,ndim=2] elxyz      = np.ndarray((MAXNODES,ndim),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] elfield_gp = np.ndarray((narr,),dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=2] field      = np.zeros((xyz.shape[0],narr),dtype=np.double)

	for ielem in range(nelem):
		elem   = elemList[ielem]
		nnod   = elem.nnod
		nodes  = elem.nodes
		ngauss = elem.ngauss
		shapef = elem.shape
		# Get the values of the field and the positions of the element
		for inod in range(nnod):
			for idim in range(ndim):
				elxyz[inod,idim] = xyz[nodes[inod],idim]
		# Compute element derivatives per each Gauss point
		_, vol = elem.derivative(elxyz)
		# Get the values at the element Gauss points
		for igauss in range(ngauss):
			for idim in range(narr):
				elfield_gp[idim] = field_gp[igaussT,idim]
			# Get the values of the field and the positions of the element
			for inod in range(nnod):
				for idim in range(narr):
					field[nodes[inod],idim] += elfield_gp[idim]*vol[igauss]*shapef[inod,igauss]
			igaussT += 1 
	return field

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # turn off zero division check
def gauss2Nodes(np.ndarray field_gp, double[:,:] xyz, object[:] elemList):
	'''
	Compute the position of the nodal field given a list 
	of elements (internal function).

	IN:
		> field_gp(ngaussT,ndim): field at the Gauss points
		> elemList(nel):    list of FEMlib.Element objects

	OUT:
		> field(nnod,ndim): field at the nodes
	'''
	return _gauss2Nodes_scaf(field_gp,xyz,elemList) if len((<object> field_gp).shape) == 1 else _gauss2Nodes_arrf(field_gp,xyz,elemList) 


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def nodes_per_element(object[:] elemList):
	'''
	Get the maximum number of nodes per element
	'''
	cdef int ielem, nelem = elemList.shape[0], nnode, nelnod = 0
	cdef object e
	for ielem in range(nelem):
		e = elemList[ielem]
		nnode  = e.nnod
		nelnod = max(nelnod,nnode)
	return nelnod


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def connectivity(object[:] elemList):
	'''
	Recover the connectivity array from the element list
	'''
	cdef int inode,ielem, nnode, nelem = elemList.shape[0], nelnod = nodes_per_element(elemList)
	cdef object e
	cdef int[:] nodes
	cdef np.ndarray[np.int32_t,ndim=2] lnods = np.zeros((nelem,nelnod),np.int32)
	cdef np.ndarray[np.int32_t,ndim=1] ltype = np.zeros((nelem,),np.int32)
	for ielem in range(nelem):
		e     = elemList[ielem]
		nnode = e.nnod
		nodes = e.nodes
		ltype[ielem] = e.type
		for inode in range(nnode):
			lnods[ielem,inode] = nodes[inode]
	return lnods, ltype


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(True)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def quad2tri(object[:] elem, int ngauss=-1):
	'''
	Given a list of 2D elements (tri or tri + quad),
	it converts the quad elements to tri elements.
	'''
	cdef object e
	cdef int iel, nelem = elem.shape[0], tp
	cdef int iel_new = 0, nelem_new = 2*nelem # That only works because from 1 quad we have 2 tri
	cdef int[:] nodes

	cdef np.ndarray[np.int32_t,ndim=1] aux_nodes    = np.ndarray((3,),dtype=np.int32)
	cdef np.ndarray[object,ndim=1]     elem_all_tri = np.ndarray((nelem_new,),dtype=object)
	cdef np.ndarray[object,ndim=1]     elem_out     = np.ndarray((nelem_new,),dtype=object)

	for iel in range(nelem):
		# Recover the element
		e     = elem[iel]
		nodes = e.nodes
		tp    = e.type
		# Convert to tris
		if tp == 10: #TRI03
			elem_all_tri[iel_new] = e
			iel_new += 1
		elif tp == 12: #QUAD04
			# First triangle
			aux_nodes[0] = nodes[0]
			aux_nodes[1] = nodes[1]
			aux_nodes[2] = nodes[2]
			elem_all_tri[iel_new] = LinearTriangle(aux_nodes) if ngauss < 1 else LinearTriangle(aux_nodes,ngauss)
			# Second triangle
			aux_nodes[0] = nodes[2]
			aux_nodes[1] = nodes[3]
			aux_nodes[2] = nodes[0]
			elem_all_tri[iel_new+1] = LinearTriangle(aux_nodes) if ngauss < 1 else LinearTriangle(aux_nodes,ngauss)
			iel_new += 2
		else:
			raiseError('Element type not recognised!')
	# Crop to the number of elements
	for iel in range(iel_new):
		elem_out[iel] = elem_all_tri[iel]
	return elem_out