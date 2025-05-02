#!/usr/bin/env python
#
# pyQvarsi, FEM mass.
#
# Small FEM module to compute derivatives and possible other
# simple stuff from Alya output for postprocessing purposes.
#
# Mass matrix computation.
#
# Last rev: 29/09/2020
from __future__ import print_function, division

import numpy as np

from ..   import vmath as math


def mass_matrix_lumped(xyz,elemList):
	'''
	Compute lumped diagonal mass matrix for 2D or 3D
	elements.

	IN:
		> xyz(nnod,3):   positions of the nodes
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> vmass(nnod):   lumped mass matrix (open)
	'''
	vmass = np.zeros((xyz.shape[0],),dtype=np.double)
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz = xyz[elem.nodes]
		# Compute element mass matrix
		mle   = elem.integrative(elxyz)
		# Assemble mass matrix
		vmass[elem.nodes] += np.sum(mle,axis=1) # sum on ngauss
	return vmass


def mass_matrix_consistent(xyz,elemList):
	'''
	Compute consistent mass matrix for 2D or 3D
	elements in CSR format.

	IN:
		> xyz(nnod,3):   positions of the nodes
		> elemList(nel): list of FEMlib.Element objects

	OUT:
		> cmass(nnod):   consistent mass matrix (open)
	'''
	cmass = math.dok_create(xyz.shape[0],xyz.shape[0],dtype=np.double)
	for elem in elemList:
		# Get the values of the field and the positions of the element
		elxyz = xyz[elem.nodes]
		# Compute element mass matrix
		mle   = elem.consistent(elxyz)
		# Assemble mass matrix
		for i, ei in enumerate(elem.nodes):
			for j, ej in enumerate(elem.nodes):
				cmass[ei,ej] += mle[i,j]
	return math.csr_tocsr(cmass)