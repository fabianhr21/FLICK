#!/usr/bin/env python
#
# pyQvarsi, FEM integral.
#
# Small FEM module to compute integrals from Alya 
# output for postprocessing purposes.
#
# Last rev: 7/04/2021
from __future__ import print_function, division

import numpy as np

from ..cr             import cr_start, cr_stop
from ..utils.common   import raiseError
from ..utils.parallel import mpi_reduce


def integralSurface(xyz, field, mask, elemList):
	'''
	Compute the surface integral of a 3D scalar 
	field given a list of elements.

	IN:
		> xyz(nnod,3):       positions of the nodes
		> field(nnod):       scalar field
		> mask(nnod):        masking field to eliminate nodes
							 from the integral
		> elemList(nel):     list of FEMlib.Element objects

	OUT:
		> integral:          value of the integral on the given area
	'''
	cr_start('integSurface',0)
	integral = 0
	for elem in elemList:
		# Get the values of the field, mask and positions of the element
		elxyz   = xyz[elem.nodes]
		elfield = field[elem.nodes]
		elmask  = mask[elem.nodes]
		# Only compute the integral if the nodes of the element are not masked
		if np.all(elmask):
			elint     = elem.integrative(elxyz)
			# Accumulate integral
			integral += np.sum(np.transpose(elint)*elfield)
	cr_stop('integSurface',0)
	return integral


def integralVolume(xyz, field, mask, elemList):
	'''
	Compute the volume integral of a 3D scalar 
	field given a list of elements.

	IN:
		> xyz(nnod,3):       positions of the nodes
		> field(nnod):       scalar field
		> mask(nnod):        masking field to eliminate nodes
							 from the integral
		> elemList(nel):     list of FEMlib.Element objects

	OUT:
		> integral:          value of the integral on the given volume
	'''
	cr_start('integVolume',0)
	integral = 0
	for elem in elemList:
		# Get the values of the field, mask and positions of the element
		elxyz = xyz[elem.nodes]
		elfield = field[elem.nodes]
		elmask = mask[elem.nodes]
		# Only compute the integral if the nodes of the element are not masked
		if np.all(elmask):
			elint = elem.integrative(elxyz)
			integral += np.sum(np.transpose(elint)*elfield)
	cr_stop('integVolume',0)
	return integral

