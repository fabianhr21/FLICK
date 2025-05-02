#!/usr/bin/env python
#
# pyQvarsi, utils.
#
# Connectivity operations that can be useful for
# other modules, such as computing the cell centers.
#
# Last rev: 25/07/2022
from __future__ import print_function, division

import numpy as np

from ..cr import cr


@cr('meshing.cellCenters')
def cellCenters(xyz,conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).

	IN:
		> xyz(nnod,3):   positions of the nodes
		> conec(nel,:):  connectivity matrix

	OUT:
		> xyz_cen(nel,3): cell center position
	'''
	xyz_cen = np.zeros((conec.shape[0],xyz.shape[1]),np.double)
	for ielem in range(conec.shape[0]):
		# Get the values of the field and the positions of the element
		c = conec[ielem,conec[ielem,:]>=0]
		xyz_cen[ielem,:] = np.mean(xyz[c,:],axis=0)
	return xyz_cen