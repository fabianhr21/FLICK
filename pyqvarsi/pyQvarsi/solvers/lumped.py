#!/usr/bin/env python
#
# pyQvarsi, lumped.
#
# Lumped solver.
#
# Last rev: 20/09/2022
from __future__ import print_function, division

from ..cr import cr


@cr('solver.lumped')
def solver_lumped(A,b,commu=None):
	'''
	Solves the system
		b = A*x
	by
		x = b/A
	where A is the diagonal lumped matrix.

	Overwrites b.
	'''
	if len(b.shape) == 1: # Scalar field
		b /= A
		# We need to communicate with the boundaries to obtain
		# the full array
		if not commu is None: commu.communicate_scaf(b)
	else: # Vectorial, matrix folder
		for ii in range(b.shape[1]):
			b[:,ii] /= A
		# We need to communicate with the boundaries to obtain
		# the full array
		if not commu is None: commu.communicate_arrf(b)
	return b