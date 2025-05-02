#!/usr/bin/env python
#
# Example of the field.selectMask feature.
#
# Last revision: 11/04/2021
from __future__ import print_function, division

import numpy as np
import pyQvarsi


BASEDIR = '/gpfs/scratch/bsc21/bsc21742/MLwallModel/new-retau1000'
CASESTR = 'cas'
VARLIST = ['VELOC']
INSTANT = 248000


## Read mesh
mesh = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=False,read_massm=False)
# We do not need to read the communications array unless
# we need to compute gradients and similar stuff


## Read variables
field, _ = pyQvarsi.Field.read(CASESTR,VARLIST,INSTANT,mesh.xyz,basedir=BASEDIR)

# Also store global numbering and connectivity
field['LNINV'] = mesh.lninv
# We have the element connectivity here but what we really need
# is the node connectivity, so we have to build that from the mesh
# connectivity
field['CONEC'] = -np.ones((len(field),6),dtype=np.double) # Max number of neighbours

# This might not work for unstructured meshes...
for inode in range(len(mesh.xyz)):
	# In which elements is the node i?
	elems = mesh.find_node_in_elems(inode)
	# Which are the node IDs of the neighbours
	nei_list = []
	for iel in elems:
		# Position of the node in the element
		inod = np.where(inode == mesh.connectivity[iel,:])[0][0]
		# Who are my neighbours in global ids?
		nei_list.append(mesh.lninv[inod-1])
		nei_list.append(mesh.lninv[inod+1])
	# Compute the unique list
	nei_unique = np.unique(nei_list)
	# Store in CONEC
	field['CONEC'][inode,:len(nei_unique)] = nei_unique


## Crop masking by the y+ or wall normal units
# Depends on how the mesh has been created...
mask = mesh.y < 5.
field_new = field.selectMask(mask)
pyQvarsi.pprint(-1,field_new)


## Now reduce to one processor
fieldG = field_new.reduce(root=0,op=pyQvarsi.fieldFastReduce)
pyQvarsi.pprint(0,fieldG)

# Now we might want to sort according to LNINV the global
# field array since the reduce process doesn't do any sort
# of ordering
if pyQvarsi.Communicator.rankID() == 0:
	fieldG.sort(array=fieldG['LNINV'])


pyQvarsi.cr_info()