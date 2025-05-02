#!/usr/bin/env python
#
# Example of pyQvarsi outputs.
#
# In this example it is done in serial after
# a reduction to master.
#
# Last revision: 18/11/2020
from __future__ import print_function, division

import numpy as np

import pyQvarsi


BASEDIR   = './'
CASESTR   = 'cavtri03'


## Create the subdomain mesh
mesh  = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=True,read_massm=False)


## We will store LNINV as an array 
#  this is done to verify that writing the
#  output file is done correctly
field = pyQvarsi.Field(xyz=pyQvarsi.truncate(mesh.xyz,6),ARRAY=mesh.lninv)


## Create HiFiTurb writer
# We will make use of the parallel capabilities of
# the h5 io so that every node will write its own 
# part of the domain and the master will write the
# masterfile. First we need the total number of nodes
nnodG, nelG = mesh.nnodG,mesh.nelG
nnodG = mesh.nnodG

# All nodes write the database files
# parallel HDF5 requires all nodes to write the file
# otherwise a deadlock is produced
writer = pyQvarsi.io.HiFiTurbDB_Writer(nnodG)

# Write the array
writer.writeDataset( 'test', mesh.filter_bc(field['ARRAY']) )


## Master writes the output files
if (mesh.comm.rank == 0):
	# Create the structure for the master file
	writer.createGroup('01_Info',{
			'n_nodes'   : writer.createDataset('n_nodes',  (1,),'i',nnodG, ret=True),
			'n_elems'   : writer.createDataset('n_elems',  (1,),'i',nelG,  ret=True),
		},
	ret=False)

	writer.createGroup('02_Entries',{
			'Test' : writer.createExternalLink('Test','./Test.h5',ret=True),
		},
	ret=False)

	# Master writes the master file
	writer.writeMaster('Statistics.h5')

pyQvarsi.cr_info()