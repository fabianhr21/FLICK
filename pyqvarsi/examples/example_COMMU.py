#!/usr/bin/env python
#
# Example of the COMMUNICATOR class, validation
# of the communications matrix.
#
# This example uses low level functions.
#
# Last revision: 24/11/2020
from __future__ import print_function, division

import os, numpy as np

import pyQvarsi


BASEDIR   = './'
CASESTR   = 'cavtri03'


# Read the communications matrix directly from Alya output
# Create the filenames for the case
partfile  = os.path.join(BASEDIR, pyQvarsi.io.MPIO_PARTFILE_FMT  % CASESTR)
commufile = os.path.join(BASEDIR, pyQvarsi.io.MPIO_BINFILE_P_FMT % (CASESTR,'COMMU',0))
commifile = os.path.join(BASEDIR, pyQvarsi.io.MPIO_BINFILE_P_FMT % (CASESTR,'COMMI',0))
# Read the mesh files
commu, _  = pyQvarsi.io.AlyaMPIO_read(commufile,partfile)
commi, _  = pyQvarsi.io.AlyaMPIO_read(commifile,partfile)

# Create the subdomain mesh
mesh  = pyQvarsi.Mesh.read(CASESTR,basedir=BASEDIR,read_commu=True,read_codno=False,read_massm=False)
mesh.save('mesh.h5')
commu1, commi1 = mesh.comm.to_commu(mesh.nnod)

# Print
if not pyQvarsi.utils.is_rank_or_serial(0):
	pyQvarsi.pprint(mesh.comm.rank,mesh.comm)
	pyQvarsi.printArray('%d - COMMU read:'%mesh.comm.rank,commu,rank=mesh.comm.rank, precision=6)
	pyQvarsi.printArray('%d - COMMU computed:'%mesh.comm.rank,commu1,rank=mesh.comm.rank, precision=6)
	pyQvarsi.printArray('%d - COMMU diff:'%mesh.comm.rank,commu-commu1,rank=mesh.comm.rank,precision=12)
	# Average in COMMU diff must be zero while max and min might not
	# due to a different order of the vector
	pyQvarsi.printArray('%d - COMMI read:'%mesh.comm.rank,commi,rank=mesh.comm.rank, precision=6)
	pyQvarsi.printArray('%d - COMMI computed:'%mesh.comm.rank,commi1,rank=mesh.comm.rank, precision=6)
	pyQvarsi.printArray('%d - COMMI diff:'%mesh.comm.rank,commi-commi1,rank=mesh.comm.rank,precision=12)
	# Average in COMMI diff must be zero while max and min might not
	# due to a different order of the vector

pyQvarsi.cr_info()