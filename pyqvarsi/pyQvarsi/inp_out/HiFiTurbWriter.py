#!/usr/bin/env python
#
# HiFiTurb Writer
#
# Last rev: 03/03/2021
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import numpy as np, h5py

from ..cr           import cr_start, cr_stop
from ..utils.common import raiseError


comm    = MPI.COMM_WORLD
rank    = comm.Get_rank()
MPIsize = comm.Get_size()


class HiFiTurbDB_Writer(object):
	'''
	Writer class for HiFiTurb dataset.
	'''
	def __init__(self,npoints,write_master=False,lninv=np.array([])):
		'''
		Class constructor
		'''
		self._npoints      = npoints
		self._write_master = write_master
		self._lninv        = lninv
		self._masterDict   = {}

	def createDataset(self,name,size,dtype,val,ret=False):
		'''
		Create a dictionary dataset for the master
		'''
		out = {
				'name' : name,
				'type' : 'dataset',
				'size' : size,
				'dtype': dtype if not dtype == 's' else h5py.special_dtype(vlen=str),
				'val'  : val		
		}
		if ret: return out
		self._masterDict[name] = out

	def createExternalLink(self,name,val,ret=False):
		'''
		Create a dictionary external link for the master
		'''
		out = {
				'name' : name,
				'type' : 'ExternalLink',
				'val'  : val		
		}
		if ret: return out
		self._masterDict[name] = out

	def createGroup(self,name,contents,ret=False):
		'''
		Create a group for the master
		'''
		out = {
			'name'    : name,
			'type'    : 'group',
			'contents': contents
		}
		if ret: return out
		self._masterDict[name] = out

	def writeMaster(self,filename):
		'''
		Write the master file
		'''
		# Create file
		file = h5py.File(filename,'w')
		# Write dictionary structure
		h5_loop_by_item(file,self._masterDict)
		# Close
		file.close()

	def writeDataset(self,name,*args):
		'''
		Store an array of 1D variables in a new dataset.
		'''
		fname = '%s.h5' % name
		nvars = len(args)
		data  = np.array(args)
		# Which mode are we using to dump the data?
		if MPIsize == 1: # Serial run
			h5_write_dataset(fname,name,self._npoints,nvars,data)
		else: # Parallel run
			if self._lninv.size == 0: # Do not order
				h5_write_dataset_mpio(fname,name,self._npoints,nvars,data,self._write_master)
			else: # Order
				h5_write_dataset_mpio_ordered(fname,name,self._npoints,nvars,data,self._lninv,self._write_master)


def h5_write_dataset(filename,dsetname,npoints,nvars,data):
	'''
	Store a list of one-dimensional arrays into
	hdf5 format given a file name and a dataset 
	name.
	'''
	cr_start('h5_write_dataset',0)
	dims = (npoints,nvars)

	file = h5py.File(filename,'w')
	dset = file.create_dataset(dsetname,dims,dtype='f')

	dset[:,:] = np.transpose(data)

	file.close()
	cr_stop('h5_write_dataset',0)

def h5_write_dataset_mpio(filename,dsetname,npoints,nvars,data,write_master=False):
	'''
	Store a list of one-dimensional arrays into
	hdf5 format given a file name and a dataset 
	name. Uses parallel mpio.
	'''
	cr_start('h5_write_dataset_mpio',0)
	dims = (npoints,nvars)

	file = h5py.File(filename,'w',driver='mpio',comm=comm)
	dset = file.create_dataset(dsetname,dims,dtype='f')

	if rank != 0 or write_master:
		# Select in which order the processors will write, do not order
		rstart = 1 if not write_master else 0
		if rank == rstart:
			istart, iend = 0, data.shape[1]
			comm.send(iend,dest=rank+1)       # send to next where to start writing
		elif rank == MPIsize-1:
			istart = comm.recv(source=rank-1) # recive from the previous where to start writing
			iend   = istart + data.shape[1]
		else:
			istart = comm.recv(source=rank-1) # recive from the previous where to start writing
			iend   = istart + data.shape[1]
			comm.send(iend,dest=rank+1)       # send to next where to start writing
	
		# Each processor writes contiguously to a space of data
		dset[istart:iend,:] = np.transpose(data)

	file.close()
	cr_stop('h5_write_dataset_mpio',0)

def h5_write_dataset_mpio_ordered(filename,dsetname,npoints,nvars,data,lninv,write_master=False):
	'''
	Store a list of one-dimensional arrays into
	hdf5 format given a file name and a dataset 
	name. Uses parallel mpio.
	'''
	cr_start('h5_write_dataset_mpio_ordered',0)
	dims = (nnodG,data.shape[0])

	file = h5py.File(filename,'w',driver='mpio',comm=comm)
	dset = file.create_dataset(dsetname,dims,dtype='f')

	if rank != 0 or write_master:
		# This is potentially slow for small datasets where reduction
		# might be faster but necessary for big datasets where reduction
		# is not possible (or too slow)
		for inod in range(lninv.shape[0]):
			dset[lninv[inod],:] = data[:,inod]

	file.close()
	cr_stop('h5_write_dataset_mpio_ordered',0)

def h5_loop_by_item(main,item):
	'''
	Recursively loop an item of the structure dictionary.
	'''
	for key in item.keys():
		# Which action to take?
		action = h5_action_by_item(main,item[key])
		# Do we need to loop a new level?
		if 'contents' in item[key]:
			h5_loop_by_item(action,item[key]['contents'])

def h5_action_by_item(main,item):
	'''
	Return an action depending on the item type.
	'''
	if item['type'].lower() == 'group':
		return main.create_group(item['name'])
	if item['type'].lower() == 'externallink':
		main[item['name']] = h5py.ExternalLink(item['val'],'/'+item['name'])
		return main
	if item['type'].lower() == 'dataset':
		dset = main.create_dataset(item['name'],item['size'],dtype=item['dtype'])
		dset[:] = item['val']
		return dset
	raiseError('Type <%s> not implemented' % item['type'].lower())