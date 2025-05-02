#!/usr/bin/env python
#
# pyQvarsi, partition table.
#
# PartitionTable class, to manage how a case is partitioned.
#
# Last rev: 01/02/2023
from __future__ import print_function, division

import os, numpy as np

from .               import inp_out as io
from .cr             import cr
from .mem            import mem
from .utils.parallel import MPI_RANK, MPI_SIZE, worksplit, is_rank_or_serial, mpi_gather, mpi_bcast


class PartitionTable(object):
	'''
	The partition table class contains information on the 
	partition used for the given dataset or  it can generate
	a new partition
	'''
	@cr('ptable.init')
#	@mem('ptable.init')
	def __init__(self,nparts,ids,elements,points,boundaries,has_master=True):
		'''
		Class constructor
		'''
		self._nparts      = nparts
		self._ids         = ids
		self._elements    = elements
		self._points      = points
		self._boundaries  = boundaries
		self._master      = has_master if MPI_SIZE > 1 else False
		self._inods       = None

	def __str__(self):
		out  = 'Partition Table:\nnumber of partitions: %d\n' % self.n_partitions
		out += '\tIds  |  Elements  |  Points  |   Boundaries\n'
		for ipart in range(self.n_partitions):
			out += '\t %03d |    %04d    |    %04d    |    %04d \n' %(self.Ids[ipart],self.Elements[ipart],self.Points[ipart],self.Boundaries[ipart])
		return out

	def __get__(self,i):
		return [self.Points[i],self.Elements[i],self.Boundaries[i]]

	@cr('ptable.check_split')
	def check_split(self):
		'''
		See if a table has the same number of subdomains
		than the number of mpi ranks
		'''
		# Deal with master and serial
		offst = 1 if self._master and not MPI_SIZE == 1 else 0
		return self._nparts + offst == MPI_SIZE

	@cr('ptable.update_points')
	def update_points(self,npoints_new):
		'''
		Update the number of points on the table
		'''
		p = mpi_gather(npoints_new,all=True)
		self._points = p[1:].copy() if isinstance(p,np.ndarray) else np.array([p],np.int32) # Hello master

	@cr('ptable.part_bounds')
	def partition_bounds(self,rank,info,ndim=1,):
		'''
		Compute the partition bounds for a given rank
		'''
		if self._master and rank == 0 and not MPI_SIZE == 1:  return 0, 1
		mask_idx = self.Ids < rank
		this_idx = self.Ids == rank
		if info == 'Points':
			table = self.Points 
		if info == 'Elements':
			table = self.Elements
		if info == 'Boundaries':
			table = self.Boundaries
		istart   = np.sum(table[mask_idx])*ndim
		iend     = istart + table[this_idx][0]*ndim
		return istart, iend

	@cr('ptable.set_inods')
	def set_partition_points(self):
		'''
		Compute the points to be read for this partition
		'''
		inods = []
		offst = 1 if self._master and MPI_SIZE > 0 else 0
		for ipart in range(0,self.n_partitions + offst):
			istart, iend = self.partition_bounds(ipart,'Points')
			inods.append( np.arange(istart,iend,1,np.int32) )
		return inods

	@cr('ptable.update_inods')
	def update_partition_points(self,ipart,npoints,conec=None,ndim=1):
		'''
		Update the points to be read for this partition
		'''
		if self._inods is None: self._inods = self.set_partition_points()
		# Find which nodes this partition has
		unods = np.unique(conec.flatten())
		inods = np.array([],np.int32)
		# Deal with multiple dimensions
		for idim in range(ndim):
			inods = np.hstack((inods,unods+idim*npoints))
		self._inods[ipart] = inods
		return inods

	@cr('ptable.part_inods')
	def get_partition_points(self,rank):
		'''
		Get the points to be read for this partition
		'''
		if self._inods is None: self._inods = self.set_partition_points()
		return self._inods[rank]

	@classmethod
	@cr('ptable.new')
	def new(cls,nparts,nelems=0,npoints=0,has_master=True):
		'''
		Create a new partition table, in serial algorithm.
		'''
		ids        = np.zeros((nparts,),np.int32)
		points     = np.zeros((nparts,),np.int32)
		elements   = np.zeros((nparts,),np.int32)
		boundaries = np.zeros((nparts,),np.int32)
		# For all the partitions do
		for ipart in range(nparts):
			ids[ipart] = ipart + 1 if has_master and MPI_SIZE > 1 else ipart
		if nelems > 0:
			for ipart in range(nparts):
				# Split the number of elements
				istart, iend = worksplit(0,nelems,ipart,nWorkers=nparts)
				# How many elements do I have
				elements[ipart] = iend - istart
		if npoints > 0:
			for ipart in range(nparts):
				# How many nodes do I have
				istart, iend  = worksplit(0,npoints,ipart,nWorkers=nparts)
				points[ipart] = iend - istart
		return cls(nparts,ids,elements,points,boundaries,has_master=has_master)

	# Functions to create a partition table from a 
	# field and a mesh class
	@classmethod
	@cr('ptable.fromField')
	def fromField(cls,field,has_master):
		'''
		Create a partition table from a field
		'''
		# Gather on rank 0 the number of elements, points and 
		# boundaries from all the processors
		npointG = mpi_gather(field.npoints if not np.any(np.isnan(field.xyz)) else 0,root=0)
		ptable  = None
		# Only rank 0 has the data to build and write the table
		if is_rank_or_serial(0):
			nparts     = max(MPI_SIZE-1,1) if has_master else MPI_SIZE
			ids        = np.zeros((nparts,),np.int32)
			points     = np.zeros((nparts,),np.int32)
			elements   = np.zeros((nparts,),np.int32)
			boundaries = np.zeros((nparts,),np.int32)
			if isinstance(npointG,np.ndarray):
				# Build the partition table
				for ii in range(nparts):
					# Build the partition table
					ids[ii]    = ii + 1 if has_master and MPI_SIZE > 1 else ii
					points[ii] = max(npointG[ii+1],0) if has_master and MPI_SIZE > 1 else max(npointG[ii],0)
			else:
				# Build the partition table
				ids[:]    = 1 if has_master and MPI_SIZE > 1 else 0
				points[:] = npointG
			# Create partition table
			ptable = cls(nparts,ids,elements,points,boundaries,has_master=has_master)
		return mpi_bcast(ptable,root=0)

	@classmethod
	@cr('ptable.fromMesh')
	def fromMesh(cls,mesh,has_master):
		'''
		Create a partition table from a mesh
		'''
		# Gather on rank 0 the number of elements, points and 
		# boundaries from allthe processors
		nelG   = mpi_gather(mesh.nel  if not np.any(mesh.eltype < 0)    else 0,root=0)
		npoinG = mpi_gather(mesh.nnod if not np.any(np.isnan(mesh.xyz)) else 0,root=0)
		ptable = None
		# Only rank 0 has the data to build and write the table
		if is_rank_or_serial(0):
			nparts     = max(MPI_SIZE-1,1) if has_master else MPI_SIZE
			ids        = np.zeros((nparts,),np.int32)
			points     = np.zeros((nparts,),np.int32)
			elements   = np.zeros((nparts,),np.int32)
			boundaries = np.zeros((nparts,),np.int32)
			if isinstance(nelG,np.ndarray):
				# Build the partition table
				for ii in range(nparts):
					# Build the partition table
					ids[ii]      = ii + 1 if has_master and MPI_SIZE > 1 else ii
					elements[ii] = max(nelG[ii+1],0) if has_master else max(nelG[ii],0)
					points[ii]   = max(npoinG[ii+1],0) if has_master else max(npoinG[ii],0)
			else:
				# Build the partition table
				ids[:]      = 1 if has_master and MPI_SIZE > 1 else 0
				elements[:] = nelG
				points[:]   = npoinG
			# Create partition table
			ptable = cls(nparts,ids,elements,points,boundaries,has_master=has_master)
		return mpi_bcast(ptable,root=0)

	# Functions to read and write a partition table 
	# for Alya
	@classmethod
	@cr('ptable.fromAlya')
	def fromAlya(cls,casestr,basedir='./'):
		'''
		Read a partition table from Alya
		'''
		partfile = os.path.join(basedir,io.MPIO_PARTFILE_FMT % casestr)
		ptable   = io.AlyaMPIO_readPartitionTable(partfile)
		return cls(ptable.shape[0],ptable[:,0],ptable[:,1],ptable[:,2],ptable[:,3],has_master=True)
	
	@cr('ptable.toAlya')
	def toAlya(self,casestr,basedir='./'):
		'''
		Write a partition file to Alya format
		'''
		partfile = os.path.join(basedir,io.MPIO_PARTFILE_FMT % casestr)
		io.AlyaMPIO_writePartitionTable(partfile,self.ptableMesh)

	# Functions to read and write a partition table 
	# for SOD2D
	@classmethod
	@cr('ptable.fromSOD2D')
	def fromSOD2DHDF(cls,casestr,basedir='./',nprocs=MPI_SIZE):
		'''
		Read a partition table from SOD2D
		'''
		sodfile    = os.path.join(basedir,io.SOD2DHDF_MESH_FMT % (casestr,MPI_SIZE))
		ptable     = io.SOD2DHDF_readPartitionTable(sodfile,nprocs=nprocs)
		nparts     = ptable['Points'].shape[0]
		ids        = np.arange(0, nparts)
		elements   = ptable['Elements'][:,1]-ptable['Elements'][:,0]
		points     = ptable['Points'][:,1]-ptable['Points'][:,0]
		boundaries = ptable['Boundaries'][:,1]-ptable['Boundaries'][:,0]
		return cls(nparts,ids,elements,points,boundaries,has_master=False)

	@property
	def n_partitions(self):
		return self._nparts
	@property
	def Ids(self):
		return self._ids
	@property
	def Elements(self):
		return self._elements
	@property
	def Points(self):
		return self._points
	@property
	def Boundaries(self):
		return self._boundaries
	@property
	def has_master(self):
		return self._master
	@property
	def is_serial(self):
		if self is None or self.n_partitions == 1:
			return True
		else:
			return False
	
	@property
	def ptableMesh(self):
		ptable = np.zeros((self._nparts,4),np.int32)
		ptable[:,0] = self._ids
		ptable[:,1] = self._elements
		ptable[:,2] = self._points
		ptable[:,3] = self._boundaries
		return ptable
	@property
	def ptableField(self):
		ptable = np.zeros((self._nparts,2),np.int32)
		ptable[:,0] = self._ids
		ptable[:,1] = self._points
		return ptable
