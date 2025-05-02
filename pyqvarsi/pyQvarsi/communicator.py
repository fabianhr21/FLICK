#!/usr/bin/env python
#
# pyQvarsi, Communicator.
#
# Class to manage the point-to-point communications between
# different subdomains in a similar manner to that of Alya.
#
# Last rev: 20/10/2020
from __future__ import print_function

import numpy as np

from .cr             import cr
from .mem            import mem
from .utils.parallel import MPI_RANK, MPI_SIZE, split_table
from .utils.parallel import mpi_send, mpi_recv, mpi_sendrecv, mpi_barrier, mpi_scatter, mpi_reduce, mpi_bcast, is_rank_or_serial


class Communicator(object):
	'''
	Class to manage the point-to-point communications between
	different subdomains in a similar manner to that of Alya.
	'''
	@cr('comm.init')
#	@mem('comm.init')
	def __init__(self, neights, size, perm):
		'''
		Class constructor.
		'''
		self._myrank  = MPI_RANK # Rank of the current mesh partition
		self._neights = neights  # Ranks of the neighbours
		self._size    = size     # Size of the boundaries
		self._perm    = perm     # Permutation matrix

	@classmethod
	@cr('comm.from_Alya')
	def from_Alya(cls,commu,lninv,lmast):
		'''
		Create a new instance of the communicator class given
		the communications matrix commu.
		'''
		if is_rank_or_serial(0):
			neights = np.array([0],np.int32)
			size    = np.array([1,1],np.int32)
			perm    = np.array([0],np.int32)
		else:
			# Modify the ordering array so that the periodic nodes have the same numbering
			lninv = lninv.copy() # Ensure making a local copy
			mask  = lmast > 0
			# Must take the positive ones as the negatives are the master
			lninv[mask] = lmast[mask] - 1 # -1 due to python numeration

			# Obtain the list of the unique neighbouring ranks
			# in the right order
			shape   = commu.shape
			commu   = commu.ravel()
			neights = np.unique(commu)[1:].astype(np.int32) # discard 0

			# Obtain the permutation array
			perm = np.array([],np.int32)
			size = np.zeros((len(neights)+1,),np.int32)
			size[0] = 1
			for ineig,neig in enumerate(neights):
				p  = np.unravel_index(np.where(commu == neig)[0],shape)[0]
				sz = len(p)
				# We need to order p in ascending lninv
				idx = np.argsort(lninv[p])
				# Append to perm and compute size
				perm = np.append(perm,p[idx])
				size[ineig+1] = sz + size[ineig]
		# Return an instance of the class
		return cls(neights=neights,size=size,perm=perm)
	
	@classmethod
	@cr('comm.from_SOD2D')
	def from_SOD2D(cls,ranksToComm,commsMemSize,nodesToComm):
		'''
		Create a new instance of the communicator class given
		the communications matrix commu.
		'''
		size = np.ones((commsMemSize.shape[0]+1,),dtype=np.int32)
		for irank in range(commsMemSize.shape[0]):
			size[irank+1] = size[irank]+commsMemSize[irank]
		return cls(neights=ranksToComm,size=size,perm=nodesToComm)

	def __str__(self):
		return 'Communicator of rank %d with neighbours: ' % self._myrank + str(self.neights)

	# Functions

	def communicate(self,field):
		'''
		Communicate a scalar or array field to its boundaries and return
		the total scalar or array field
		'''
		return self.communicate_scaf(field) if len(field.shape) == 1 else self.communicate_arrf(field)

	@cr('comm.comm_scaf')
	def communicate_scaf(self,scaf):
		'''
		Communicate a scalar field to its boundaries and return
		the total scalar field
		'''
		# Preallocate send and recieve buffers
		sendbuff = np.zeros((self.dim,),dtype=scaf.dtype)
		recvbuff = np.zeros((self.dim,),dtype=scaf.dtype)
		# Load send buffer
		sendbuff[:] = scaf[self.perm]
		# Point to point communication
		for ineig in range(self.nneights):
			dom_i = self.neights[ineig]
			ini = self.size[ineig]-1
			bsz = self.size[ineig+1]-1
			recvbuff[ini:bsz] = self.sendrecv(sendbuff[ini:bsz],dest=dom_i,source=dom_i)

		# Sum the received buffer
		for jj in range(self.dim):
			ip = self.invp[jj]
			scaf[ip] += recvbuff[jj]
		return scaf

	@cr('comm.comm_arrf')
	def communicate_arrf(self,arrf):
		'''
		Communicate an array field to its boundaries 
		and return the total field
		'''
		# Preallocate send and recieve buffers
		sendbuff = np.zeros((self.dim,arrf.shape[1]),dtype=arrf.dtype)
		recvbuff = np.zeros((self.dim,arrf.shape[1]),dtype=arrf.dtype)
		# Load send buffer
		sendbuff[:,:] = arrf[self.perm,:]
		# Point to point communication
		for ineig in range(self.nneights):
			dom_i = self.neights[ineig]

			ini = self.size[ineig]-1
			bsz = self.size[ineig+1]-1

			recvbuff[ini:bsz,:] = self.sendrecv(sendbuff[ini:bsz,:],dest=dom_i,source=dom_i)
		# Sum the received buffer
		for jj in range(self.dim):
			ip = self.invp[jj]
			arrf[ip,:] += recvbuff[jj,:]
		return arrf

	@cr('comm.bc')
	def communicate_bc(self,lninv):
		'''
		Communicate which boundaries of the array
		must be removed and which must not
		'''
		bc_array = np.zeros(lninv.shape,dtype=bool)
		# Preallocate send and recieve buffers
		sendbuff = np.zeros((self.dim,),dtype=lninv.dtype)	
		recvbuff = np.zeros((self.dim,),dtype=lninv.dtype)	
		bcarbuff = np.zeros((self.dim,),dtype=np.int32)	
		# Load send buffer
		sendbuff[:] = lninv[self.perm]
		# Point to point communication
		for ineig in range(self.nneights):
			dom_i = self.neights[ineig]

			ini = self.size[ineig]-1
			bsz = self.size[ineig+1]-1

			recvbuff[ini:bsz] = self.sendrecv(sendbuff[ini:bsz],dest=dom_i,source=dom_i)

			# If the neighbour is less than the current rank
			# flag to remove the neighbours from the array
			if (dom_i < self.rank):
				bcarbuff[ini:bsz] += sendbuff[ini:bsz] == recvbuff[ini:bsz]
		# Finally set the flag vector
		for jj in range(self.dim):
			ip = self.invp[jj]
			bc_array[ip] = True if bcarbuff[jj] > 0 else bc_array[ip]
		return bc_array

	@staticmethod
	def send(f,dest,tag=0):
		'''
		Implements the send operation
		'''
		return mpi_send(f,dest,tag=tag)

	@staticmethod
	def recv(**kwargs):
		'''
		Implements the recieve operation
		'''
		return mpi_recv(**kwargs)

	@staticmethod
	def sendrecv(buff,**kwargs):
		'''
		Implements the sendrecv operation
		'''
		return mpi_sendrecv(buff,**kwargs)

	@staticmethod
	def reduce(f,root=0,op='sum'):
		'''
		Implements the reduce operation
		'''
		return mpi_reduce(f,op=op,root=root,all=False)

	@staticmethod
	def allreduce(f,op='sum'):
		'''
		Implements the allreduce operation
		'''
		return mpi_reduce(f,op=op,all=True)

	@staticmethod
	def bcast(f,root=0):
		'''
		Implements the broadcast operation
		'''
		return mpi_bcast(f,root=root)

	@staticmethod
	def scatter(f,root=0,do_split=False):
		'''
		Implements the scatter operation
		'''
		return mpi_scatter(f,root=root,do_split=do_split)

	@staticmethod
	def scatterp(f,idx,pdx,root=0):
		'''
		Implements the scatter operation according
		to a partition table
		'''
		return mpi_scatter(split_table(f,idx,pdx),root=root,do_split=False)

	@staticmethod
	def barrier():
		'''
		Implements the barrier
		'''
		mpi_barrier()

	@staticmethod
	def rankID():
		'''
		Easily return the rank
		'''
		return MPI_RANK

	@staticmethod
	def sizeID():
		'''
		Easily return the rank
		'''
		return MPI_SIZE

	@staticmethod
	def serial():
		'''
		Return True if rank==0 or serial (MPI_SIZE==1)	
		'''
		return MPI_RANK == 0 or MPI_SIZE == 1

	@cr('comm.to_commu')
	def to_commu(self,nnod):
		'''
		Recreate the communications matrix generated by Alya.
		For debugging purposes at the moment.
		'''
		nneightsG = self.allreduce(self.nneights,op='max')
		commu     = np.zeros((nnod,nneightsG),dtype=np.int32)
		num_neig  = np.zeros((nnod,),dtype=np.int32)

		# Recover commu from array
		for ineig in range(self.nneights):
			dom_i  = self.neights[ineig]
			dom_st = self.size[ineig]-1
			dom_ed = self.size[ineig+1]-1
			for kk in range(dom_st,dom_ed):
				ii = self.perm[kk]
				if num_neig[ii] == nneightsG: continue
				if nneightsG > 1:
					commu[ii,num_neig[ii]] = dom_i
				else:
					commu[ii] = dom_i
				num_neig[ii] += 1

		# Filter zeros on commu array
		filter_zeros  = np.where(np.any(commu != 0,axis=0))[0]
		filter_zerosG = self.allreduce(filter_zeros[-1] if len(filter_zeros) > 0 else 0,op='max')
		commu = commu[:,np.arange(0,filter_zerosG+1)]

		return commu

	# Properties
	@property
	def rank(self):
		return self._myrank
	@property
	def neights(self):
		return self._neights
	@property
	def nneights(self):
		return len(self._neights)
	@property
	def size(self):
		return self._size
	@property
	def perm(self):
		return self._perm
	@property
	def invp(self):
		return self._perm
	@property
	def dim(self):
		return len(self._perm)
	@property
	def nbound(self):
		return len(np.unique(self._perm))
