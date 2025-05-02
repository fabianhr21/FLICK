#!/usr/bin/env python
#
# pyQvarsi, checkpoint.
#
# Checkpoint class, for those long cases.
#
# Last rev: 16/12/2020
from __future__ import print_function, division

import os, copy, mpi4py, numpy as np
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

from .   import inp_out as io
from .cr import cr_start, cr_stop

comm     = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()


def raiseError(errmsg):
	'''
	Raise a controlled error and abort execution on
	all processes.
	'''
	print('%d - %s' % (rank,errmsg),file=sys.stderr,flush=True)
	comm.Abort(1)


class Checkpoint(object):
	'''
	Class to manage the restart of a long case from the point
	where it was left. 
	'''
	def __init__(self,freq,ii_start,ii_end,outdir,step=1,restarted=False,schedule=-1):
		self._counter   = 0
		self._freq      = freq
		self._outdir    = outdir
		self._ii_start  = ii_start
		self._ii_end    = ii_end
		self._ii_curr   = -1
		self._step      = step
		self._restarted = restarted
		self._schedule  = schedule
		self._ndiv      = mpi_size//schedule if schedule > 0 else 1
		self._flag      = -1 # User submitted flag to restart different parts of the program
		self._varnames  = []

	def __str__(self):
		'''
		String representation
		'''
		s  = 'Checkpoint (flag=%d), restarted = %s and counting = %d(%d) (%d:%d:%d)\n' % (self._flag,self._restarted,self._counter,self._ii_curr,self._ii_start,self._step,self._ii_end)
		s += 'With variables:\n'
		for key in self.varnames: s += '\t> %s' % key
		return s

	# Set and get functions
	def __getitem__(self,key):
		'''
		Checkpoint[key]

		Recover the value of a variable given its key
		'''
		return self._load_pkl(self._outdir,key)

	def clean(self,rank=0):
		'''
		Eliminate any existing checkpoints.
		'''
		if mpi_rank == rank: os.system('rm -rf %s' % (self.dir))

	def reset(self,flag=-1):
		'''
		Reset the counters
		'''
		self._ii_curr = -1
		self._counter = 0
		self._flag    = flag

	def enter_part(self,flag=None,barrier=True):
		'''
		Returns true if the case has not been restarted or
		the given flag is equal to the restarted flag.

		Used to protect areas of the code.
		'''
		if barrier: comm.Barrier() # Force all procs to enter at the same time
		return True if not self._restarted else flag == self._flag

	def save(self,flag,ii,msg,**kwargs):
		'''
		Save a checkpoint at a given instant if the saving
		condition is met, otherwise increase the counter.

		Store the current instant and the given variables
		inside a pkl file.
		'''
		if self.save_checkpoint:
			# First restart the counter
			self._counter = 0
			# Save
			self.force_save(flag,ii,msg,**kwargs)
		else:
			self._counter += 1 # Increase the counter

	def force_save(self,flag,ii,msg,**kwargs):
		'''
		Save a checkpoint at a given instant. This function is used
		inside "save" or can be used standalone to force the saving
		of a checkpoint, regardless of the frequency condition.

		It will not increment or restart the counter
		'''
		# Barrier to synchronize
		comm.Barrier()
		# This barrier is here otherwise master waits inside this function
		# and keeps the timer going in while the others are computing
		cr_start('checkpoint save',0)
		self._flag = flag
		tmp_folder = os.path.join(self._outdir,'tmp')
		# Master creates tmp folder
		if mpi_rank == 0:
			os.system('mkdir -p %s' % tmp_folder)
		# Barrier to synchronize
		comm.Barrier()
		# Update instant and variable names
		self._ii_curr  = ii     # Update current step
		self._varnames = [name for name in kwargs.keys()]
		# One by one save pkl, block the rest with a receive
		if self._schedule > 0 and mpi_rank%self._ndiv > 0: comm.recv(source=mpi_rank-1)
		# Save checkpoint class
		self._save_pkl(tmp_folder,'checkpoint',self)
		# Save variables
		for var in self._varnames:
			self._save_pkl(tmp_folder,var,kwargs[var])
		# Send to the next to start saving
		if self._schedule > 0 and not mpi_rank == mpi_size-1: comm.send(1,dest=mpi_rank+1)
		# Master prints save message and updates the checkpoint
		comm.Barrier()
		if mpi_rank == 0:
			print(msg + ' (flag=%d)'%self._flag,flush=True)
			os.system('mv %s/* %s' % (tmp_folder,self._outdir))
			os.system('rm -rf %s' % (tmp_folder))
		cr_stop('checkpoint save',0)

	def load(self):
		'''
		Load a checkpoint in pickle file and 
		return a checkpoint class.
		'''
		cr_start('checkpoint load',0)
		checkp = self._load_pkl(self._outdir,'checkpoint')		
		checkp._restarted = True
		cr_stop('checkpoint load',0)
		return checkp

	@classmethod
	def create(cls,freq,ii_start,ii_end,step=1,basedir='.',checkpdir='checkpoint',schedule=-1):
		'''
		Create or load checkpoint. Anyway this function returns an instance
		of the checkpoint class.
		'''
		cr_start('checkpoint create',0)
		outdir     = os.path.join(basedir,checkpdir)
		checkpfile = os.path.join(outdir,'checkpoint_%d.pkl' % mpi_rank)
		checkp     = cls(freq,ii_start,ii_end,outdir,step=step,restarted=False,schedule=schedule)
		if os.path.exists(checkpfile):
			cr_stop('checkpoint create',0)
			return checkp.load()
		else:
			if mpi_rank == 0 and not os.path.exists(outdir): os.mkdir(outdir)
			cr_stop('checkpoint create',0)
			return checkp

	@staticmethod
	def _save_pkl(outdir,varname,var):
		'''
		Save a variable in pkl format
		'''
		fname = os.path.join(outdir,'%s_%d.pkl' % (varname,mpi_rank))
		io.pkl_save(fname,var)

	@staticmethod
	def _load_pkl(outdir,varname):
		'''
		Load a variable in pkl format
		'''	
		fname = os.path.join(outdir,'%s_%d.pkl' % (varname,mpi_rank))
		return io.pkl_load(fname)

	@property
	def dir(self):
		return self._outdir
	@property
	def start(self):
		return self._ii_start if self._ii_curr < 0 else self._ii_curr
	@property
	def end(self):
		return self._ii_end
	@property
	def range(self):
		return range(self.start,self.end+self._step,self._step)
	@property
	def listrange(self):
		return [ii for ii in range(self.start,self.end+self._step,self._step)]
	@property
	def restarted(self):
		return self._restarted
	@property
	def flag(self):
		return self._flag
	
	@property
	def save_checkpoint(self):
		return self._counter == self._freq
	@property
	def varnames(self):
		return self._varnames