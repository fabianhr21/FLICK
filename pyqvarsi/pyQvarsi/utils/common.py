#!/usr/bin/env python
#
# pyQvarsi, utils.
#
# Common utility routines.
#
# Last rev: 01/10/2020
from __future__ import print_function, division

import sys, numpy as np, subprocess

from .parallel import MPI_COMM, MPI_RANK, MPI_SIZE


def pprint(rank,*args,**kwargs):
	'''
	Print alternative for parallel codes. It works as
	python's print with the rank variable, which can 
	be negative for everyone to print or equal to the
	rank that should print.
	'''
	if rank < 0 or MPI_SIZE == 1:
		print(MPI_RANK,*args,**kwargs)
	elif rank == MPI_RANK:
		print(MPI_RANK,*args,**kwargs)


def raiseError(errmsg,all=True):
	'''
	Raise a controlled error and abort execution on
	all processes.
	'''
	if all:
		print('Error: %d - %s' % (MPI_RANK,errmsg),file=sys.stderr,flush=True)
	else:
		if MPI_RANK == 0 or MPI_SIZE == 1:
			print('Error: %d - %s' % (MPI_RANK,errmsg),file=sys.stderr,flush=True)
	MPI_COMM.Abort(1)


def raiseWarning(warnmsg,all=True):
	'''
	Raise a controlled warning but don't abort execution on
	all processes.
	'''
	if all:
		print('Warning: %d - %s' % (MPI_RANK,warnmsg),file=sys.stderr,flush=True)
	else:
		if MPI_RANK == 0 or MPI_SIZE == 1:
			print('Warning: %s' % (warnmsg),file=sys.stderr,flush=True)


def truncate(value,precision):
	'''
	Truncate array by a certain precision
	'''
	fact  = 10**precision
	return np.round(value*fact)/fact


def printArray(name,Array,rank=-1,precision=4):
	'''
	An easy and fancy way of printing arrays for debugging
	'''
	nanstr = '(has NaNs)' if np.any(np.isnan(Array)) else ''
	pprint(rank,name,nanstr,
			   'size=',Array.shape,
			   'max=',truncate(np.nanmax(Array,axis=0),precision),
			   'min=',truncate(np.nanmin(Array,axis=0),precision),
			   'avg=',truncate(np.nanmean(Array,axis=0),precision),
			   flush=True)

def run_subprocess(runpath,runbin,runargs,nprocs=1,host=None,log=None,srun=False):
	'''
	Use python to call a terminal command
	'''
	# Build command to run
	if srun:
		# Sometimes we will need to use srun...
		cmd = 'cd %s && srun -n %d %s %s'%(runpath,nprocs,runbin,runargs) if log is None else 'cd %s && srun -n %d %s %s > %s 2>&1'%(runpath,nprocs,runbin,runargs,log)
	else:
		if nprocs == 1:
			# Run a serial command
			cmd = 'cd %s && %s %s'%(runpath,runbin,runargs) if log is None else 'cd %s && %s %s > %s 2>&1'%(runpath,runbin,runargs,log)
		else:
			# Run a parallel command
			if host is None:
				cmd = 'cd %s && mpirun -np %d %s %s'%(runpath,nprocs,runbin,runargs) if log is None else 'cd %s && mpirun -np %d %s %s > %s 2>&1'%(runpath,nprocs,runbin,runargs,log)
			else:
				cmd = 'cd %s && mpirun -np %d -host %s %s %s'%(runpath,nprocs,host,runbin,runargs) if log is None else 'cd %s && mpirun -np %d -host {3} %s %s > %s 2>&1'%(runpath,nprocs,host,runbin,runargs,log)
	# Execute run
	subprocess.call(cmd,shell=True)