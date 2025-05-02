#!/usr/bin/env python
#
# SOD2D H5 Input Output
#
# Last rev: 23/11/2022
from __future__ import print_function, division

import os, numpy as np, h5py

from .                import SOD2DHDF_MESH_FMT, SOD2DHDF_RESULTS_FMT
from ..cr             import cr
from ..utils.common   import raiseError, raiseWarning
from ..utils.parallel import MPI_RANK, MPI_SIZE, MPI_COMM


class SOD2D_header(object):
	'''
	'''
	def __init__(self,
		itime       = 0,
		nsub        = 0, 
		time        = 0.):
		'''
		Class constructor
		'''
		# Set the header dict with the default parameters
		self._header = {
			'TstepNo'  : itime,
			'NSubdom'  : nsub,
			'Time'     : time
		}

	def __str__(self):
		retstr = ''
		for key in self._header:
			retstr += '%s : '%key + str(self._header[key]) + '\n'
		return retstr

	@property
	def header(self):
		return self._header

	@property
	def nsubd(self):
		return self._header['NSubdom']
	@nsubd.setter
	def nsubd(self,value):
		self._header['NSubdom'] = value

	@property
	def itime(self):
		return self._header['TstepNo']
	@itime.setter
	def itime(self,value):
		self._header['TstepNo'] = value

	@property
	def time(self):
		return self._header['Time']
	@time.setter
	def time(self,value):
		self._header['Time'] = value

## Functions
def readSOD2DHDFResultsSerial(filename,varlist,rank=MPI_RANK):
	'''
	Read a SOD2D HDF5 results file in serial mode.
	'''
	# Return None for the rest of the ranks
	if not rank == MPI_RANK: return None
	# Open HDF5 file (in serial)
	file = h5py.File(filename,'r')
	file_keys = list(file['VTKHDF']['PointData'].keys())
	# Read the time value
	if 'time' in file_keys:
		time = np.array(file['time'],np.double)
	else:
		time = np.array([0],np.double)
	# Read the requested variables from the file
	data = {}
	for var in varlist:
		if not var in file_keys: raiseError('Variable <%s> not inside the file <%s>!'%(var,filename))
		data[var] = np.array(file['VTKHDF']['PointData'][var]) # Already picks the dtpye
		# Fix float32 dtpye
		if data[var].dtype == np.float32: data[var] = data[var].astype(np.double)
	# Close file
	file.close()
	return data, time

def readSOD2DHDFResultsParallel(filename,varlist,partition):
	'''
	Read a SOD2D HDF5 results file in parallel mode.
	'''
	# Open HDF5 file (in parallel)
	file = h5py.File(filename,'r',driver='mpio',comm=MPI_COMM)
	file_keys = list(file['VTKHDF']['PointData'].keys())
	# Read the time value
	if 'time' in file_keys:
		time = np.array(file['time'],np.double)
	else:
		time = np.array([0],np.double)
	# Recover start and end for the partition
	start = partition['Points'][MPI_RANK,0]
	end   = partition['Points'][MPI_RANK,1]
	# Read the requested variables from the file
	data = {}
	for var in varlist:
		if not var in file_keys: raiseError('Variable <%s> not inside the file <%s>!'%(var,filename))
		data[var] = np.array(file['VTKHDF']['PointData'][var][start:end]) # Already picks the dtpye
		# Fix float32 dtpye
		if data[var].dtype == np.float32: data[var] = data[var].astype(np.double)
	# Close file
	file.close()
	return data, time

@cr('SOD2DIO.read_repart')
def readSOD2DHDFResultsRepart(filename,varlist,partition,mapping,ranks):
	'''
	Read a SOD2D HDF5 results file in parallel mode.
	'''
	# Open HDF5 file (in parallel)
	file = h5py.File(filename,'r',driver='mpio',comm=MPI_COMM)
	file_keys = list(file['VTKHDF']['PointData'].keys())
	# Read the time value
	if 'time' in file_keys:
		time = np.array(file['time'],np.double)
	else:
		time = np.array([0],np.double)
	# Read the requested variables from the file
	data = {}
	myranks = np.sort(np.unique(ranks))
	for var in varlist:
		if not var in file_keys: raiseError('Variable <%s> not inside the file <%s>!'%(var,filename))
		for ii, iproc in enumerate(myranks):
			start = partition['Points'][iproc,0]
			end   = partition['Points'][iproc,1]
			rankdata = np.array(file['VTKHDF']['PointData'][var][start:end])
			if ii == 0:
				mydata = np.zeros((mapping.shape[0], rankdata.shape[1]), dtype=rankdata.dtype) if len(rankdata.shape) > 1 else np.zeros(mapping.shape)
			mydata[ranks==iproc] = rankdata[mapping[ranks==iproc]]
		data[var] = mydata
		# Fix float32 dtpye
		if data[var].dtype == np.float32: data[var] = data[var].astype(np.double)
	# Close file
	file.close()
	return data, time

def SOD2DHDF_readPartitionTable(filename,nprocs=MPI_SIZE):
	'''
	Read partitions from the mesh file and return the partition table.	
	'''
	# Open h5 file in serial
	file = h5py.File(filename,'r')
	info    = file.keys()
	has_bou = 'Boundary_data' in info
	if nprocs > 1:
		# Read the start and end part for nodes and elements
		# Subtracting 1 due to python indexing
		if 'Parallel_data' in info:
			node_start = np.array(file['Parallel_data']['rankNodeStart'],np.int32) - 1
			node_end   = np.array(file['Parallel_data']['rankNodeEnd'],np.int32)
			elem_start = np.array(file['Parallel_data']['rankElemStart'],np.int32) - 1
			elem_end   = np.array(file['Parallel_data']['rankElemEnd'],np.int32)
			# Find the start and end for boundary elements
			bou_start     = np.zeros(MPI_SIZE, dtype=np.int32)
			bou_end       = np.zeros(MPI_SIZE, dtype=np.int32)
			if has_bou:   
				nelbou        = np.array(file['Boundary_data']['numBoundsRankPar'])
				bou_end[0]    = nelbou[0]
				bou_start[1:] = np.cumsum(nelbou[:-1])
				bou_end[1:]   = bou_start[1:] + nelbou[1:]
		else:
			npoints    = np.array(file['VTKHDF']['NumberOfPoints'],np.int32) 
			nelems     = np.array(file['VTKHDF']['NumberOfCells'],np.int32)
			node_end   = np.cumsum(npoints)
			node_start = np.hstack(([0],node_end[:-1]))
			elem_end   = np.cumsum(nelems)
			elem_start = np.hstack(([0],elem_end[:-1]))
			bou_end    = np.zeros(node_end.shape, dtype=int)
			bou_start  = np.zeros(node_end.shape, dtype=int)
	else:
		node_start = 0
		node_end   = file['VTKHDF']['Points'].shape[0]
		elem_start = 0
		elem_end   = file['Connectivity']['connecParWork'].shape[0]//((file['order']['porder'][0]+1)**3)
		bou_start  = 0
		if has_bou:
			bou_end = np.array(file['Boundary_data']['numBoundsRankPar'])[0]
		else:
			bou_end = 0
	file.close()
	partition_table = {
		'Points'     : np.vstack([node_start,node_end]).T,
		'Elements'   : np.vstack([elem_start,elem_end]).T,
		'Boundaries' : np.vstack([bou_start,bou_end]).T,
	}
	return partition_table

@cr('SOD2DHDF.read_mesh')
def SOD2DHDF_read_mesh(casename,ptable,nprocs=MPI_SIZE,rank=0,basedir='./'):
	'''
	Read SOD2D mesh file.
	'''
	filename_mesh = os.path.join(basedir,SOD2DHDF_MESH_FMT%(casename,nprocs))
	# Read in serial or parallel
	if MPI_SIZE == 1 and rank != -1:
		file  = h5py.File(filename_mesh,'r')
		ismpi = False
	else:
		file  = h5py.File(filename_mesh,'r',driver='mpio',comm=MPI_COMM)
		ismpi = True
	info    = list(file.keys())
	if len(info) == 1 and info[0] == 'VTKHDF':
		raiseWarning("WARNING!! MESH HAS BEEN REPARTITIONED, ONLY CAN BE USED FOR VISUALIZATION PURPOSES, NOT FOR COMPUTATIONS",all=False)
		raiseWarning("WARNING!! ASSUMING THAT THE MESH HAS BEEN PREVIOUSLY LINEARIZED",all=False)
		ismpi   = False # No parallel data avialable
		porder  = 1
		nnodxel = (porder+1)**3
		nnodxbo = (porder+1)**2
		# Recover start and end for the partition
		plim = ptable.partition_bounds(MPI_RANK,'Points')
		elim = ptable.partition_bounds(MPI_RANK,'Elements')
		startp, starte = plim[0], elim[0]
		endp,   ende,  = plim[1], elim[1]
		# Read node and element numbering
		lninv = np.linspace(startp, endp, num=endp-startp)
		leinv = np.linspace(starte, ende, num=ende-starte)
		# Read node coordinates
		xyz = np.array(file['VTKHDF']['Points'][startp:endp,:],dtype=np.double) 
		# Read element connectivity
		lnods = np.array(file['VTKHDF']['Connectivity'][nnodxel*starte:nnodxel*ende]).reshape(leinv.shape[0],nnodxel,order='C')+1
	else:
		is_lo = np.array(file['meshOutputInfo']['isLinealOutput'], dtype=bool)[0]
		raiseWarning("WARNING!! MESH IS NOT LINEALIZED, MESH IS TAKEN AT GLL COORDINATES BUT FIELD DATA WILL BE IN EQUISPACED COORDINATES!!" if not is_lo else "Linearized mesh",all=False)
		porder  = file['order']['porder'][0]
		nnodxel = (porder+1)**3
		nnodxbo = (porder+1)**2
		# Recover start and end for the partition
		plim = ptable.partition_bounds(MPI_RANK, 'Points')
		elim = ptable.partition_bounds(MPI_RANK, 'Elements')
		blim = ptable.partition_bounds(MPI_RANK, 'Boundaries')
		startp, starte, startb = plim[0], elim[0], blim[0]
		endp,   ende,   endb   = plim[1], elim[1], blim[1]
		# Read node and element numbering
		lninv = np.array(file['globalIds']['globalIdPar'][startp:endp])
		leinv = np.array(file['globalIds']['elemGid'][starte:ende])
		# Read node coordinates
		xyz = np.array(file['VTKHDF']['Points'][startp:endp,:],dtype=np.double) if is_lo else np.array(file['Coords']['Points'][startp:endp,:],dtype=np.double)
		# Read element connectivity
		lnods = np.array(file['Connectivity']['connecParOrig'][nnodxel*starte:nnodxel*ende]).reshape(leinv.shape[0],nnodxel,order='C')
		# Read node ordering
		a2ijk = np.array(file['order']['a2ijk']) - 1
		lnods = lnods[:, a2ijk]
	# Read parallel data
	if ismpi:
		numRanksWithComms = np.array(file['Parallel_data']['numRanksWithComms'])
		iRankStart        = np.sum(numRanksWithComms[:MPI_RANK])
		iRankEnd          = iRankStart + numRanksWithComms[MPI_RANK]
		ranksToComm       = np.array(file['Parallel_data']['ranksToComm'][iRankStart:iRankEnd])
		numNodesToComm    = np.array(file['Parallel_data']['numNodesToComm'])
		commsMemSize      = np.array(file['Parallel_data']['commsMemSize'][iRankStart:iRankEnd])
		iRankStart        = np.sum(numNodesToComm[:MPI_RANK])
		iRankEnd          = iRankStart + numNodesToComm[MPI_RANK]
		nodesToComm       = np.array(file['Parallel_data']['nodesToComm'][iRankStart:iRankEnd])
	else:
		ranksToComm  = np.array([])
		commsMemSize = np.array([])
		nodesToComm  = np.array([])
	# Read boundary mesh
	if 'Boundary_data' in info:
		nelbou   = endb - startb
		boucodes = np.array(file['Boundary_data']['bouCodesPar'][startb:endb])
		bouconec = np.array(file['Boundary_data']['boundParOrig'][startb*nnodxbo:endb*nnodxbo]).reshape(nelbou,nnodxbo,order='C')
		# Read node ordering
		a2ij     = np.array(file['order']['a2ij']) - 1
		bouconec = bouconec[:,a2ij]
	else:
		boucodes = np.array([])
		bouconec = np.array([[]])
		# VTKHDF can contain the CellData/boundCode array
		if 'boundCode' in file['VTKHDF']['CellData'].keys():
			boucodes = np.array(file['VTKHDF']['CellData']['boundCode'][starte:ende]) 
	# We don't care about periodicity
	lmast = np.array([])
	# Close file
	file.close()
	return xyz, lnods, lninv, leinv, porder, boucodes, bouconec, lmast, ranksToComm, commsMemSize, nodesToComm

@cr('SOD2DHDF.read_results')
def SOD2DHDF_read_results(casename,varlist,instant,nprocs,rank=0,basedir='./',force_partition_data=False):
	'''
	Read SOD2D results file and return the 
	specified variables and the time.
	'''
	filename_mesh    = os.path.join(basedir,SOD2DHDF_MESH_FMT%(casename,nprocs))
	filename_results = os.path.join(basedir,SOD2DHDF_RESULTS_FMT%(casename,nprocs,instant))
	# Read in serial or parallel
	if MPI_SIZE == 1 and rank != -1:
		data, time = readSOD2DHDFResultsSerial(filename_results,varlist,rank=rank)
	else:
		if MPI_SIZE != nprocs:
			## Read partition table of the postprocessing mesh
			postmesh      = os.path.join(basedir,SOD2DHDF_MESH_FMT%(casename,MPI_SIZE))
			postpartition = SOD2DHDF_readPartitionTable(postmesh,nprocs=MPI_SIZE)
			start, end    = postpartition['Points'][MPI_RANK,0], postpartition['Points'][MPI_RANK,1]
			## Read mapping between postprocessing and running meshes
			file    = h5py.File('%s-mapping-%i-to-%i.hdf' % (casename, MPI_SIZE, nprocs),'r')
			mapping = np.array(file['mappingNode'][start:end]) - 1
			ranks   = np.array(file['mappingRank'][start:end])
			file.close()
			## Read partition table of the running mesh
			runpartition = SOD2DHDF_readPartitionTable(filename_mesh,nprocs=nprocs)
			data, time = readSOD2DHDFResultsRepart(filename_results,varlist,runpartition,mapping,ranks)
		else:
			if force_partition_data or not hasattr(SOD2DHDF_read_results,'partition_data'):
				SOD2DHDF_read_results.partition_data = SOD2DHDF_readPartitionTable(filename_mesh,nprocs=nprocs)
			data, time = readSOD2DHDFResultsParallel(filename_results,varlist,SOD2DHDF_read_results.partition_data)
	return data, SOD2D_header(itime=instant,nsub=nprocs,time=time)