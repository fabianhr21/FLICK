#!/usr/bin/env python
#
# H5 Input Output
#
# Last rev: 19/02/2021
from __future__ import print_function, division

import os, numpy as np, h5py

from ..cr              import cr
from ..partition_table import PartitionTable
from ..utils.common    import raiseError
from ..utils.parallel  import MPI_RANK, MPI_SIZE, MPI_COMM, mpi_reduce, is_rank_or_serial


PYQVARSI_H5_STRING  = 'pyQvarsi HDF5 exchange file'
PYQVARSI_H5_VERSION = (2,0)


def h5_write_attributes(file):
	'''
	Writes the attributes for an HDF5 using pyQvarsi
	'''
	file.attrs['Description'] = PYQVARSI_H5_STRING
	file.attrs['Version']     = PYQVARSI_H5_VERSION	

def h5_check_attributes(file):
	'''
	Check the attributes of an HDF5 to see if it matches
	with a pyQvarsi file
	'''
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYQVARSI_H5_VERSION:
		raiseError('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYQVARSI_H5_VERSION)))


def h5_write_metadata(file,metadata):
	'''
	Write metadata inside an HDF5
	'''
	group = file.create_group('METADATA') if not 'METADATA' in file.keys() else file['METADATA']
	# Create the datasets - all ranks
	mdict = {}
	for var in metadata.keys():
		mdict[var] = group[var] if var in group else group.create_dataset(var,(1,),dtype=metadata[var][1])
	# Store in the datasets - only rank 0 or serial
	if is_rank_or_serial(0):
		for var in mdict.keys(): mdict[var][:] = metadata[var][0]


def h5_write_partition_table(file,ptable,is_field=False,overwrite=True):
	'''
	Write partition table data for field and mesh HDF5
	'''
	if 'PARTITION' in file.keys() and not overwrite: return
	if 'PARTITION' in file.keys() and overwrite: del file['PARTITION']
	group = file.create_group('PARTITION') 
	pdset = {}
	nsubd = ptable.n_partitions
	if is_field:
		# Write datasets for field
		pdset['NSubD']  = group.create_dataset('NSubD', (1,),    dtype='i')
		pdset['Ids']    = group.create_dataset('Ids',   (nsubd,),dtype='i')
		pdset['Points'] = group.create_dataset('Points',(nsubd,),dtype='i')
	else:
		# Write datasets for mesh
		pdset['NSubD']      = group.create_dataset('NSubD',     (1,),    dtype='i')
		pdset['Ids']        = group.create_dataset('Ids',       (nsubd,),dtype='i')
		pdset['Elements']   = group.create_dataset('Elements',  (nsubd,),dtype='i')
		pdset['Points']     = group.create_dataset('Points',    (nsubd,),dtype='i')
		pdset['Boundaries'] = group.create_dataset('Boundaries',(nsubd,),dtype='i')
	pdset['has_master'] = group.create_dataset('has_master',(1,),dtype='i')
	# Store
	if is_rank_or_serial(0):
		if 'NSubD'      in pdset.keys(): pdset['NSubD'][:]      = nsubd
		if 'Ids'        in pdset.keys(): pdset['Ids'][:]        = ptable.Ids
		if 'Elements'   in pdset.keys(): pdset['Elements'][:]   = ptable.Elements
		if 'Points'     in pdset.keys(): pdset['Points'][:]     = ptable.Points
		if 'Boundaries' in pdset.keys(): pdset['Boundaries'][:] = ptable.Boundaries
		if 'has_master' in pdset.keys(): pdset['has_master'][:] = ptable.has_master

def h5_load_partition_table(file,is_field=False):
	'''
	Load partition table data for field and mesh HDF5
	'''
	group = file['PARTITION']
	# Load common parts
	nsubd  = int(group['NSubD'][0])
	ids    = np.array(group['Ids'][:])
	points = np.array(group['Points'][:])
	# Load parts that are only stored in mesh
	if not is_field:
		elements   = np.array(group['Elements'][:])
		boundaries = np.array(group['Boundaries'][:])
	else:
		elements   = np.zeros((nsubd,),np.int32)
		boundaries = np.zeros((nsubd,),np.int32)
	has_master = bool(group['has_master'][0])
	# Return a proper partition table
	return PartitionTable(nsubd,ids,elements,points,boundaries,has_master=has_master)

def h5_edit_partition_table(fname,ptable):
	'''
	Edit the partition table of an existing HDF5 file.
	And by edit I mean change it for the input partition table
	'''
	file     = h5py.File(fname,'a')
	is_field = True if not 'MESH' in file.keys() else False
	h5_write_partition_table(file,ptable,is_field=is_field,overwrite=True)
	file.close()


@cr('h5IO.save_field')
def h5_save_field(fname,ptable,xyz,instant,time,varDict,mpio=True,write_master=False,metadata={}):
	'''
	Save a field in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		h5_save_field_mpio(fname,ptable,xyz,instant,time,varDict,write_master,metadata)
	else:
		h5_save_field_serial(fname,ptable,xyz,instant,time,varDict,metadata)

def h5_save_variables_field(file,ptable,xyz,varDict,npoints,write_master,serial=False):
	'''
	Store variables into an HDF5 file
	'''
	group = file.create_group('VARIABLES')
	# Generate the datasets
	dset  = {}
	dset['xyz'] = file.create_dataset('xyz',(npoints,xyz.shape[1]),dtype=xyz.dtype)
	for var in varDict.keys():
		v     = varDict[var]
		shape = (npoints,) if len(v.shape) == 1 else (npoints,v.shape[1])
		dset[var] = group.create_dataset(var,shape,dtype=v.dtype)
	# Write into the datasets
	if serial or MPI_RANK != 0 or write_master: # To be changed in NOMASTER
		# Obtain start and end points for the array
		istart,iend = ptable.partition_bounds(MPI_RANK if not serial else 0,'Points')
		# Store the data
		dset['xyz'][istart:iend,:] = xyz
		for var in varDict.keys():
			v = varDict[var]
			if len(v.shape) == 1: # Scalar field
				dset[var][istart:iend]   = v
			else: # Vectorial or tensorial field
				dset[var][istart:iend,:] = v

def h5_save_field_serial(fname,ptable,xyz,instant,time,varDict,metadata={},overwrite=True):
	'''
	Save a field in HDF5 in serial mode
	'''
	# Open file for writing
	file  = h5py.File(fname,'w' if not os.path.exists(fname) else 'a')
	if 'FIELD' in file.keys() and not overwrite: return
	if 'FIELD' in file.keys() and overwrite: del file['FIELD']  
	group = file.create_group('FIELD')
	h5_write_attributes(file)
	# Metadata
	h5_write_metadata(file,metadata)
	# Partition table
	ptable = PartitionTable.new(1,nelems=0,npoints=xyz.shape[0],has_master=False)
	h5_write_partition_table(file,ptable,is_field=True,overwrite=False)
	# Store number of points, instant and time
	group.create_dataset('npoints',(1,),dtype='i',data=xyz.shape[0])
	group.create_dataset('instant',(1,),dtype='i',data=instant)
	group.create_dataset('time'   ,(1,),dtype='f',data=time)
	# Store the variables
	h5_save_variables_field(group,ptable,xyz,varDict,xyz.shape[0],True,serial=True)
	file.close()

def h5_save_field_mpio(fname,ptable,xyz,instant,time,varDict,write_master=False,metadata={},overwrite=True):
	'''
	Save a field in HDF5 in parallel mode
	'''
	# Compute the total number of points
	npoints  = 0 if MPI_RANK == 0 and not write_master else xyz.shape[0]
	npointsG = int(mpi_reduce(npoints,op='sum',all=True))
	# Open file
	file  = h5py.File(fname,'w' if not os.path.exists(fname) else 'a',driver='mpio',comm=MPI_COMM)
	if 'FIELD' in file.keys() and not overwrite: return
	if 'FIELD' in file.keys() and overwrite: del file['FIELD']
	group = file.create_group('FIELD')
	h5_write_attributes(file)
	# Metadata
	h5_write_metadata(file,metadata)
	# Partition table
	h5_write_partition_table(file,ptable,is_field=True,overwrite=False)
	# Create groups for number of points, instant, time and coordinates
	dset = {}
	dset['npoints'] = group.create_dataset('npoints',(1,),dtype='i')
	dset['instant'] = group.create_dataset('instant',(1,),dtype='i')
	dset['time']    = group.create_dataset('time'   ,(1,),dtype='f')
	# Rank 0 stores npoints, instant and time
	if is_rank_or_serial(0):
		dset['npoints'][:] = npointsG
		dset['instant'][:] = instant
		dset['time'][:]    = time
	# Store the variables
	h5_save_variables_field(group,ptable,xyz,varDict,npointsG,write_master)
	file.close()


@cr('h5IO.load_field')
def h5_load_field(fname,mpio=True,vars=[],inods=None):
	'''
	Load a field in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		return h5_load_field_mpio(fname,vars,inods)
	else:
		return h5_load_field_serial(fname,vars,inods)

def h5_load_variables_field(file,ptable,npoints,varList,inods):
	'''
	Load variables from an HDF5 file
	'''
	# Redo the partition table
	if not ptable.check_split():
		ptable = PartitionTable.new(MPI_SIZE,nelems=0,npoints=npoints)
	# Use the partition table to get the indices to read
	inods = ptable.get_partition_points(MPI_RANK) if inods is None else inods
	# Load node coordinates
	xyz     = np.array(file['xyz'][inods,:])
	varDict = {}
	# Load the variables in the varDict
	group = file['VARIABLES']
	if len(varList) == 0: varList = list(group.keys())
	for var in varList:
		if len(group[var].shape) == 1:
			# Scalar field
			varDict[var] = np.array(group[var][inods])
		else:
			# Vectorial field
			varDict[var] = np.array(group[var][inods,:])
	# Return
	return xyz, varDict, ptable

def h5_load_field_serial(fname,varList,inods):
	'''
	Load a field in HDF5 in serial
	'''
	# Open file for reading
	file  = h5py.File(fname,'r')
	h5_check_attributes(file)
	group = file['FIELD']
	# Read partition table
	ptable = h5_load_partition_table(file,is_field=True)
	# Read number of points, instant and time
	npoints = int(group['npoints'][0])
	instant = int(group['instant'][0])
	time    = float(group['time'][0])
	# Read variables
	xyz, varDict, ptable = h5_load_variables_field(group,ptable,npoints,varList,inods if not inods is None else np.s_[:])
	file.close()
	return ptable, xyz, instant, time, varDict

def h5_load_field_mpio(fname,varList,inods):
	'''
	Load a field in HDF5 in parallel
	'''
	# Open file for reading
	file  = h5py.File(fname,'r',driver='mpio',comm=MPI_COMM)
	h5_check_attributes(file)
	group = file['FIELD']
	# Read partition table
	ptable = h5_load_partition_table(file,is_field=True)
	# Read number of points, instant and time
	npoints = int(group['npoints'][0])
	instant = int(group['instant'][0])
	time    = float(group['time'][0])
	# Read variables
	xyz, varDict, ptable = h5_load_variables_field(group,ptable,npoints,varList,inods)
	file.close()
	return ptable, xyz, instant, time, varDict


@cr('h5IO.save_mesh')
def h5_save_mesh(fname,xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ptable,ngauss,
	consmas,massm,mpio=True,write_master=False,metadata={}):
	'''
	Save a field in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		h5_save_mesh_mpio(fname,xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ptable,ngauss,
			consmas,massm,write_master,metadata)
	else:
		h5_save_mesh_serial(fname,xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ptable,ngauss,
			consmas,massm,metadata)

def h5_save_variables_mesh_serial(file,ptable,xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ngauss,consmas,massm,write_master):
	'''
	Store the mesh variables into an HDF5 file
	'''
	# Compute the total number of points and elements
	nnodeG = lninv.shape[0]
	nelemG = leinv.shape[0]
	# Number of nodes, elements and gauss points
	file.create_dataset('nnodes' ,(1,),dtype='i',data=nnodeG)
	file.create_dataset('nelems' ,(1,),dtype='i',data=nelemG)
	file.create_dataset('ngauss' ,(1,),dtype='i',data=ngauss)
	file.create_dataset('consmas',(1,),dtype='i',data=consmas)
	# Arrays
	file.create_dataset('xyz'  ,(nnodeG,xyz.shape[1])  ,dtype=xyz.dtype  ,data=xyz)
	file.create_dataset('lnods',(nelemG,lnods.shape[1]),dtype=lnods.dtype,data=lnods)
	file.create_dataset('ltype',(nelemG,)              ,dtype=ltype.dtype,data=ltype)
	file.create_dataset('lninv',(nnodeG,)              ,dtype=lninv.dtype,data=lninv)
	file.create_dataset('leinv',(nelemG,)              ,dtype=leinv.dtype,data=leinv)
	# Optional arrays
	write_lmast = nnodeG == lmast.shape[0]
	write_codno = nnodeG == codno.shape[0]
	write_exnor = nnodeG == exnor.shape[0]
	write_commu = nnodeG == commu.shape[0]
	file.create_dataset('lmast',(nnodeG,)              ,dtype=lmast.dtype,data=lmast) if write_lmast else None
	file.create_dataset('codno',(nnodeG,codno.shape[1]),dtype=codno.dtype,data=codno) if write_codno else None
	file.create_dataset('exnor',(nnodeG,exnor.shape[1]),dtype=exnor.dtype,data=exnor) if write_exnor else None
	file.create_dataset('commu',(nnodeG,commu.shape[1]),dtype=commu.dtype,data=commu) if write_commu else None
	if not consmas:
		write_massm = nnodeG == massm.shape[0]
		file.create_dataset('massm',(nnodeG,),dtype=massm.dtype,data=massm) if write_massm else None

def h5_save_variables_mesh_mpio(file,ptable,xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ngauss,consmas,massm,write_master):
	'''
	Store the mesh variables into an HDF5 file
	'''
	# Compute the total number of points and elements
	nnodeG = mpi_reduce(lninv.shape[0] if not np.all(np.isnan(xyz)) else 0.,op='sum',all=True)
	nelemG = mpi_reduce(leinv.shape[0] if not np.all(np.isnan(xyz)) else 0.,op='sum',all=True)
	dset   = {}
	# Number of nodes, elements and gauss points
	dset['nnodes']  = file.create_dataset('nnodes' ,(1,),dtype='i')
	dset['nelems']  = file.create_dataset('nelems' ,(1,),dtype='i')
	dset['ngauss']  = file.create_dataset('ngauss' ,(1,),dtype='i')
	dset['consmas'] = file.create_dataset('consmas',(1,),dtype='i')
	# Arrays
	dset['xyz']     = file.create_dataset('xyz'  ,(nnodeG,xyz.shape[1])  ,dtype=xyz.dtype)
	dset['lnods']   = file.create_dataset('lnods',(nelemG,lnods.shape[1]),dtype=lnods.dtype)
	dset['ltype']   = file.create_dataset('ltype',(nelemG,)              ,dtype=ltype.dtype)
	dset['lninv']   = file.create_dataset('lninv',(nnodeG,)              ,dtype=lninv.dtype)
	dset['leinv']   = file.create_dataset('leinv',(nelemG,)              ,dtype=leinv.dtype)
	# Optional arrays
	write_lmast     = nnodeG == mpi_reduce(lmast.shape[0] if not np.all(np.isnan(xyz)) else 0.,op='sum',all=True)
	write_codno     = nnodeG == mpi_reduce(codno.shape[0] if not np.all(np.isnan(xyz)) else 0.,op='sum',all=True)
	write_exnor     = nnodeG == mpi_reduce(exnor.shape[0] if not np.all(np.isnan(xyz)) else 0.,op='sum',all=True)
	write_commu     = nnodeG == mpi_reduce(commu.shape[0] if not np.all(np.isnan(xyz)) else 0.,op='sum',all=True)
	dset['lmast']   = file.create_dataset('lmast',(nnodeG,)              ,dtype=lmast.dtype) if write_lmast else None
	dset['codno']   = file.create_dataset('codno',(nnodeG,codno.shape[1]),dtype=codno.dtype) if write_codno else None
	dset['exnor']   = file.create_dataset('exnor',(nnodeG,exnor.shape[1]),dtype=exnor.dtype) if write_exnor else None
	dset['commu']   = file.create_dataset('commu',(nnodeG,commu.shape[1]),dtype=commu.dtype) if write_commu else None
	if not consmas:
		write_massm   = nnodeG == mpi_reduce(massm.shape[0] if not np.all(np.isnan(xyz)) else 0.,op='sum',all=True)
		dset['massm'] = file.create_dataset('massm',(nnodeG,),dtype=massm.dtype) if write_massm else None
	# Write the number of nodes, elements and gauss points
	if is_rank_or_serial(0):
		# Metadata
		dset['nnodes'][:]  = nnodeG
		dset['nelems'][:]  = nelemG
		dset['ngauss'][:]  = ngauss
		dset['consmas'][:] = consmas
	if MPI_RANK != 0 or write_master:
		# Partition for point arrays
		istart, iend = ptable.partition_bounds(MPI_RANK,points=True)
		# Store point arrays
		dset['xyz'][istart:iend,:]   = xyz
		dset['lninv'][istart:iend]   = lninv
		# Optional arrays
		if dset['lmast'] is not None: dset['lmast'][istart:iend]   = lmast
		if dset['codno'] is not None: dset['codno'][istart:iend,:] = codno
		if dset['exnor'] is not None: dset['exnor'][istart:iend,:] = exnor
		if dset['commu'] is not None: dset['commu'][istart:iend,:] = commu
		if dset['massm'] is not None: dset['massm'][istart:iend]   = massm
		# Partition for element arrays
		istart, iend = ptable.partition_bounds(MPI_RANK,points=False)
		# Store element arrays
		dset['lnods'][istart:iend,:] = lnods + istart # To global ids
		dset['ltype'][istart:iend]   = ltype
		dset['leinv'][istart:iend]   = leinv

def h5_save_mesh_serial(fname,xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ptable,ngauss,
	consmas,massm,metadata={},overwrite=True):
	'''
	Save a mesh in HDF5 in serial mode
	'''
	# Open file for writing
	file  = h5py.File(fname,'w' if not os.path.exists(fname) else 'a')
	if 'MESH' in file.keys() and not overwrite: return
	if 'MESH' in file.keys() and overwrite: del file['MESH']
	group = file.create_group('MESH')
	h5_write_attributes(file)
	# Metadata
	h5_write_metadata(file,metadata)
	# Partition table
	ptable = PartitionTable.new(1,nelems=lnods.shape[0],npoints=xyz.shape[0],has_master=False)
	h5_write_partition_table(file,ptable,is_field=False)
	# Mesh data
	h5_save_variables_mesh_serial(group,ptable,xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ngauss,consmas,massm,True)
	file.close()

def h5_save_mesh_mpio(fname,xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ptable,ngauss,
	consmas,massm,write_master=False,metadata={},overwrite=True):
	'''
	Save a mesh in HDF5 in parallel mode
	'''
	# Open file
	file  = h5py.File(fname,'w' if not os.path.exists(fname) else 'a',driver='mpio',comm=MPI_COMM)
	if 'MESH' in file.keys() and not overwrite: return
	if 'MESH' in file.keys() and overwrite: del file['MESH']  
	group = file.create_group('MESH')
	h5_write_attributes(file)
	# Metadata
	h5_write_metadata(file,metadata)
	# Partition table
	h5_write_partition_table(file,ptable,is_field=False)
	# Mesh data
	h5_save_variables_mesh_mpio(group,ptable,xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ngauss,consmas,massm,write_master)
	file.close()


@cr('h5IO.load_mesh')
def h5_load_mesh(fname,mpio=True):
	'''
	Load a mesh in HDF5
	'''
	if mpio and not MPI_SIZE == 1:
		return h5_load_mesh_mpio(fname)
	else:
		return h5_load_mesh_serial(fname)

def h5_load_variables_mesh(file,ptable):
	'''
	Load variables from an HDF5 file
	'''
	repart  = False
	# Read the number of points, elements, Gauss points and consmas
	npoints = int(file['nnodes'][0])
	nelems  = int(file['nelems'][0])
	ngauss  = int(file['ngauss'][0])
	consmas = bool(file['consmas'][0])
	# If necessary, redo the partition table
	if not ptable.check_split():
		repart = True
		print(MPI_SIZE,nelems,npoints,ptable.has_master)
		ptable = PartitionTable.new(MPI_SIZE,nelems=nelems,npoints=npoints,has_master=ptable.has_master)
	# If we repartitioned we have a correct element partition
	# however the node partition will not be correct
	# Read element arrays
	istart, iend = ptable.partition_bounds(MPI_RANK, 'Elements')
	lnods = np.array(file['lnods'][istart:iend,:]) - istart
	ltype = np.array(file['ltype'][istart:iend])
	leinv = np.array(file['leinv'][istart:iend])
	# If necessary, update partition points
	if repart:
		inods = ptable.update_partition_points(MPI_RANK,npoints,conec=lnods+istart)
		ptable.update_points(inods.shape[0])
	# Read point arrays
	inods = ptable.get_partition_points(MPI_RANK)
	xyz   = np.array(file['xyz'][inods,:])
	lninv = np.array(file['lninv'][inods])
	lmast = np.array(file['lmast'][inods])   if 'lmast' in file.keys() else np.array([],np.int32)
	codno = np.array(file['codno'][inods,:]) if 'codno' in file.keys() else np.array([[]],np.int32)
	exnor = np.array(file['exnor'][inods,:]) if 'exnor' in file.keys() else np.array([[]],np.double)
	commu = np.array(file['commu'][inods,:]) if 'commu' in file.keys() else np.array([[]],np.int32)
	massm = np.array(file['massm'][inods])   if 'massm' in file.keys() else np.array([],np.double)
	# Return
	return xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ngauss,consmas,massm,ptable

def h5_load_mesh_serial(fname):
	'''
	Load a mesh in HDF5 in serial mode
	'''
	# Open file for reading
	file    = h5py.File(fname,'r')
	h5_check_attributes(file)
	group = file['MESH']
	# Read partition table
	ptable = h5_load_partition_table(file,is_field=False)
	xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ngauss,consmas,massm,ptable = h5_load_variables_mesh(group,ptable)
	file.close()
	return xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ptable,ngauss,consmas,massm

def h5_load_mesh_mpio(fname):
	'''
	Load a mesh in HDF5 in parallel using a pre computed partition
	'''
	# Open file for reading
	file = h5py.File(fname,'r',driver='mpio',comm=MPI_COMM)
	h5_check_attributes(file)
	group = file['MESH']
	# Read partition table
	ptable = h5_load_partition_table(file,is_field=False)
	xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ngauss,consmas,massm,ptable = h5_load_variables_mesh(group,ptable)
	file.close()
	return xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ptable,ngauss,consmas,massm
