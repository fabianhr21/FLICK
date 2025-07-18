#!/usr/bin/env python
#
# VTKH5 Input Output
#
# Last rev: 28/10/2022
from __future__ import print_function, division

import numpy as np, h5py

from ..cr             import cr
from ..utils.common   import raiseError
from ..utils.parallel import MPI_RANK, MPI_SIZE, MPI_COMM, writesplit, mpi_reduce, mpi_bcast

VTKTYPE = np.bytes_('UnstructuredGrid')
VTKVERS = np.array([1,0],np.int32)

alya2VTKCellTypes ={
	# Linear cells
	# Ref: https://github.com/Kitware/VTK/blob/master/Common/DataModel/vtkCellType.h
	-1 : 0 , # Empty cell
	 2 : 3 , # Line element
	 4 : 68, # Lagrangian curve
	10 : 5 , # Triangular cell
	12 : 9 , # Quadrangular cell
	15 : 70 , # Lagrangian quadrangle
	30 : 10, # Tetrahedral cell
	37 : 12, # Hexahedron
	34 : 13, # Linear prism
	32 : 14, # Pyramid
	39 : 29, # Triquadratic hexahedron 
	40 : 72, # Lagrangian hexahedron
}

def _vtkh5_create_structure(file):
	'''
	Create the basic structure of a VTKH5 file
	'''
	# Create main group
	main = file.create_group('VTKHDF')
	main.attrs['Type']    = VTKTYPE
	main.attrs['Version'] = VTKVERS
	# Create cell data group
	cdata = main.create_group('CellData')
	pdata = main.create_group('PointData')
	fdata = main.create_group('FieldData')
	# Return created groups
	return main, cdata, pdata, fdata

def _vtkh5_connectivity_and_offsets(lnods):
	'''
	Build the offsets array (starting point per each element)

	'''
	# Compute the number of points per cell
	ppcell = np.sum(lnods >= 0,axis=1)
	# First we flatten the connectivity array
	lnodsf = lnods.flatten('c')
	# Now we get rid of any -1 entries for mixed meshes
	lnodsf = lnodsf[lnodsf>=0]
	# Now build the offsets vector
	offset = np.zeros((ppcell.shape[0]+1,),np.int32)
	offset[1:] = np.cumsum(ppcell)
	return lnodsf, offset

def _vtkh5_write_mesh_serial(file,xyz,lnods,ltype):
	'''
	Write the mesh and the connectivity to the VTKH5 file.
	'''
	# Create dataset for number of points
	npoints, ndim = xyz.shape
	file.create_dataset('NumberOfPoints',(1,),dtype=int,data=npoints)
	d = file.create_dataset('Points',(npoints,3),dtype=np.double)
	d[:,:ndim] = xyz
	# Create dataset for number of cells
	lnods, offsets = _vtkh5_connectivity_and_offsets(lnods)
	ncells = ltype.shape[0]
	ncsize = lnods.shape[0]
	file.create_dataset('NumberOfCells',(1,),dtype=int,data=ncells)
	file.create_dataset('NumberOfConnectivityIds',(1,),dtype=int,data=ncsize)
	file.create_dataset('Connectivity',(ncsize,),dtype=int,data=lnods)
	file.create_dataset('Offsets',(ncells+1,),dtype=int,data=offsets)
	file.create_dataset('Types',(ncells,),dtype=np.uint8,data=np.array([alya2VTKCellTypes[t] for t in ltype], np.uint8))
	# Return some parameters
	return npoints, ncells 

def _vtkh5_write_mesh_mpio(file,xyz,lnods,ltype,write_master):
	'''
	Write the mesh and the connectivity to the VTKH5 file.
	'''
	myrank  = MPI_RANK - 1 if not write_master else MPI_RANK # Discard the master if needed
	nparts  = MPI_SIZE - 1 if not write_master else MPI_SIZE # Number of partitions discarding master (if necessary)
	# Create datasets for point data
	istartp, iendp = writesplit(xyz.shape[0],write_master)
	npoints = 0 if MPI_RANK == 0 and not write_master else xyz.shape[0] # Number of points of this partition
	ndim    = xyz.shape[1]
	npG     = int(mpi_reduce(npoints,op='sum',all=True)) # Total number of points
	npoints_dset = file.create_dataset('NumberOfPoints',(nparts,),dtype=int)
	points_dset  = file.create_dataset('Points',(npG,3),dtype=np.double) # Only works for 3D
	# Create datasets for cell data
	lnods, offsets = _vtkh5_connectivity_and_offsets(lnods)
	ncells  = 0 if MPI_RANK == 0 and not write_master else ltype.shape[0]
	ncG     = int(mpi_reduce(ncells,op='sum',all=True))
	ncsize  = 0 if MPI_RANK == 0 and not write_master else lnods.shape[0]
	nsG     = int(mpi_reduce(ncsize,op='sum',all=True))
	istartc,iendc = writesplit(ltype.shape[0],write_master)
	istarts,iends = writesplit(lnods.shape[0],write_master)
	ncells_dset = file.create_dataset('NumberOfCells',(nparts,),dtype=int)
	nids_dset   = file.create_dataset('NumberOfConnectivityIds',(nparts,),dtype=int)
	conec_dset  = file.create_dataset('Connectivity',(nsG,),dtype=int)
	offst_dset  = file.create_dataset('Offsets',(ncG+nparts,),dtype=int)
	types_dset  = file.create_dataset('Types',(ncG,),dtype=np.uint8)
	# Each partition writes its own part
	if MPI_RANK != 0 or write_master:
		# Point data
		npoints_dset[myrank]             = npoints
		points_dset[istartp:iendp,:ndim] = xyz
		# Cell data
		ncells_dset[myrank] = ncells
		nids_dset[myrank]   = ncsize
		conec_dset[istarts:iends] = lnods
		offst_dset[istartc+myrank:iendc+(myrank+1)] = offsets
		types_dset[istartc:iendc] = np.array([alya2VTKCellTypes[t] for t in ltype], np.uint8)
	# Return some parameters
	return npG, ncG

def _vtkh5_link_mesh(file,lname):
	'''
	Create external link mesh to the VTKH5 file.
	'''
	file['NumberOfPoints']          = h5py.ExternalLink(lname,'VTKHDF/NumberOfPoints')
	file['NumberOfCells']           = h5py.ExternalLink(lname,'VTKHDF/NumberOfCells')
	file['NumberOfConnectivityIds'] = h5py.ExternalLink(lname,'VTKHDF/NumberOfConnectivityIds')
	file['Points']                  = h5py.ExternalLink(lname,'VTKHDF/Points')
	file['Connectivity']            = h5py.ExternalLink(lname,'VTKHDF/Connectivity')
	file['Offsets']                 = h5py.ExternalLink(lname,'VTKHDF/Offsets')
	file['Types']                   = h5py.ExternalLink(lname,'VTKHDF/Types')


@cr('vtkh5IO.save_mesh')
def vtkh5_save_mesh(fname,xyz,lnods,ltype,mpio=True,write_master=False):
	'''
	Save the mesh component into a VTKH5 file
	'''
	if mpio and not MPI_SIZE == 1:
		vtkh5_save_mesh_mpio(fname,xyz,lnods,ltype,write_master)
	else:
		vtkh5_save_mesh_serial(fname,xyz,lnods,ltype)

def vtkh5_save_mesh_serial(fname,xyz,lnods,ltype):
	'''
	Save the mesh component into a VTKH5 file (serial)
	'''
	# Open file for writing
	file = h5py.File(fname,'w')
	# Create the file structure
	main, _, _, _ = _vtkh5_create_structure(file)
	# Write the mesh
	_vtkh5_write_mesh_serial(main,xyz,lnods,ltype)
	# Close file
	file.close()

def vtkh5_save_mesh_mpio(fname,xyz,lnods,ltype,write_master):
	'''
	Save the mesh component into a VTKH5 file (serial)
	'''
	# Open file for writing
	file = h5py.File(fname,'w',driver='mpio',comm=MPI_COMM)
	# Create the file structure
	main, _, _, _ = _vtkh5_create_structure(file)
	# Write the mesh
	_vtkh5_write_mesh_mpio(main,xyz,lnods,ltype,write_master)
	# Close file
	file.close()


@cr('vtkh5IO.link_mesh')
def vtkh5_link_mesh(fname,lname,mpio=True):
	'''
	Link the mesh component into a VTKH5 file
	'''
	if mpio and not MPI_SIZE == 1:
		vtkh5_link_mesh_mpio(fname,lname)
	else:
		vtkh5_link_mesh_serial(fname,lname)

def vtkh5_link_mesh_serial(fname,lname):
	'''
	Save the mesh component into a VTKH5 file (serial)
	'''
	# Open file for writing
	file = h5py.File(fname,'w')
	# Create the file structure
	main, _, _, _ = _vtkh5_create_structure(file)
	# Link the mesh
	_vtkh5_link_mesh(main,lname)
	# Close file
	file.close()

def vtkh5_link_mesh_mpio(fname,lname):
	'''
	Save the mesh component into a VTKH5 file (serial)
	'''
	# Open file for writing
	file = h5py.File(fname,'w',driver='mpio',comm=MPI_COMM)
	# Create the file structure
	main, _, _, _ = _vtkh5_create_structure(file)
	# Link the mesh
	_vtkh5_link_mesh(main,lname)
	# Close file
	file.close()


@cr('vtkh5IO.save_field')
def vtkh5_save_field(fname,instant,time,varDict,mpio=True,write_master=False):
	'''
	Save the mesh component into a VTKH5 file
	'''
	if mpio and not MPI_SIZE == 1:
		vtkh5_save_field_mpio(fname,instant,time,varDict,write_master)
	else:
		vtkh5_save_field_serial(fname,instant,time,varDict)

def vtkh5_save_field_serial(fname,instant,time,varDict):
	'''
	Save the field component into a VTKH5 file (serial)
	'''
	# Open file for writing (append to a mesh)
	file = h5py.File(fname,'a')
	main = file['VTKHDF']
	npoints = int(main['NumberOfPoints'][0])
	# Write dt and instant as field data
	main['FieldData'].create_dataset('InstantValue',(1,),dtype=int,data=instant)
	main['FieldData'].create_dataset('TimeValue',(1,),dtype=float,data=time)
	# Write the variables
	for var in varDict.keys():
		# Obtain in which group to write
		group = 'PointData' if varDict[var].shape[0] == npoints else 'CellData'
		# Create and write
		main[group].create_dataset(var,varDict[var].shape,dtype=varDict[var].dtype,data=varDict[var])
	# Close file
	file.close()

def vtkh5_save_field_mpio(fname,instant,time,varDict,write_master):
	'''
	Save the mesh component into a VTKH5 file (serial)
	'''
	myrank = MPI_RANK - 1 if not write_master else MPI_RANK     # Discard the master if needed
	# Open file for writing
	file = h5py.File(fname,'a',driver='mpio',comm=MPI_COMM)
	main = file['VTKHDF']
	npoints = int(main['NumberOfPoints'][myrank])
	# Write dt and instant as field data
	main['FieldData'].create_dataset('InstantValue',(1,),dtype=int,data=instant)
	main['FieldData'].create_dataset('TimeValue',(1,),dtype=float,data=time)
	# Create the dictionaries
	dsets = {}
	for var in varDict.keys():
		group   = mpi_bcast('PointData' if varDict[var].shape[0] == npoints else 'CellData',root=1)
		npoints = 0 if MPI_RANK == 0 and not write_master else varDict[var].shape[0] # Number of points of this partition
		npG     = int(mpi_reduce(npoints,op='sum',all=True)) # Total number of points
		nsize   = 0 if len(varDict[var].shape) == 1 or (MPI_RANK == 0 and not write_master) else varDict[var].shape[1]
		nsizeG  = int(mpi_reduce(nsize,op='max',all=True))
		dsets[var] = main[group].create_dataset(var,(npG,) if nsizeG == 0 else (npG,nsizeG),dtype=varDict[var].dtype)
	# Write the variables
	if MPI_RANK != 0 or write_master:
		for var in varDict.keys():
			istart,iend = writesplit(varDict[var].shape[0],write_master)
			if len(varDict[var].shape) == 1: # Scalar field
				dsets[var][istart:iend] = varDict[var]
			else: # Vectorial or tensorial field
				dsets[var][istart:iend,:] = varDict[var]
	# Close file
	file.close()