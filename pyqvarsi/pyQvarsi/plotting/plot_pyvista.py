#!/usr/bin/env python
#
# pyQvarsi, plotting.
#
# VTK pyVista plotting routines.
#
# Last rev: 25/01/2023
from __future__ import print_function, division

import numpy as np

from ..cr              import cr
from ..utils.common    import raiseWarning
from ..utils.parallel  import MPI_SIZE, MPI_RANK, mpi_gather
from ..mesh            import Mesh
from ..partition_table import PartitionTable

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

def _cells_and_offsets(lnods):
	'''
	Build the offsets and cells array to create am
	UnstructuredGrid
	'''
	# Compute points per cell
	ppcell = np.sum(lnods >= 0,axis=1)
	# Compute cells for pyVista, with the number of points per cell on front
	cells = np.c_[ppcell,lnods]
	# Now we get rid of any -1 entries for mixed meshes
	cellsf = cells.flatten('c')
	cellsf = cellsf[cellsf>=0]
	# Now build the offsets vector
	#offset = np.zeros((ppcell.shape[0]+1,),np.int32)
	#offset[1:] = np.cumsum(ppcell)
	offset = np.cumsum(ppcell)
	return cellsf, offset

def _polydataToMesh(polydata,dtype):
	'''
	Build pyAlya mesh from VTK polyData
	#TODO: Not necessarily class MeshSOD2D
	'''
	## Build pyAlya mesh from the polydata
	nels  = polydata.GetNumberOfCells()
	lnods = np.zeros((nels,3),np.int32) # Contours are made of triangles
	ltype = 10*np.ones((nels,),np.int32)
	for cellId in range(polydata.GetNumberOfCells()):
		# Get the cell
		cell = polydata.GetCell(cellId)   
		# Get the point IDs that make up this cell
		cellPointIds = cell.GetPointIds()
		for i in range(cellPointIds.GetNumberOfIds()):
			lnods[cellId,i] = cellPointIds.GetId(i)
	xyz      = polydata.points.astype(dtype)
	npoints  = xyz.shape[0]

	## Generate partition table
	ids      = np.arange(0,MPI_SIZE,1,np.int32)
	elpoints = np.array([nels, npoints]) 
	elpointG = mpi_gather(elpoints, all=True).reshape((MPI_SIZE,2))
	bounds   = np.zeros((MPI_SIZE),np.int32)
	ptable   = PartitionTable(MPI_SIZE, ids, elpointG[:,0], elpointG[:,1], bounds, has_master=False)
	pstart, pend = ptable.partition_bounds(MPI_RANK,'Points')
	lninv  = np.arange(pstart, pend, 1, dtype=np.int32)
	estart, eend = ptable.partition_bounds(MPI_RANK,'Elements')
	leinv  = np.arange(estart, eend, 1, dtype=np.int32)
	Qmesh  = Mesh(xyz, lnods, lninv, leinv, ltype, ptable=ptable, ngauss=1)
	return Qmesh

try:
	import pyvista as pv

	@cr('pyvista.plot')
	def pvplot(mesh,field,vars=[],**kwargs):
		'''
		Plot using pyVista
		'''
		# First create the unstructured grid
		cells, offsets = _cells_and_offsets(mesh.connectivity_vtk)
		# Create the types array
		types = np.array([alya2VTKCellTypes[t] for t in mesh.eltype],np.uint8)
		# Create the unstructured grid
		ugrid =  pv.UnstructuredGrid(offsets,cells,types,mesh.xyz) if pv.vtk_version_info < (9,) else pv.UnstructuredGrid(cells,types,mesh.xyz)
		# Load the variables inside the unstructured grid
		for v in vars: ugrid.point_data[v] = field[v]
		# Launch plot
		return ugrid.plot(**kwargs)
	
	@cr('pyvista.contour')
	def pvcontour(mesh,field,varcontour,valuecontour):
		'''
		Plot using pyVista
		'''
		# First create the unstructured grid
		cells, offsets = _cells_and_offsets(mesh.connectivity_vtk)
		# Create the types array
		types = np.array([alya2VTKCellTypes[t] for t in mesh.eltype_linear],np.uint8)
		# Create the unstructured grid
		ugrid = pv.UnstructuredGrid(offsets,cells,types,mesh.xyz) if pv.vtk_version_info < (9,) else pv.UnstructuredGrid(cells,types,mesh.xyz)
		if ugrid.GetNumberOfPoints() == 0:
			field[varcontour] = np.array([])
		## Do the contour
		ugrid.point_data[varcontour] = field[varcontour]
		contours = ugrid.contour(isosurfaces=[valuecontour], scalars=varcontour)
		## Convert polyData to pyAlya mesh
		contour_mesh = _polydataToMesh(contours)
		## Recover contour normals to improve visualization
		contours.compute_normals(cell_normals=False, point_normals=True, inplace=True)
		pointData = contours.GetPointData()
		xyz       = contours.points
		npoints   = xyz.shape[0]
		normals   = np.zeros((npoints,3), dtype=np.double)
		for i in range(pointData.GetNumberOfArrays()):
			array = pointData.GetArray(i)
			if array.GetName() == 'Normals':
				for ii in range(npoints):
					normals[ii,:] = array.GetTuple(ii)[0:3]
		return contour_mesh, normals
	
	@cr('pyvista.slice')
	def pvslice(mesh, direction, origin=[0,0,0]):
		'''
		Slice using pyVista
		'''
		# First create the unstructured grid
		cells, offsets = _cells_and_offsets(mesh.connectivity_vtk)
		# Create the types array
		types = np.array([alya2VTKCellTypes[t] for t in mesh.eltype_linear],np.uint8)
		# Create the unstructured grid
		ugrid  = pv.UnstructuredGrid(offsets,cells,types,mesh.xyz) if pv.vtk_version_info < (9,) else pv.UnstructuredGrid(cells,types,mesh.xyz)
		sliced = ugrid.slice(normal=direction, origin=origin, generate_triangles=True)
		## Convert polyData to pyAlya mesh
		slice_mesh = _polydataToMesh(sliced,mesh.xyz.dtype)

		return slice_mesh
except:
	def pvplot(mesh,field,vars=[],**kwargs):
		'''
		Plot using pyVista
		'''
		raiseWarning('Import - Problems loading pyVista!',all=False)

	def pvcontour(mesh,field,varcontour,valuecontour):
		'''
		Contour using pyVista
		'''
		raiseWarning('Import - Problems loading pyVista!',all=False)

	def pvslice(mesh, direction, origin=[0,0,0]):
		'''
		Slice using pyVista
		'''
		raiseWarning('Import - Problems loading pyVista!',all=False)