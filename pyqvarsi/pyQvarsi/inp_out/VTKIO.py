#!/usr/bin/env python
#
# VTK IO module
#
# ToDo:
# [-] Tidy up vtkIO.info, separating VTK info and custom info.
# [-] get_vars_3D for cell data
# [-] Cell data connectivity and stuff for Alya
# [-] Partitioned VTK data
# [-] Parallel workers data
#
# Last rev: 28/09/2021
from __future__ import print_function, division

import warnings, numpy as np
from scipy.interpolate import griddata

import vtk as VTK2
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from ..mem import mem
from ..cr  import cr


vtk2AlyaCellTypes ={
	# Linear cells
	vtk.VTK_EMPTY_CELL      : -1,
#	vtk.VTK_VERTEX = 1,
#	vtk.VTK_POLY_VERTEX = 2,
	vtk.VTK_LINE            :  2,
#	vtk.VTK_POLY_LINE = 4,
	vtk.VTK_TRIANGLE        : 10,
#	vtk.VTK_TRIANGLE_STRIP = 6,
#	vtk.VTK_POLYGON = 7,
#	vtk.VTK_PIXEL = 8,
	vtk.VTK_QUAD            : 12,
	vtk.VTK_TETRA           : 30,
#	vtk.VTK_VOXEL = 11,
	vtk.VTK_HEXAHEDRON      : 37,
#	vtk.VTK_WEDGE = 13,
	vtk.VTK_PYRAMID         : 32,
#	vtk.VTK_PENTAGONAL_PRISM = 15,
	vtk.VTK_HEXAGONAL_PRISM : 34,
	# Quadratic, isoparametric cells
#	vtk.VTK_QUADRATIC_EDGE = 21,
#	vtk.VTK_QUADRATIC_TRIANGLE = 22,
#	vtk.VTK_QUADRATIC_QUAD = 23,
#	vtk.VTK_QUADRATIC_POLYGON = 36,
#	vtk.VTK_QUADRATIC_TETRA = 24,
#	vtk.VTK_QUADRATIC_HEXAHEDRON = 25,
#	vtk.VTK_QUADRATIC_WEDGE = 26,
#	vtk.VTK_QUADRATIC_PYRAMID = 27,
#	vtk.VTK_BIQUADRATIC_QUAD = 28,
	vtk.VTK_TRIQUADRATIC_HEXAHEDRON : 39,
#	vtk.VTK_TRIQUADRATIC_PYRAMID = 37,
#	vtk.VTK_QUADRATIC_LINEAR_QUAD = 30,
#	vtk.VTK_QUADRATIC_LINEAR_WEDGE = 31,
#	vtk.VTK_BIQUADRATIC_QUADRATIC_WEDGE = 32,
#	vtk.VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33,
#	vtk.VTK_BIQUADRATIC_TRIANGLE = 34,
}


class vtkIO(object):
	'''
	IO class for VTK files
	'''
#	@mem('vtkIO')
	def __init__(self,filename,mode='read',varlist=None):
		'''
		filename: path of the vtk file(s)
		varlist: list of variables to keep (None = all)
		'''
		self._fname   = filename
		self._mode    = mode
		self._varlist = varlist

		# init public class properties
		self.datatype = filename.split('.')[-1]
		self.reader   = None
		self.data     = None

		# read data, if requested
		if self._mode == 'read': self.read()

	def __str__(self):
		out  = 'Dataset <%s> (%s), variables: %s\n' % (self._fname,self._mode,self._varlist.__str__())
		info = self.info
		for key in info.keys():
			out += '%s: %s\n' % (key,info[key].__str__())
		return out

	@cr('vtkIO.read')
	def read(self):
		'''
		Read VTK data
		'''
		reader_case = {
			'vti': vtk.vtkXMLImageDataReader(),
			'vtr': vtk.vtkXMLRectilinearGridReader(),
			'vtu': vtk.vtkXMLUnstructuredGridReader(),
			# 'pvti': vtk.vtkXMLPImageDataReader(),
			# 'pvtr': vtk.vtkXMLPUnstructuredGridReader(),
			'pvtu': vtk.vtkXMLPUnstructuredGridReader(),
		}
		self.reader = reader_case[self.datatype]
		self.reader.SetFileName(self._fname)
		self.reader.Update()
		self.data = self.reader.GetOutput()

	@cr('vtkIO.get_vars')
	def get_vars(self,varlist=None):
		'''
		Returns dictionary of numpy arrays with the requestes 
		varlist fields for both point and cell data.
		
		varlist: Array name or list of array names, which can only 
		be equal to self._varlist or a subset.
		'''
		out = {}
		varlist = varlist if varlist is not None else self._varlist
		for varname,var in self.info['point_data'].items():
			if varname in varlist:
				x = self.data.GetPointData().GetAbstractArray(varname)
				name = x.GetName()+'_pointdata'
				out[name] = vtk_to_numpy(x)
		for varname,var in self.info['cell_data'].items():
			if varname in varlist:
				x = self.data.GetCellData().GetAbstractArray(varname)
				name = x.GetName()+'_celldata'
				out[name] = vtk_to_numpy(x)
		for varname in varlist:
			if varname not in self.info['point_data'] and varname not in self.info['cell_data']:
				warnings.warn('Could not find `{}` in the VTK data.'.format(varname))
		return out

	@cr('vtkIO.get_vars_3D')
	def get_vars_3D(self,varlist=None):
		'''
		Same as get_vars, but the point data is interpolated in a 3-D mesh.
		The cell data is ommited (To Do!)
		'''
		vars = self.get_vars(varlist)
		x_3d, y_3d, z_3d = self.mesh_3d
		out = {}
		for varname,var in vars.items():
			if '_celldata' in varname: continue
			out[varname] = griddata(self.points, var, (x_3d,y_3d,z_3d), method='nearest')
		return out

	@property
	def pointsVTK(self):
		return self.data.GetPoints().GetData()

	@property
	def points(self):
		return vtk_to_numpy(self.data.GetPoints().GetData()).astype(np.double)

	@property
	def mesh_3d(self):
		xline, yline, zline = np.unique(self.points[:, 0]), np.unique(self.points[:, 1]), np.unique(self.points[:, 2])
		return np.meshgrid(xline, yline, zline, indexing='ij')

	@property
	@cr('vtkIO.connectivity')
	def connectivity(self):
		cells        = self.data.GetCells()
		try:
			connectivity = vtk_to_numpy(cells.GetConnectivityArray())
			offset       = vtk_to_numpy(cells.GetOffsetsArray())
			out          = np.array_split(connectivity,offset)
			conec        = out[np.array([item.size>0 for item in out])]
		except:
			# Only working if the cell size is the same for all the mesh
			# https://vtk.org/doc/nightly/html/classvtkCellArray.html
			nncell = cells.GetMaxCellSize()
			conec  = vtk_to_numpy(cells.GetData())
			conec  = conec.reshape((conec.shape[0]//(nncell+1),nncell+1),order='C')[:,1:]
		return conec.astype(np.int32)

	@property
	def cellTypesVTK(self):
		return np.array([self.data.GetCell(i).GetCellType() for i in range(self.data.GetCells().GetNumberOfCells())],np.int32)

	@property
	def cellTypes(self):
		return np.array([vtk2AlyaCellTypes[self.data.GetCell(i).GetCellType()] for i in range(self.data.GetCells().GetNumberOfCells())],np.int32)

	@property
	@cr('vtkOIO.pointDataVTK')
	def pointDataVTK(self,varlist=None):
		varlist = varlist if varlist is not None else self._varlist
		point_data = {}
		pd = self.data.GetPointData()
		for i in range(pd.GetNumberOfArrays()):
			name = pd.GetAbstractArray(i).GetName()
			if name not in varlist: continue
			point_data[name] = pd.GetAbstractArray(i)
		return point_data

	@property
	@cr('vtkIO.cellDataVTK')
	def cellDataVTK(self,varlist=None):
		varlist = varlist if varlist is not None else self._varlist
		cell_data = {}
		cd = self.data.GetCellData()
		for i in range(cd.GetNumberOfArrays()):
			name = cd.GetAbstractArray(i).GetName()
			if name not in varlist: continue
			cell_data[name] = cd.GetAbstractArray(i)
		return cell_data

	@property
	@cr('vtkIO.info')
	def info(self):
		'''
		Information (metadata) regarding the dataset
		'''
		out  = {}
		out['point_data'] = {}
		for varname,vardata in self.pointDataVTK.items():
			out['point_data'][varname] = str(vardata)
		out['cell_data'] = {}
		for varname,vardata in self.cellDataVTK.items():
			out['cell_data'][varname] = str(vardata)
		return out


@cr('vtkIO.cell2PointData')
def vtkCelltoPointData(dataVTK,varlist,keepdata=False):
	'''
	Converts cell data arrays in `varlist` from the dataVTK object to point data.
	If `keepdata` is False, the dataVTK original cell data will remain, otherwise not.
	'''
	converter = vtk.vtkCellDataToPointData()
	converter.ProcessAllArraysOff()
	if keepdata: converter.PassCellDataOn()

	for varname in varlist:
		converter.AddCellDataArray(varname)
	converter.SetInputConnection(dataVTK.reader.GetOutputPort())
	converter.Update()

	point_data = converter.GetOutput().GetPointData()
	for i in range(point_data.GetNumberOfArrays()):
		varname = point_data.GetAbstractArray(i).GetName()
		dataVTK.data.GetPointData().AddArray(point_data.GetAbstractArray(i))
		if keepdata:
			cell_array = converter.GetOutput().GetCellData().GetAbstractArray(varname)
			dataVTK.data.GetCellData().AddArray(cell_array)
	return dataVTK