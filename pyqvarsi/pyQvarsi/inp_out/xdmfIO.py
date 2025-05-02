#!/usr/bin/env python
#
# PKL Input Output
#
# Last rev: 08/11/2021
from __future__ import print_function, division

import numpy as np

from ..cr             import cr
from ..utils.parallel import is_rank_or_serial


ELTYPE = {
	10 : 'Triangle',
	12 : 'Quadrilateral',
	30 : 'Tetrahedron',
	32 : 'Pyramid',
	34 : 'Wedge',
	37 : 'Hexahedron',
}

ELNOD = {
	10 : 3,
	12 : 4,
	30 : 4,
	32 : 5,
	34 : 6,
	37 : 8,
}


@cr('xdmfIO.save')
def xdmf_save(fname,fileDict,timeVec,gridname='DataGrid',write_rank=0):
	'''
	Create an XDMF file to be visualized in ParaView.
	'''
	# Set some defaults
	if not 'ndim' in fileDict['mesh'].keys(): fileDict['mesh']['ndim'] = 3
	# Only the specified rank writes this file
	if is_rank_or_serial(write_rank):
		# Open file for reading
		f     = open(fname,'w')
		# Write header
		f.write('<Xdmf Version="3.0">\n')
		f.write('<Domain>\n')
		f.write('<Grid CollectionType="Temporal" GridType="Collection">\n')
		# Write timesteps
		for itime,time in enumerate(timeVec):
			f.write('<Grid CollectionType="Spatial" GridType="Collection" Name="Mesh">\n')
			f.write('<Grid Name="%s">\n'%gridname)
			# Write connectivity
			f.write('<Topology NumberOfElements="%d" TopologyType="%s">\n'%(fileDict['mesh']['nel'],ELTYPE[fileDict['mesh']['type']]))
			f.write('<DataItem DataType="Int" Dimensions="%d %d" Format="HDF">\n'%(fileDict['mesh']['nel'],ELNOD[fileDict['mesh']['type']]))
			f.write('%s:/lnods\n'%(fileDict['mesh']['file']))
			f.write('</DataItem>\n')
			f.write('</Topology>\n')
			# Write node coordinates
			f.write('<Geometry GeometryType="XYZ">\n')
			f.write('<DataItem DataType="Float" Dimensions="%d %d" Format="HDF" Precision="8">\n'%(fileDict['mesh']['nnod'],fileDict['mesh']['ndim']))
			f.write('%s:/xyz\n'%(fileDict['mesh']['file']))
			f.write('</DataItem>\n')
			f.write('</Geometry>\n')
			# Write variables
			for varDict in fileDict['variables']:
				# Set variable type
				vtype = 'Matrix' # Generical MxN
				if varDict['ndim'] == 1: vtype = 'Scalar'
				if varDict['ndim'] == 3: vtype = 'Vector'
				if varDict['ndim'] == 6: vtype = 'Tensor6'
				if varDict['ndim'] == 9: vtype = 'Tensor'
				# Set number of points and dimensions
				npoints = fileDict['mesh']['nnod'] if varDict['type'] == 'Node' else fileDict['mesh']['nel']
				dims    = '%d' % npoints if varDict['ndim'] == 1 else '%d %d' % (npoints,varDict['ndim'])
				# Write file
				f.write('<Attribute AttributeType="%s" Center="%s" Name="%s">\n'%(vtype,varDict['type'],varDict['name']))
				f.write('<DataItem DataType="Float" Dimensions="%s" Format="HDF" Precision="8">\n'%dims)
				f.write('%s:/%s\n'%(varDict['file']%itime,varDict['name']))
				f.write('</DataItem>\n')
				f.write('</Attribute>\n')
			# Write temporal grid
			f.write('</Grid>\n')
			f.write('<Time TimeType="Single" Value="%f" />\n'%time)
			f.write('</Grid>\n')
		# Write ending
		f.write('</Grid>\n')
		f.write('</Domain>\n')
		f.write('</Xdmf>\n')
		f.close()