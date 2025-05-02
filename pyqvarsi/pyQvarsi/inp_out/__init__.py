#!/usr/bin/env python
#
# pyQvarsi, IO Module.
#
# Input/output routines for Alya
# mpio files and hdf5 database.
#
# Last rev: 01/10/2020

__VERSION__ = 3.1

# Some definitions
MPIO_PARTFILE_FMT   = '%s.post.alyapar'
MPIO_AUXFILE_P_FMT  = '%s-%s.post.mpio.bin'
MPIO_BINFILE_P_FMT  = '%s-%s-%08d.post.mpio.bin'
MPIO_AUXFILE_S_FMT  = '%s-%s.mpio.bin'
MPIO_BINFILE_S_FMT  = '%s-%s-%08d.mpio.bin'
MPIO_XFLFILE_S_FMT  = '%s-XFIEL.%08d.%08d.mpio.bin'
MPIO_NOTEMP_VARS    = ['EXNOR'] # Variables that don't have a temporal dependency

# Import mpio routines
from .AlyaMPIO import AlyaMPIO_header, AlyaMPIO_readPartitionTable, AlyaMPIO_writePartitionTable, AlyaMPIO_read, AlyaMPIO_read_serial, AlyaMPIO_readByChunk, AlyaMPIO_readByChunk_serial, AlyaMPIO_write, AlyaMPIO_write_serial, AlyaMPIO_writeByChunk, AlyaMPIO_writeByChunk_serial, AlyaMPIO_writeHeader
# Import Ensight routines
from .EnsightIO import Ensight_readCase, Ensight_writeCase, Ensight_readGeo, Ensight_writeGeo, Ensight_readField, Ensight_writeField
# Import pkl routies
from .pklio import pkl_load, pkl_save

SOD2DHDF_MESH_FMT = '%s-%d.hdf'
SOD2DHDF_RESULTS_FMT = 'results_%s-%d_%d.hdf'
# Import SOD2D routines
from .sod2dIO import SOD2DHDF_readPartitionTable, SOD2DHDF_read_mesh, SOD2DHDF_read_results
del sod2dIO

del AlyaMPIO, EnsightIO, pklio


## Import IO using HDF5
try:
	VTKH5_FILE_FMT = '%s-vtk.hdf'
	# Import hdf5 routies
	from .h5io    import h5_save_field, h5_load_field, h5_save_mesh, h5_load_mesh, h5_edit_partition_table
	from .xdmfIO  import xdmf_save
	from .vtkh5io import vtkh5_save_mesh, vtkh5_link_mesh, vtkh5_save_field
	# Import HiFiTurb routines
	from .HiFiTurbReader import HiFiTurbDB_Reader
	from .HiFiTurbWriter import HiFiTurbDB_Writer

	del h5io, vtkh5io, HiFiTurbReader, HiFiTurbWriter
except:
	from ..utils.common import raiseWarning
	raiseWarning('Import - Problems loading HDF5 IO!',all=False)
	del raiseWarning


## Import VTK routines
try:
	from .VTKIO import vtkIO, vtkCelltoPointData, vtk_to_numpy
	del VTKIO
except:
	from ..utils.common import raiseWarning
	raiseWarning('Import - Problems loading VTK IO!',all=False)
	del raiseWarning