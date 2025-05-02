#!/usr/bin/env python
#
# pyQvarsi, Mesh.
#
# Small FEM module to compute derivatives and possible other
# simple stuff from Alya output for postprocessing purposes.
#
# Mesh structure to wrap arrays, based on Field class.
#
# Last rev: 20/10/2020
from __future__ import print_function, division

import os, numpy as np

from .                    import FEM, Geom, inp_out as io, vmath as math, solvers
from .communicator        import Communicator
from .partition_table     import PartitionTable
from .cr                  import cr
from .mem                 import mem
from .utils.common        import raiseWarning, raiseError
from .utils.parallel      import MPI_RANK, MPI_SIZE, is_master, is_rank_or_serial, mpi_reduce, mpi_gather, mpi_scatterp
from .meshing.mesh        import planeMesh, cubeMesh
from .meshing.interpolate import interpolateNearestNeighbour, interpolateFEM, interpolateFEMNearestNeighbour
from .FEM                 import defineHighOrderElement
from .field				  import FieldSOD2D
from .postproc.yplus      import computeWallDistancesSOD2D


class Mesh(object):
	'''
	The mesh class wraps the nodes, the elements and a number of variables
	and relates them so that the operations in parallel are easier.
	'''
	@cr('mesh.init')
	def __init__(self, xyz, lnods, lninv, leinv, ltype, ptable=None, ngauss=-1):
		'''
		Class constructor.

		IN:
			> xyz(nnod,3):            position of the nodes.
			> lnods(nelem,nnodxelem): connectivity matrix.
			> lninv(nnod):            global node numbering.
			> leinv(nelem):           global element numbering.
			> ltype(nelem):           type of element.
			> ptable:                 partition table.
		'''
		self._xyz      = xyz
		self._lnods    = lnods
		self._lninv    = lninv if not np.any(np.isnan(lninv)) else np.array([],dtype=lninv.dtype)
		self._leinv    = leinv if not np.any(np.isnan(leinv)) else np.array([],dtype=leinv.dtype)
		self._ltype    = ltype
		self._elemList = None
		self._ptable   = ptable
		self._ngauss   = ngauss

	def __len__(self):
		return self._xyz.shape[0]

	def __str__(self):
		'''
		String representation
		'''
		s   = 'Mesh of %d nodes and %d elements:\n' % (self.nnod,self.nel)
		s  += '  > xyz  - max = ' + str(np.nanmax(self._xyz,axis=0)) + ', min = ' + str(np.nanmin(self._xyz,axis=0)) + '\n'
		return s
	
	# Operators
	def __eq__(self, other):
		'''
		Mesh1 == Mesh2

		Two nodes are equal if all the coordinates
		are the same
		'''
		self._switch_to_elemList()
		other._switch_to_elemList()
		return np.all(self._xyz == other._xyz) and np.all(self._elemList == other._elemList)
	
	@cr('mesh.newArray')
	def newArray(self,ndim=0,dtype=np.double):
		'''
		Create a new array initialized to zeros.
		Use ndim = 0 for scalar arrays.
		'''
		return np.zeros((self.nnod ,),dtype=dtype) if ndim == 0 else np.zeros((self.nnod ,ndim),dtype=dtype)

	@cr('mesh.find_nodes')
	def find_nodes(self,xyz):
		'''
		Return all the nodes where self._xyz == xyz
		'''
		return np.where(np.all(self._xyz == xyz,axis=1))[0]

	@cr('mesh.find_node_in_elems')
	def find_node_in_elems(self,inode):
		'''
		Return all the elements where the node is
		'''
		return np.where(np.any(np.isin(self.connectivity,inode),axis=1))[0]
	
	@cr('mesh.find_elems')
	def find_elems(self,elem):
		'''
		Return all the elements where self._elemList == elem
		'''
		self._switch_to_elemList()
		return np.where(np.all(self._elemList == elem))[0]

	@cr('mesh.renumber')
	def renumber(self,nodes=True,elements=True,root=0):
		'''
		Renumber the nodes and elements on the mesh
		'''
		# Renumber algorithm for nodes
		if nodes:
			# Gather the points per partition and lninv to root 0
			npart = mpi_gather(len(self._lninv),root=root)
			lninv = mpi_gather(self._lninv,root=root)
			# Algorithm in serial on rank=root
			if is_rank_or_serial(root):
				# What we want to do at this point is a unique to lninv
				# and obtain the mask that orders back the points
				ulninv, mask = np.unique(lninv,return_inverse=True)
				# Now reconstruct the unique array from 0 to len(ulninv)
				ulninv = np.arange(0,len(ulninv),dtype=np.int32)
				# Finally, obtain lninv through mask
				lninv = ulninv[mask]
			# At this point lninv has been recreated
			# now scatter the array back to the processors
			self._lninv = mpi_scatterp(lninv,np.arange(0,MPI_SIZE),npart,root=root) if MPI_SIZE > 1 else lninv
		
		# Renumber algorithm for elements
		if elements:
			# Gather the points per partition and leinv to root 0
			npart = mpi_gather(len(self._leinv),root=root)
			leinv = mpi_gather(self._leinv,root=root)
			# Algorithm in serial on rank=root
			if is_rank_or_serial(root):
				# What we want to do at this point is a unique to leinv
				# and obtain the mask that orders back the points
				uleinv, mask = np.unique(leinv,return_inverse=True)
				# Now reconstruct the unique array from 0 to len(uleinv)
				uleinv = np.arange(0,len(uleinv),dtype=np.int32)
				# Finally, obtain leinv through mask
				leinv = uleinv[mask]
			# At this point leinv has been recreated
			# now scatter the array back to the processors
			self._leinv = mpi_scatterp(leinv,np.arange(0,MPI_SIZE),npart,root=root) if MPI_SIZE > 1 else leinv			

	@cr('mesh.partitions')
	def partitions(self,leinv,lninv,ptable=None,serial=False,has_master=False):
		'''
		Generate the partition table if it does not exist
		'''
		# Return the partition table if it is correct
		if not ptable is None: return ptable
		# Create new partition table and broadcast to
		# all the ranks
		self._ptable = PartitionTable.fromMesh(self,has_master=has_master) if not serial else PartitionTable.new(1,nelems=leinv.shape[0],npoints=lninv.shape[0],has_master=has_master)
		return self._ptable

	@cr('mesh.clip')
	def clip(self,poly):
		'''
		Given an entity of pyAlya.Geom, it returns a new Mesh and a mask with 
		the cropped values that are inside the region defined by the
		geometric element.		
		'''
		# Obtain a mask of the cell centers inside the polygon
		elmask  = poly.areinside(self.xyz_center)
		nodmask = np.zeros((self._xyz.shape[0],),dtype=bool)
		self._switch_to_connectivity()
		conec   = self.connectivity
		if np.any(elmask):
			# Obtain a mask for the nodes belonging to the selected elements
			cconec  = conec[elmask]
			rconec  = cconec.ravel()
			nodmask[np.unique(rconec)] = True
			# Obtained the nodes
			xyz = self._xyz[nodmask].copy()
			# Rebuild the mesh connectivity
			uni,rlnods = np.unique(rconec,return_inverse=True)
			lnods      = rlnods.reshape(cconec.shape).astype(np.int32).copy()
			ltype      = self._ltype[elmask].astype(np.int32).copy()
			if uni[0] == -1: lnods -= 1 # Multi-elements will have -1 as a value
		else:
			xyz   = np.empty((0,3),self.xyz.dtype) 
			lnods = np.empty((0,0),np.int32)
			ltype = np.empty((0),np.int32)
		# Recreate global numbering arrays
		lninv = np.copy(self._lninv[nodmask] if len(self._lninv) > 0 else self._lninv)
		leinv = np.copy(self._leinv[elmask]  if len(self._leinv) > 0 else self._leinv)
		# Obtain the minimum local value for the partititon
		lninv_loc = np.min(lninv) if np.any(elmask) else np.nan
		leinv_loc = np.min(leinv) if np.any(elmask) else np.nan
		# Subtract the minimum lninv and leinv of the partititon
		offst = mpi_reduce(lninv_loc,op='nanmin',all=True)
		if not np.isnan(offst): lninv -= int(offst)
		offst = mpi_reduce(leinv_loc,op='nanmin',all=True) ##TODO: OJO Q POT SER SUPER BUG
		if not np.isnan(offst): leinv -= int(offst)
		# Build and return the new mesh
		mesh_new = Mesh(xyz, lnods, lninv, leinv, ltype)
		return mesh_new, nodmask
	
	@cr('mesh.computeCellCenters')
	def computeCellCenters(self):
		'''
		Compute the positions of the cell centers.
		'''
		self._switch_to_elemList()
		return FEM.cellCenters(self._xyz,self._elemList)
	
	@cr('mesh.computeNormals')
	def computeNormals(self):
		'''
		Compute normals on the nodes of the elements
		TODO: Communicate normals of elements in the boarder between two ranks
		'''
		self._switch_to_elemList()
		norm  = np.zeros(self.xyz.shape, dtype=np.double)
		count = np.zeros((self.xyz.shape[0],), dtype=int)
		for elem in self._elemList:
			xyzel = self.xyz[elem.nodes]
			elnorm = elem.normal(xyzel)
			norm[elem.nodes,:] -= elnorm
			count[elem.nodes]  += 1
		norm[:,0] /= count
		norm[:,1] /= count
		norm[:,2] /= count

		return norm

	@cr('mesh.gradient')
	def gradient(self,f,on_Gauss=False,consistent=None):
		'''
		Compute the gradient of a field f on the current mesh.
		Optionally, obtain the gradients at the Gauss points.
		'''
		self._switch_to_elemList()
		if consistent is None: consistent = self._consmas
		if consistent and not self._consmas: consistent = False
		if on_Gauss:
			# Compute gradients on Gauss points
			ngaussT = self.ngaussT
			if self._xyz.shape[1] == 2:
				ndim = 2 if len(f.shape) == 1 else 2*f.shape[1]
				g = FEM.gradient2Dgp(self._xyz,f,self._elemList,ngaussT) if not np.all(self._elemList == None) else np.nan*np.zeros((self._xyz.shape[0],ndim))
			else:
				ndim = 3 if len(f.shape) == 1 else 3*f.shape[1]
				g = FEM.gradient3Dgp(self._xyz,f,self._elemList,ngaussT) if not np.all(self._elemList == None) else np.nan*np.zeros((self._xyz.shape[0],ndim))
		else:
			# Compute gradients on nodes
			if self._xyz.shape[1] == 2:
				ndim = 2 if len(f.shape) == 1 else 2*f.shape[1]
				g = FEM.gradient2D(self._xyz,f,self._elemList) if not np.all(self._elemList is None) else np.nan*np.zeros((self._xyz.shape[0],ndim))
			else:
				ndim = 3 if len(f.shape) == 1 else 3*f.shape[1]
				g = FEM.gradient3D(self._xyz,f,self._elemList) if not np.all(self._elemList is None) else np.nan*np.zeros((self._xyz.shape[0],ndim))
			# Apply linear solver
			g = solvers.solver_lumped(self.volume,g,commu=self._comm)
			if consistent: g = solvers.solver_approxInverse(self._vmass,self.volume,g,commu=self._comm)
		return g
	
	@cr('mesh.divergence')
	def divergence(self,f,on_Gauss=False,consistent=None):
		'''
		Compute the divergence of a field f
		on the current mesh.
		'''
		self._switch_to_elemList()
		if consistent is None: consistent = self._consmas
		if consistent and not self._consmas: consistent = False
		if on_Gauss:
			# Compute divergence on Gauss points
			ngaussT = self.ngaussT
			if self._xyz.shape[1] == 2:
				ndim = 0 if len(f.shape) == 1 else f.shape[1]//2
				d = FEM.divergence2Dgp(self._xyz,f,self._elemList,ngaussT) if not np.all(self._elemList == None) else np.nan*np.zeros((self._xyz.shape[0],ndim))
			else:
				ndim = 0 if len(f.shape) == 1 else f.shape[1]//3
				d = FEM.divergence3Dgp(self._xyz,f,self._elemList,ngaussT) if not np.all(self._elemList == None) else np.nan*np.zeros((self._xyz.shape[0],ndim))
		else:
			# Compute divergence on nodes
			if self._xyz.shape[1] == 2:
				ndim = 0 if len(f.shape) == 1 else f.shape[1]//2
				d = FEM.divergence2D(self._xyz,f,self._elemList) if not np.all(self._elemList is None) else np.nan*np.zeros((self._xyz.shape[0],ndim))
			else:
				ndim = 0 if len(f.shape) == 1 else f.shape[1]//3
				d = FEM.divergence3D(self._xyz,f,self._elemList) if not np.all(self._elemList is None) else np.nan*np.zeros((self._xyz.shape[0],ndim))
			# Apply linear solver
			d = solvers.solver_lumped(self.volume,d,commu=self._comm)
			if consistent: d = solvers.solver_approxInverse(self._vmass,self.volume,d,commu=self._comm)
		return d

	@cr('mesh.laplacian')
	def laplacian(self,f,on_Gauss=False,consistent=None):
		'''
		Compute the laplacian of a field f
		on the current mesh.
		'''
		self._switch_to_elemList()
		# Compute the gradient of f outside the Gauss points
		grad_f = super(self.__class__, self).gradient(f,on_Gauss=on_Gauss,consistent=consistent)
		# Compute the divergence of the gradient
		lap    = super(self.__class__, self).divergence(grad_f,on_Gauss=on_Gauss,consistent=consistent)
		return lap

	@cr('mesh.integral')
	def integral(self,f,mask=None,kind='surf'):
		'''
		Computes the integral of a field f on the current mesh
		'''
		self._switch_to_elemList()
		if mask is None: mask = np.ones((self._xyz.shape[0],),dtype=bool)
		if kind.lower() in ['surf','surface']:
			return mpi_reduce(FEM.integralSurface(self._xyz,f,mask,self._elemList),root=0,op='sum',all=True)
		if kind.lower() in ['vol','volume']:
			return mpi_reduce(FEM.integralVolume(self._xyz,f,mask,self._elemList),root=0,op='sum',all=True)
		raiseError('Integral kind <%s> not recognized!'%kind)

	@cr('mesh.interpolate')
	def interpolate(self,xyz,f,method='FEM',fact=2.0,f_incr=1.0,r_incr=1.0,
		global_max_iter=1,ball_max_iter=5,target_mask=None):		
		fun = interpolateFEM
		if method.lower() in ['nn','nearestneigbour','nearest','nearest_neigbour']: fun = interpolateNearestNeighbour
		if method.lower() in ['fem2','femnn','femnearestneigbour','femnearest']:    fun = interpolateFEMNearestNeighbour
		tf  = fun(self,xyz,f,fact=fact,f_incr=f_incr,r_incr=r_incr,
			global_max_iter=global_max_iter,ball_max_iter=ball_max_iter,target_mask=target_mask)
		# Return output field
		return tf

	@cr('mesh.massMatrix')
	def computeMassMatrix(self,consistent=False):
		'''
		Computes the Mass Matrix given the node positions
		and the list of elements.
		'''
		self._switch_to_elemList()
		if not np.all(self._elemList == None):
			if consistent:
				vmass = FEM.mass_matrix_consistent(self._xyz,self._elemList)
			else:
				vmass = FEM.mass_matrix_lumped(self._xyz,self._elemList)
				# Communicate the boundaries if there is a communicator
				if not self._comm == None:
					vmass = self._comm.communicate_scaf(vmass)
		else: 
			if consistent:
				vmass = math.csr_create(self._xyz.shape[0],1,np.double)
			else:
				vmass = np.array([],np.double)
		return vmass

	@cr('mesh.computeCellCenters')
	def computeCellCenters(self):
		'''
		Compute the positions of the cell centers.
		'''
		self._switch_to_elemList()
		return FEM.cellCenters(self._xyz,self._elemList)

	@cr('mesh.computeVolume')
	def computeVolume(self):
		'''
		Compute the volume of the elements at the nodes.
		'''
		if self._vmass is None: self._vmass = self.computeMassMatrix()
		# Return exception for the master
		if self._vmass.data.shape[0] == 0: return np.array([0.],np.double)
		# Return the lumped mass as the volume if not using
		# the consistent mass
		if not self._consmas: return self._vmass
		# Compute volume from the consistent mass
		v = np.squeeze(np.asarray(self._vmass.sum(axis=1)))
		# Add the contributions of neighbouring nodes
		if not self._comm == None:
			self._comm.communicate_scaf(v)
		return v
	
	@cr('mesh.write')
	def write(self,casestr,basedir='./',fmt='vtkhdf',nsub=max(MPI_SIZE-1,1),use_post=True,linkfile=None):
		'''
		Store the data in the mesh using various formats.

		This method differs from save in the fact that save is used
		to recover the mesh, write only outputs the data.
		'''
		self._switch_to_connectivity()
		if fmt.lower() == 'mpio':
			meshWriteMPIO(self,casestr,basedir,nsub,use_post)
		elif fmt.lower() in ['ensi','ensight']:
			meshWriteENSIGHT(self,casestr,basedir)
		elif fmt.lower() in ['vtkhdf','vtkh5']:
			meshWriteVTKH5(self,casestr,basedir,linkfile)
		else:
			raiseError('Format <%s> not implemented!'%fmt)

	@cr('mesh.save')
	def save(self,fname,**kwargs):
		'''
		Store the field in various formats.
		'''
		self._switch_to_connectivity()
		# Guess format from extension
		fmt = os.path.splitext(fname)[1][1:] # skip the .
		# Pickle format
		if fmt.lower() == 'pkl': io.pkl_save(fname,self)
		# HDF5 format
		if fmt.lower() == 'h5':
			# Set default parameters
			if not 'metadata'     in kwargs.keys(): kwargs['metadata']     = {}
			if not 'mpio'         in kwargs.keys(): kwargs['mpio']         = True
			if not 'write_master' in kwargs.keys(): kwargs['write_master'] = not self._ptable.has_master
			commu = self._comm.to_commu(self.nnod) if self._comm is not None else np.array([[]])
			io.h5_save_mesh(fname,self.xyz,self.connectivity,self._ltype,self._lninv,self._leinv,self._lmast,
							self._codno,self._exnor,commu,self._ptable,self._ngauss,self._consmas,self._vmass,
							**kwargs)

	@classmethod
	@cr('mesh.load')
	def load(cls,fname,**kwargs):
		'''
		Load a field from various formats
		'''
		# Guess format from extension
		fmt = os.path.splitext(fname)[1][1:] # skip the .
		compute_massMatrix = kwargs.pop('compute_massMatrix')
		# Pickle format
		if fmt.lower() == 'pkl':
			return io.pkl_load(fname)
		# HDF 5 format
		if fmt.lower() == 'h5':
			if not 'mpio' in kwargs.keys(): kwargs['mpio'] = True
			xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ptable,ngauss,consmas,massm = io.h5_load_mesh(fname,**kwargs)
			return cls(xyz,lnods,ltype,lninv,leinv,codno=codno,exnor=exnor,lmast=lmast,commu=commu,ngauss=ngauss,
				compute_massMatrix=compute_massMatrix,massm=massm,ptable=ptable,consistent_mass=consmas)
		# Return
		raiseError('Cannot load file <%s>!'%fname)

	def _switch_to_elemList(self):
		'''
		Create the element list and deallocate ltype and lnods
		'''
		if self._elemList is None:
			self._elemList = self.createElementList(self._lnods,self._ltype,ngauss=self._ngauss)

	def _switch_to_connectivity(self):
		'''
		Create connectivity and element type from element list
		and deallocate element list
		'''
		if self._lnods is None and self._ltype is None:
			self._lnods, self._ltype = self.createConnectivity(self._elemList)
			self._elemList = None

	@staticmethod
	@cr('mesh.elemList')
	def createElementList(lnods,ltype,ngauss=-1):
		'''
		Create an element list given the node connectivity
		and the element type.
		'''
		return np.array([FEM.createElementByType(ltype[iel],lnods[iel,:],ngauss) for iel in range(ltype.shape[0]) if ltype[iel] > 0],dtype=object)

	@staticmethod
	@cr('mesh.createConec')
	def createConnectivity(elemList):
		'''
		Obtain the connectivity and type of element from the
		element list
		'''
		if elemList.shape[0] > 0:
			return FEM.connectivity(elemList)
		else:
			return np.empty((0,0)), np.empty((0))

	# Properties
	@property
	def nnod(self):
		return self._lninv.shape[0]
	@property
	def nnodG(self):
		return mpi_reduce(self._lninv.max() if self._lninv.shape[0] > 0 else 0,op='max',all=True) + 1 # To account for python indexing
	@property
	def nnodT(self):
		return mpi_reduce(self._xyz.shape[0] if not np.any(np.isnan(self._xyz)) else 0,op='sum',all=True)
	@property
	def ngaussT(self):
		if self._ngaussT == 0 and not self._elemList is None:
			for elem in self._elemList: self._ngaussT += elem.ngauss
		return max(self._ngaussT,1)
	@property
	def ndim(self):
		return self._xyz.shape[1]
	@property
	def nel(self):
		return self._leinv.shape[0]
	@property
	def nelG(self):
		return mpi_reduce(self._leinv.max() if self._leinv.shape[0] > 0 else 0,op='max',all=True) + 1 # To account for python indexing
	@property
	def nelT(self):
		return mpi_reduce(self._leinv.shape[0] if not np.any(self._ltype == -1) else 0,op='sum',all=True)
	@property
	def xyz(self):
		return self._xyz
	@xyz.setter
	def xyz(self,xyz):
		self._xyz = xyz
	@property
	def x(self):
		return self._xyz[:,0]
	@property
	def y(self):
		return self._xyz[:,1]
	@property
	def z(self):
		return self._xyz[:,2]
	@property
	def xyz_center(self):
		return self.computeCellCenters()
	@property
	def eltype(self):
		if self._ltype is None:
			_, ltype = self.createConnectivity(self._elemList)
			if is_master(): ltype = np.array([-1],np.int32)
		else:
			ltype    = self._ltype
		return ltype
	@property
	def eltype_linear(self):
		return self.eltype
	@property
	def partition_table(self):
		return self._ptable
	@property
	def connectivity(self):
		return self._lnods
	@property
	def lninv(self):
		return self._lninv
	@property
	def leinv(self):
		return self._leinv
	@property
	def leinv_linear(self):
		return self._leinv
	@property
	def lmast(self):
		return self._lmast
	@property
	def connectivity_vtk(self):
		return self._lnods
	@property
	def boundingBox(self):
		xmin, xmax = np.min(self._xyz[:,0]),np.max(self._xyz[:,0])
		ymin, ymax = np.min(self._xyz[:,1]),np.max(self._xyz[:,1])
		zmin, zmax = np.min(self._xyz[:,2]),np.max(self._xyz[:,2])
		return Geom.SimpleCube(xmin,xmax,ymin,ymax,zmin,zmax)
	@property
	def volume(self):
		if not hasattr(self,'_volume'):
			self._volume = self.computeVolume()
		return self._volume


class MeshAlya(Mesh):
	'''
	The mesh class wraps the nodes, the elements and a number of variables
	and relates them so that the operations in parallel are easier.
	'''
	@cr('meshAlya.init')
#	@mem('mesh.init')
	def __init__(self, xyz, lnods, ltype, lninv, leinv, ngauss=-1, codno=np.array([[]],np.int32), lmast=np.array([],np.int32),
		exnor=np.array([[]],np.double), commu=np.array([[]],np.int32), massm=np.array([],np.double), ptable=None, 
		consistent_mass=False, compute_massMatrix=True, serial=False):
		'''
		Class constructor.

		IN:
			> xyz(nnod,3):            position of the nodes.
			> lnods(nelem,nnodxelem): connectivity matrix.
			> ltype(nelem):           type of element.
			> lninv(nnod):            global node numbering.
			> leinv(nelem):           global element numbering.
			> ngauss:                 number of Gauss points (-1 to use default).
			> codno(nnod,3):          boundary codes at the nodes.
			> lmast(nnod):            list of master and slave nodes.
			> exnor(nnod,3):          normals at the boundary nodes (0 a the rest of the domain).
			> commu(nnod,:):          communications array.
			> massm(nnod):            mass matrix.
			> consistent_mass:        activate/deactivate the consistent mass computation.
		'''
		# Compute partititon table
		super(MeshAlya,self).__init__(xyz,lnods,lninv,leinv,ltype,ptable)
		self.partitions(lninv,leinv,ptable=ptable,serial=serial)
		self._xyz_cen = None
		self._ngauss  = ngauss
		self._ngaussT = 0
		self._lmast   = lmast if not np.any(np.isnan(lmast)) else np.array([],dtype=lmast.dtype)
		self._codno   = codno if not np.any(np.isnan(codno)) else np.array([[]],dtype=codno.dtype)
		self._exnor   = exnor if not np.any(np.isnan(exnor)) else np.array([[]],dtype=exnor.dtype)
		self._consmas = consistent_mass
		# Initialize communications
		self.communications(commu,lmast)
		# Set up the boundaries
		if not self.comm is None:
			self._is_bc    = np.zeros((self.nnod,),dtype=bool)
			self._is_bc[self.comm.perm] = True
			self._bc_to_rm = self._comm.communicate_bc(self.lninv)
		else:
			self._is_bc    = None
			self._bc_to_rm = None
		# Compute mass matrix
		if len(massm) > 0: compute_massMatrix = False
		if compute_massMatrix:
			self._vmass = self.computeMassMatrix(consistent=consistent_mass)
		else:
			if len(massm) > 0:
				self._vmass = massm
			else:
				self._vmass = math.csr_create(self._xyz.shape[0],1,np.double) if consistent_mass else np.array([],np.double)

	def __str__(self):
		'''
		String representation
		'''
		s   = 'Mesh of %d nodes and %d elements (%s):\n' % (self.nnod,self.nel,str(self._elemList[0]) if not self._elemList is None else self._ltype[0]) if not is_master() else 'Mesh of %d nodes and %d elements (master):\n' % (self.nnod,self.nel)
		s  += '  > xyz  - max = ' + str(np.nanmax(self._xyz,axis=0)) + ', min = ' + str(np.nanmin(self._xyz,axis=0)) + '\n'
		return s

	# Functions
	@cr('mesh.newArray')
	def newArray(self,ndim=0,dtype=np.double,on_Gauss=False):
		'''
		Create a new array initialized to zeros.
		Use ndim = 0 for scalar arrays.
		'''
		npoints = self.nnod if not on_Gauss else self.ngaussT
		return np.zeros((npoints,),dtype=dtype) if ndim == 0 else np.zeros((npoints,ndim),dtype=dtype)

	@cr('meshAlya.find_bc')
	def find_bc(self,idx):
		'''
		Return all the nodes that belong to a given boundary condition
		'''
		if type(idx) is not list:
			return np.where(np.any(self._codno == idx,axis=1))[0]
		else:
			bc_list = [np.any(self._codno == id, axis=1) for id in idx]
			return np.where(np.logical_or.reduce(bc_list))[0]

	@cr('meshAlya.sort')
	def sort(self):
		'''
		Sort mesh according to lninv and leinv
		'''
		self._switch_to_connectivity()
		# Sort node dependent data
		idx = self._lninv.copy()
		# Sort node dependent data
		self._xyz   = self._xyz[idx]
		self._lninv = self._lninv[idx]
		self._codno = self._codno[idx]
		self._lmast = self._lmast[idx]
		self._exnor = self._exnor[idx]
		if not self._consmas: self._massm = self._massm[idx]
		# Sort communications
		if not self._comm == None:
			commu = self._comm.to_commu(self.nnod)
			self.communications(commu[idx],self._lmast)
		# Sort element dependent data
		idx = self._leinv.copy()
		self._lnods = self._lnods[idx]
		self._ltype = self._ltype[idx]
		self._leinv = self._leinv[idx]
		return self

	@cr('meshAlya.comms')
	def communications(self,commu,lmast):
		'''
		Initialize a communicator class for the current
		mesh subdomain.
		'''
		# Create a new instance of the communicator class
		self._comm = Communicator.from_Alya(commu,self.lninv,lmast) if commu.size > 0 and commu is not None else None
		return self._comm

	@cr('meshAlya.filter_bc')
	def filter_bc(self,array):
		'''
		Return an array without repeated boundary
		nodes.
		'''
		# Return the same array if there is no communicator
		if self._comm == None:
			return array
		# Delete the flagged bocos
		return np.delete(array,np.where(self._bc_to_rm)[0],axis=0)

	@cr('meshAlya.rotate')
	def rotate(self,angles,center=np.array([0.,0.,0.],np.double)):
		'''
		Rotate the mesh coordinates given the Euler angles and a center.
		'''
		self._xyz = math.vecRotate(self._xyz,angles[0],angles[1],angles[2],center)
		if self._xyz_cen is not None:
			self._xyz_cen = math.vecRotate(self._xyz_cen,angles[0],angles[1],angles[2],center)

	@cr('meshAlya.clip')
	def clip(self, poly):
		# Call the base class clip method and get the basic mesh and nodmask
		mesh_new, nodmask = super(MeshAlya, self).clip(poly)
		# Handling other attributes specific to MeshAlya
		codno = self._codno[nodmask].copy() if not self._codno.size == 0 else np.array([[]], np.double)
		exnor = self._exnor[nodmask].copy() if not self._exnor.size == 0 else np.array([[]], np.double)
		commu = np.array([[]], np.int32)  # Assuming communication info is lost
		massm = self._vmass[nodmask].copy() if not self._consmas and not len(self._vmass) == 0 else np.array([], np.double)
		if is_master() or mesh_new._ltype.shape[0] == 0:
			mesh_new._xyz   = np.array([[np.nan, np.nan, np.nan]])
			mesh_new._lnods = -2*np.ones((1,self._lnods.shape[1]))
			mesh_new._ltype = np.array([-1])
			#mesh_new._lninv = self._lninv
			#mesh_new._leinv = self._leinv
		return MeshAlya(mesh_new._xyz, mesh_new._lnods, mesh_new._ltype, mesh_new._lninv, mesh_new._leinv, codno=codno, exnor=exnor, commu=commu, massm=massm, compute_massMatrix=False), nodmask

	@cr('meshAlya.extract_bc')
	def extract_bc(self,ibc,casestr,basedir='./',use_post=True):
		'''
		Given a boundary id, it returns a new Mesh and a mask with 
		the cropped values that are on the requested boundary.	
		'''
		self._switch_to_connectivity()
		auxfile_fmt = io.MPIO_AUXFILE_P_FMT if MPI_SIZE > 1 or use_post else io.MPIO_AUXFILE_S_FMT
		# Read boundary arrays
		partfile  = os.path.join(basedir, io.MPIO_PARTFILE_FMT % casestr)
		lnodbfile = os.path.join(basedir, auxfile_fmt % (casestr,'LNODB'))
		codbofile = os.path.join(basedir, auxfile_fmt % (casestr,'CODBO'))
		ltypbfile = os.path.join(basedir, auxfile_fmt % (casestr,'LTYPB'))
		# Read the mesh boundary files
		lnodb,_ = io.AlyaMPIO_read(lnodbfile,partfile)
		ltypb,_ = io.AlyaMPIO_read(ltypbfile,partfile)
		codbo,_ = io.AlyaMPIO_read(codbofile,partfile)
		# Account for python starting to count in 0
		lnodb -= 1
		# Fix NaNs
		lnodb[np.isnan(lnodb)] = -1
		codbo[np.isnan(codbo)] = -1
		# Find which elements in the boundary belong to the requested element
		bcmask  = np.zeros((codbo.shape[0],),bool)
		for ib in ibc:
			bcmask = np.logical_or(bcmask,codbo == ibc)
		nodmask = np.zeros((self._xyz.shape[0],),dtype=bool)
		elmask  = np.zeros((self._ltype.shape[0],),dtype=bool)
		if np.any(bcmask):
			# Obtain a mask for the nodes belonging to the selected elements
			cconec  = lnodb[bcmask]
			ltype   = ltypb[bcmask].astype(np.int32)
			rconec  = cconec.ravel()
			relem   = np.array([ielem for inode in np.unique(rconec) for ielem in self.find_node_in_elems(inode)],np.int32)
			nodmask[np.unique(rconec)] = True
			elmask[np.unique(relem)]   = True
			# Obtained the nodes
			xyz = self._xyz[nodmask].copy()
			# Rebuild the mesh connectivity
			uni,rlnods = np.unique(rconec,return_inverse=True)
			lnods      = rlnods.reshape(cconec.shape).astype(np.int32).copy()
			if uni[0] == -1: lnods -= 1 # Multi-elements will have -1 as a value
			# Generate explicit codno
			codno = ibc*np.ones((xyz.shape[0],),np.int32)
		else:
			xyz   = np.nan*np.ones((1,self._xyz.shape[1]),np.double)
			lnods = -1*np.ones((1,lnodb.shape[1]),np.int32)
			ltype = -1*np.ones((1,),np.int32)
			codno = np.array([[]])
		# Recreate global numbering arrays
		lninv = np.copy(self._lninv[nodmask] if len(self._lninv) > 0 else self._lninv)
		leinv = np.copy(self._leinv[elmask]  if len(self._leinv) > 0 else self._leinv)
		# Obtain the minimum local value for the partititon
		lninv_loc = np.min(lninv) if np.any(elmask) else np.nan
		leinv_loc = np.min(leinv) if np.any(elmask) else np.nan
		# Subtract the minimum lninv and leinv of the partititon
		offst = mpi_reduce(lninv_loc,op='nanmin',all=True)
		if not np.isnan(offst): lninv -= int(offst)
		offst -= mpi_reduce(leinv_loc,op='nanmin',all=True)
		if not np.isnan(offst): leinv -= int(offst)
#		sizenod = xyz.shape[0]   if not np.any(np.isnan(xyz)) else 0
#		sizeelm = lnods.shape[0] if not np.all(lnods == -1)   else 0
#		ofstelm = np.sum(mpi_gather(sizeelm,all=True)[:MPI_RANK]) if MPI_SIZE > 1 else 0
#		ofstnod = np.sum(mpi_gather(sizenod,all=True)[:MPI_RANK]) if MPI_SIZE > 1 else 0
#		leinv   = np.arange(0,sizeelm,dtype=np.int32) + ofstelm
#		lninv   = np.arange(0,sizenod,dtype=np.int32) + ofstnod
		# Build new mesh array
		mesh_new = Mesh(xyz,lnods,ltype,lninv,leinv,ngauss=-1,codno=codno, 
			exnor = self._exnor[nodmask].copy() if not self._exnor.size == 0 else np.array([[]],np.double), 
			commu = np.array([[]],np.int32), # Information on commu is lost
			massm = self._vmass[nodmask].copy() if not self._consmas and not len(self._vmass) == 0 else np.array([],np.double)
		)
		return mesh_new, nodmask, elmask

	@cr('meshAlya.nodes2Gauss')
	def nodes2Gauss(self,f):
		'''
		Convert a field on the nodes to the Gauss points.
		'''
		self._switch_to_elemList()
		return FEM.nodes2Gauss(f,self._elemList,self.ngaussT) if not np.all(self._elemList == None) else np.nan*np.zeros((self._xyz.shape[0],f.shape[1]) if len(f.shape) > 1 else (self._xyz.shape[0],))

	@cr('meshAlya.gauss2Nodes')
	def gauss2Nodes(self,f_gp,consistent=None):
		'''
		Convert a field on the nodes to the Gauss points.
		'''
		self._switch_to_elemList()
		if consistent is None: consistent = self._consmas
		if consistent and not self._consmas: consistent = False
		dims = (self._xyz.shape[0],f_gp.shape[1]) if len(f_gp.shape) > 1 else (self._xyz.shape[0],) 
		n = FEM.gauss2Nodes(f_gp,self._xyz,self._elemList) if not np.all(self._elemList == None) else np.nan*np.zeros(dims,np.double)
		# Apply linear solver
		n = solvers.solver_lumped(self.volume,n,commu=self._comm)
		if consistent: n = solvers.solver_approxInverse(self._vmass,self.volume,n,commu=self._comm)
		return n

	@cr('meshAlya.smooth')
	def smooth(self,f,iters=5,consistent=None):
		'''
		Smoothes a field f on the current mesh.
		'''
		for it in range(iters):
			f = self.gauss2Nodes(self.nodes2Gauss(f),consistent=consistent)
		return f

	@cr('meshAlya.partitions')
	def partitions(self,leinv,lninv,ptable=None,serial=False):
		'''
		Generate the partition table if it does not exist
		'''
		return super(MeshAlya,self).partitions(leinv,lninv,ptable=ptable,serial=serial,has_master=False if serial else True)

	def _switch_to_connectivity(self):
		'''
		Create connectivity and element type from element list
		and deallocate element list
		'''
		if self._lnods is None and self._ltype is None:
			self._lnods, self._ltype = self.createConnectivity(self._elemList)
			nelnod = mpi_reduce(self._lnods.shape[1] if not is_master() else 0,op='max')
			if is_master():
				self._ltype = np.array([-1],np.int32)
				self._lnods = np.array([[-2]*nelnod],np.int32)
			self._elemList = None

	@classmethod
	@cr('meshAlya.read')
	def read(cls,casestr,basedir='./',fmt='mpio',ngauss=-1,read_commu=False,read_massm=False,
		read_codno=True,read_exnor=False,read_lmast=True,alt_basedir=None,
		use_consistent=False,compute_massMatrix=True,use_post=True):
		'''
		Read the necessary files to create a mesh in Alya and return
		an instance of the mesh class.

		IN:
			> casestr:    name of the Alya case.
			> basedir:    (optional) main directory of the simulation.
			> ngauss:     (optional) number of gauss points.
			> read_codno: (optional) read the nodes code numbers.
			> read_exnor: (optional) read the nodes exterior normal.
			> read_commu: (optional) read the communications matrix.
			> read_massm: (optional) read the mass matrix.
			> read_lmast: (optional) read the periodic connectivity.

		OUT:
			> mesh:       instance of the Mesh class.
		'''
		if fmt.lower() == 'mpio':
			if use_consistent: read_massm = False # Ensure not reading the massm if using the consistent
			coord, lnods, ltype, lninv, leinv, lmast, commu, massm, codno, exnor, ptable = meshReadMPIO(casestr,basedir,read_commu,read_massm,read_codno,read_exnor,read_lmast,alt_basedir,use_post)
		elif fmt.lower() in ['ensi','ensight']:
			coord, lnods, ltype, lninv, leinv, lmast, commu, massm, codno, exnor, ptable = meshReadENSIGHT(casestr,basedir)
		elif fmt.lower() in ['vtk','VTK']:
			coord, lnods, ltype, lninv, leinv, lmast, commu, massm, codno, exnor, ptable = meshReadVTK(casestr,basedir)
		else:
			raiseError('Format <%s> not implemented!'%fmt)
		# Return an instance of Mesh
		return cls(coord,lnods.astype(np.int32),ltype.astype(np.int32),lninv.astype(np.int32),
			leinv.astype(np.int32),lmast=lmast.astype(np.int32),codno=codno.astype(np.int32),
			exnor=exnor.astype(np.double),commu=commu,ngauss=ngauss,massm=massm,ptable=ptable,
			consistent_mass=use_consistent,compute_massMatrix=compute_massMatrix)

	@cr('meshAlya.write')
	def write(self,casestr,basedir='./',fmt='mpio',nsub=max(MPI_SIZE-1,1),use_post=True,linkfile=None):
		'''
		Store the data in the mesh using various formats.

		This method differs from save in the fact that save is used
		to recover the mesh, write only outputs the data.
		'''
		self._switch_to_connectivity()
		if fmt.lower() == 'mpio':
			meshWriteMPIO(self,casestr,basedir,nsub,use_post)
		elif fmt.lower() in ['ensi','ensight']:
			meshWriteENSIGHT(self,casestr,basedir)
		elif fmt.lower() in ['vtkhdf','vtkh5']:
			meshWriteVTKH5(self,casestr,basedir,linkfile)
		else:
			raiseError('Format <%s> not implemented!'%fmt)

	@classmethod
	@cr('meshAlya.load')
	def load(cls,fname,**kwargs):
		'''
		Load a field from various formats
		'''
		# Guess format from extension
		fmt = os.path.splitext(fname)[1][1:] # skip the .
		compute_massMatrix = kwargs.pop('compute_massMatrix')
		# Pickle format
		if fmt.lower() == 'pkl':
			return io.pkl_load(fname)
		# HDF 5 format
		if fmt.lower() == 'h5':
			if not 'mpio' in kwargs.keys(): kwargs['mpio'] = True
			xyz,lnods,ltype,lninv,leinv,lmast,codno,exnor,commu,ptable,ngauss,consmas,massm = io.h5_load_mesh(fname,**kwargs)
			return cls(xyz,lnods,ltype,lninv,leinv,codno=codno,exnor=exnor,lmast=lmast,commu=commu,ngauss=ngauss,
				compute_massMatrix=compute_massMatrix,massm=massm,ptable=ptable,consistent_mass=consmas)
		# Return
		raiseError('Cannot load file <%s>!'%fname)

	@cr('meshAlya.save')
	def save(self,fname,**kwargs):
		'''
		Store the field in various formats.
		'''
		self._switch_to_connectivity()
		# Guess format from extension
		fmt = os.path.splitext(fname)[1][1:] # skip the .
		# Pickle format
		if fmt.lower() == 'pkl': io.pkl_save(fname,self)
		# HDF5 format
		if fmt.lower() == 'h5':
			# Set default parameters
			if not 'metadata'     in kwargs.keys(): kwargs['metadata']     = {}
			if not 'mpio'         in kwargs.keys(): kwargs['mpio']         = True
			if not 'write_master' in kwargs.keys(): kwargs['write_master'] = not self._ptable.has_master
			commu = self._comm.to_commu(self.nnod) if self._comm is not None else np.array([[]])
			io.h5_save_mesh(fname,self.xyz,self.connectivity,self._ltype,self._lninv,self._leinv,self._lmast,
							self._codno,self._exnor,commu,self._ptable,self._ngauss,self._consmas,self._vmass,
							**kwargs)

	@classmethod
	@cr('meshAlya.plane')
	def plane(cls,p1,p2,p4,n1,n2,ngauss=-1,bunching='',f=0.2,ptable=None,
		use_consistent=False,compute_massMatrix=True):
		'''
		3D mesh plane, useful for slices.

		4-------3
		|		|
		|		|
		1-------2

		IN:
			> p1, p2, p4: points of the rectangle
			> n1, n2:     number of points per direction
			> ngauss:     (optional) number of gauss points
			> bunching:   direction in which to concentrate the mesh
						  allowed 'x', 'y' or 'all'
			> f:          mesh concentration factor

		OUT:
			> mesh:       instance of the Mesh class.
		'''
		conc = 0
		if 'x'   in bunching: conc = 1
		if 'y'   in bunching: conc = 2
		if 'all' in bunching: conc = 3
		coord,lnods,ltype,lninv,leinv = planeMesh(p1,p2,p4,n1,n2,conc=conc,f=0.2)
		# Return
		return cls(coord,lnods,ltype,lninv,leinv,ngauss=ngauss,ptable=ptable,serial=True,
			consistent_mass=use_consistent,compute_massMatrix=compute_massMatrix)

	@classmethod
	@cr('meshAlya.cube')
	def cube(cls,p1,p2,p4,p5,n1,n2,n3,ngauss=-1,bunching='',f=0.2,ptable=None,
		use_consistent=False,compute_massMatrix=True):
		'''
		3D mesh cube, useful for volumes.

		  8-------7
		 /|      /|
		4-------3 |
		| 5-----|-6
		|/      |/
		1-------2

		IN:
			> p1, p2, p4, p5: points of the cube
			> n1, n2, n3:     number of points per direction
			> ngauss:         (optional) number of gauss points
			> bunching:       direction in which to concentrate the mesh
							  allowed 'x', 'y' or 'all'
			> f:              mesh concentration factor
		
		OUT:
			> mesh:           instance of the Mesh class.
		#TODO: particularize from the super Mesh class
		'''
		conc = 0
		if 'x'   in bunching: conc = 1
		if 'y'   in bunching: conc = 2
		if 'z'   in bunching: conc = 3
		if 'all' in bunching: conc = 4
		coord,lnods,ltype,lninv,leinv = cubeMesh(p1,p2,p4,p5,n1,n2,n3,conc=conc,f=f)
		# Return
		return cls(coord,lnods,ltype,lninv,leinv,ngauss=ngauss,ptable=ptable,serial=True,
			consistent_mass=use_consistent,compute_massMatrix=compute_massMatrix)

	# Properties
	@property
	def nbound(self):
		return np.sum(self._is_bc)
	@property
	def nelnod(self):
		return FEM.nodes_per_element(self._elemList) if not self._elemList is None else self._lnods.shape[1]
	@property
	def elem(self):
		return self._elemList
	@property
	def comm(self):
		return self._comm
	@property
	def mass_matrix(self):
		return self._vmass
	@property
	def connectivity(self):
		if self._lnods is None:
			lnods, _  = self.createConnectivity(self._elemList)
			nelnod    = mpi_reduce(lnods.shape[1] if not is_master() else 0,op='max')
			if is_master(): 
				lnods = np.array([[-2]*nelnod],np.int32)
		else:
			lnods     = self._lnods
		return lnods	
	@property
	def codno(self):
		return self._codno
	@property
	def exnor(self):
		return self._exnor
	@property
	def bc2rm_mask(self):
		return np.logical_not(self._bc_to_rm)
	@property
	def discreteBox(self):
		maxSize = 2.0*np.cbrt(np.nanmax(self.volume))
		points  = np.vstack([self.xyz,self.xyz_center])
		return Geom.DiscreteBox(self.boundingBox,points,maxSize)


class MeshSOD2D(Mesh):
	'''
	The mesh class wraps the nodes, the elements and a number of variables
	and relates them so that the operations in parallel are easier.
	'''
	@cr('meshSOD2D.init')
	def __init__(self, xyz, lnods, lninv, leinv, ltype, porder, boucodes=None, bouconec=None, lmast = None, ptable=None, ranksToComm=None, commsMemSize=None, nodesToComm=None, ndime=3):
		'''
		Class constructor.

		IN:
			> xyz(nnod,3):            position of the nodes.
			> lnods(nelem,nnodxelem): connectivity matrix.
			> lninv(nnod):            global node numbering.
			> leinv(nelem):           global element numbering.
		'''
		super(MeshSOD2D,self).__init__(xyz, lnods, lninv, leinv, ltype, ptable)
		self.partitions(lninv,leinv,ptable)
		self._porder   = porder
		self._ngaussT  = 0
		self._comm     = self.communications(ranksToComm, commsMemSize, nodesToComm) if ranksToComm is not None else None
		self._boucodes = boucodes
		self._bouconec = bouconec
		self._lmast    = lmast
		self._ndime    = ndime
		self._consmas  = None
		self._vmass    = None

	@cr('meshSOD2D.clip')
	def clip(self, poly):
		# Call the base class clip method and get the basic mesh and nodmask
		mesh_new, nodmask = super(MeshSOD2D, self).clip(poly) # Boundary and communication information are lost
		return MeshSOD2D(mesh_new._xyz, mesh_new._lnods, mesh_new._lninv, mesh_new._leinv, mesh_new._ltype, self._porder), nodmask

	@cr('meshSOD2D.gradient')
	def gradient(self, f):
		return super(MeshSOD2D, self).gradient(f,on_Gauss=False,consistent=None)
	
	@cr('meshSOD2D.divergence')
	def divergence(self, f):
		return super(MeshSOD2D, self).divergence(f,on_Gauss=False,consistent=None)
	
	@cr('meshSOD2D.laplacian')
	def laplacian(self, f):
		return super(MeshSOD2D, self).laplacian(f,on_Gauss=False,consistent=None)
	
	@cr('meshSOD2D.mass_matrix')
	def computeMassMatrix(self):
		return super(MeshSOD2D, self).computeMassMatrix(consistent=False)

	@cr('meshSOD2D.partitions')
	def partitions(self,leinv,lninv,ptable=None):
		return super(MeshSOD2D,self).partitions(leinv,lninv,ptable=ptable,serial=False,has_master=False)

	@cr('meshSOD2D.extract_bc')
	def extract_bc(self,ibc):
		'''
		Given a list of boundary ids, it returns a new Mesh and a mask with 
		the cropped values that are on the requested boundary.	
		'''
		elmask = np.zeros((self._boucodes.shape[0],),bool)
		for ib in ibc:
			elmask = np.logical_or(elmask,self._boucodes == ib)
		elindex = np.argwhere(elmask)[:,0]
		if elindex.shape[0] == 0:
			xyzbo      = np.empty((0,3)) 
			lnodbo2    = np.empty((0,0))
			lninbo     = np.empty((0))
			leinbo     = np.empty((0)) 
			ltype      = np.empty((0))
		else:
			nnodxbo    = (self._porder+1)**2
			nelbo      = elindex.shape[0]
			lnodbo     = self._bouconec[elindex,:]
			lninbo     = np.unique(lnodbo.reshape((nelbo*nnodxbo,), order='C'))
			mapping    = {value: index for index, value in enumerate(lninbo)}
			lnodbo2    = np.vectorize(mapping.get)(lnodbo)
			xyzbo      = self._xyz[lninbo]
			leinbo     = np.arange(0, elindex.shape[0], 1, dtype=np.int32)
			ltype      = 15*np.ones((leinbo.shape[0],), dtype=np.int32)
		nparts  = MPI_SIZE
		ids     = np.arange(0,MPI_SIZE,1,dtype=np.int32)
		nodpart = mpi_gather(lninbo.shape[0], all=True)
		elpart  = mpi_gather(leinbo.shape[0], all=True)
		bpart   = elpart
		# leinbo must be fixed so that we account for the total number of elements
		# in the global mesh
		leinbo += np.sum(elpart[:MPI_RANK])
		ptable  = PartitionTable(nparts, ids, nodpart, elpart, bpart, has_master=False)

		return MeshSOD2D(xyzbo, lnodbo2.astype(np.int32), lninbo.astype(np.int32), leinbo.astype(np.int32), ltype.astype(np.int32), self._porder, ptable=ptable, ndime=2), lninbo, elmask
	
	@cr('meshSOD2D.HexaWallDistances')
	def HexaWallDistances(mesh, bcmesh, surf_mask, field, uStreamwise, WALLMODEL):

		'''
		Computes wall distance walues and outputs a Field Object containing them.

		IN:
			> mesh           : a SOD2D Mesh Class Object
			> bcmesh         : a SOD2D Mesh Class Object
			> surf_mask:     : an array containing the nodes after extract_bc operation
			> field          : a SOD2D Field Class Object
			> uStreamwise    : a 3 component array specifying the streamwise velocity component
			> WALLMODEL      : flag True or Flase whether wall modeling is on or off in the solver

		OUT:
			> xPlus          : array containing X+ values on the bcmesh
			> yPlus          : array containing Y+ values on the bcmesh
			> zPlus          : array containing Z+ values on the bcmesh
			> surfField      : a SOD2D Field Class Object
		'''

		# Wall model flag
		if not WALLMODEL:
			# Computing gradient
			if 'avvel' in field._vardict:
				field['gradVel'] = mesh.gradient(field['avvel'])
			else:
				raise KeyError("Please import 'avvel' in your field!")

		# Working only with process that have nodes on surface
		is_not_empty = surf_mask.shape[0] != 0 

		fieldWall = field.selectMask(surf_mask)

		if is_not_empty:
			# Computing normals at wall
			normDir = bcmesh.computeNormals()

			# Computing tangential direction (valid for extruded mesh in spanwise direcion)
			tangDir  = np.transpose(np.array([normDir[:,1], -normDir[:,0], normDir[:,2]]))

			if WALLMODEL:
				if 'avtw' in fieldWall._vardict:
					tauW = math.dot(fieldWall['avtw'], tangDir)
				else:
					raise KeyError("Please import 'avtw' in your field!")
			else:
				duDotN   = math.tensVecProd(fieldWall['gradVel'], normDir)
				duTangdy = math.dot(duDotN, tangDir)
				if 'avmueff' in fieldWall._vardict:
					tauW = fieldWall['avmueff']*duTangdy
				else:
					raise KeyError("Please import 'avmueff' in your field!")

			# Computing wall distances
			connecvtk = mesh.connectivity_vtk
			coord = mesh.xyz
			if len(uStreamwise) == 3:
				uStreamwise = uStreamwise.astype(np.double)
			else:
				raise ValueError("uStreamwise must be a 3 component array!")
			firstNodeMask, deltaX, deltaZ, deltaY = computeWallDistancesSOD2D(surf_mask, connecvtk, coord, np.array(uStreamwise))
			# Wall unit distances
			xPlus = deltaX/fieldWall['avmueff']*np.sqrt(abs(tauW))
			yPlus = deltaY/fieldWall['avmueff']*np.sqrt(abs(tauW))
			zPlus = deltaZ/fieldWall['avmueff']*np.sqrt(abs(tauW))

			# Output VTK
			surfField = FieldSOD2D(xyz=bcmesh.xyz, ptable=mesh.partition_table,
								   xPlus = xPlus, yPlus = yPlus, zPlus = zPlus,
								   deltaX = deltaX, deltaY = deltaY, deltaZ = deltaZ)   
		else:
			xPlus  = np.zeros(surf_mask.shape)
			yPlus  = np.zeros(surf_mask.shape)
			zPlus  = np.zeros(surf_mask.shape)
			deltaX = np.zeros(surf_mask.shape)
			deltaY = np.zeros(surf_mask.shape)
			deltaZ = np.zeros(surf_mask.shape)

			surfField = FieldSOD2D(xyz=bcmesh.xyz, ptable=mesh.partition_table,
								   xPlus = xPlus, yPlus = yPlus, zPlus = zPlus,
								   deltaX = deltaX, deltaY = deltaY, deltaZ = deltaZ)
			
		return surfField, xPlus, yPlus, zPlus
	
	@classmethod
	@cr('meshSOD2D.plane')
	def plane(cls,p1,p2,p4,n1,n2,porder=1,bunching='',nparts=1):
		'''
		3D mesh plane, useful for slices.

		4-------3
		|		|
		|		|
		1-------2

		IN:
			> p1, p2, p4: points of the rectangle
			> n1, n2:     number of points per direction
			> ngauss:     (optional) number of gauss points
			> bunching:   direction in which to concentrate the mesh
						  allowed 'x', 'y' or 'all'
			> f:          mesh concentration factor

		OUT:
			> mesh:       instance of the Mesh class.
		#TODO: Implement plane HO mesh
		'''
		conc = 0
		if 'x'   in bunching: conc = 1
		if 'y'   in bunching: conc = 2
		if 'all' in bunching: conc = 3
		coord,lnods,ltype,lninv,leinv = planeMesh(p1,p2,p4,n1,n2,conc=conc,f=0.2)
		ltype = 15*np.ones(((n1-1)*(n2-1),),dtype=np.int32)
		ptable = PartitionTable.new(nparts, nelems=leinv.shape[0], npoints=lninv.shape[0], has_master=False)
		# Return
		return cls(coord, lnods, lninv, leinv, ltype, porder, ptable=ptable)
	
	@classmethod
	@cr('mesh.cube')
	def cube(cls, p1, p2, p4, p5, n1, n2, n3, porder=1,bunching='',f=0.2,nparts=1):
		"""
		3D mesh cube, useful for volumes.
		#TODO: Implement cube HO mesh
		"""
		conc = 0
		if 'x' in bunching: conc = 1
		if 'y' in bunching: conc = 2
		if 'z' in bunching: conc = 3
		if 'all' in bunching: conc = 4
		coord, lnods, ltype, lninv, leinv = cubeMesh(p1, p2, p4, p5, n1, n2, n3, conc=conc, f=f)
		ltype =40*np.ones(((n1-1)*(n2-1)*(n3-1),),dtype=np.int32)
		ptable = PartitionTable.new(nparts, nelems=leinv.shape[0], npoints=lninv.shape[0], has_master=False)
		return cls(coord, lnods, lninv, leinv, ltype, porder, ptable=ptable)
		
	@cr('meshSOD2D.comms')
	def communications(self, ranksToComm, commsMemSize, nodesToComm):
		'''
		Initialize a communicator class for the current
		mesh subdomain.
		'''
		# Create a new instance of the communicator class
		self._comm = Communicator.from_SOD2D(ranksToComm, commsMemSize, nodesToComm)
		return self._comm

	def _switch_to_elemList(self):
		'''
		Create the element list and deallocate ltype and lnods
		'''
		if self._elemList is None:
			self._elemList = self.createElementList(self._lnods,self._ltype,self._porder,self._ndime)
			#self._lnods    = None # TODO: Que fem amb aquests d'aquí? i el switch to connectivity? Fem un mapa high_performance i un mapa low_mem
			#self._ltype    = None # TODO: Que fem amb aquests d'aquí? i el switch to connectivity?

	def _switch_to_connectivity(self):
		'''
		Create connectivity and element type from element list
		and deallocate element list
		'''
		if self._lnods is None and self._ltype is None:
			self._lnods, self._ltype = self.createConnectivity(self._elemList)
			self._elemList = None

	@staticmethod
	@cr('meshSOD2D.elemList')
	def createElementList(lnods,ltype,porder,ndime):
		'''
		Create an element list given the node connectivity
		and the node type.
		'''
		ngauss, xi, posnod, weigp, shapef, gradi = defineHighOrderElement(porder, ndime)

		return np.array([FEM.createElementByType(ltype[iel],lnods[iel,:],ngauss,xi=xi,posnod=posnod,weigp=weigp,shapef=shapef,gradi=gradi) for iel in range(ltype.shape[0]) if ltype[iel] > 0],dtype=object)

	@classmethod
	@cr('meshSOD2D.read')
	def read(cls,casestr,basedir='./'):
		'''
		Read the necessary files to create a mesh in Alya and return
		an instance of the mesh class.

		IN:
			> casestr:    name of the Alya case.
			> basedir:    (optional) main directory of the simulation.
			> ngauss:     (optional) number of gauss points.
			> read_codno: (optional) read the nodes code numbers.
			> read_exnor: (optional) read the nodes exterior normal.
			> read_commu: (optional) read the communications matrix.
			> read_massm: (optional) read the mass matrix.
			> read_lmast: (optional) read the periodic connectivity.

		OUT:
			> mesh:       instance of the Mesh class.
		'''
		coord, lnods, lninv, leinv, porder, ltype, boucodes, bouconec, lmast, ranksToComm, commsMemSize, nodesToComm, ptable = meshReadSOD2DHDF(casestr,basedir)
		# Return an instance of Mesh
		return cls(coord,lnods.astype(np.int32),lninv.astype(np.int32),leinv.astype(np.int32),ltype.astype(np.int32),porder,boucodes=boucodes.astype(np.int32),bouconec=bouconec.astype(np.int32),lmast=lmast.astype(np.int32),ranksToComm=ranksToComm.astype(np.int32),commsMemSize=commsMemSize.astype(np.int32),nodesToComm=nodesToComm.astype(np.int32),ptable=ptable)

	@cr('meshSOD2D.write')
	def write(self,casestr,basedir='./',linkfile=None):
		'''
		Store the data in the mesh using vtkh5.
		'''
		self._switch_to_connectivity()
		meshWriteVTKH5(self,casestr,basedir,linkfile)

	# Properties
	@property
	@cr('meshSOD2D.connectivity_vtk')
	def connectivity_vtk(self):
		##Linearizes the mesh and orders it according to the VTK connectivity
		if self._porder == 1:
			return self._lnods
		else:
			nelVTKxel   = self._porder**self._ndime
			npoint      = self._porder + 1
			lnods_vtk   = np.zeros((nelVTKxel*self.nel,2**self._ndime),dtype=np.int32)
			point_order = np.arange(self._porder+1, dtype=np.int32)
			point_order[1:-1] = point_order[2:]
			point_order[-1] = 1
			if self.nel > 0:
				if self._ndime == 2:
					indices = np.array([[point_order[ii] * npoint + point_order[jj],
								 point_order[ii + 1] * npoint + point_order[jj],
								 point_order[ii + 1] * npoint + point_order[jj + 1],
								 point_order[ii] * npoint + point_order[jj + 1]]
								for ii in range(self._porder) for jj in range(self._porder)], dtype=np.int32)
				else:
					indices = np.array([[point_order[kk]*npoint**2 + point_order[ii]*npoint + point_order[jj],
								 point_order[kk]*npoint**2 + point_order[ii+1]*npoint + point_order[jj],
								 point_order[kk]*npoint**2 + point_order[ii+1]*npoint + point_order[jj+1],
								 point_order[kk]*npoint**2 + point_order[ii]*npoint + point_order[jj+1],
								 point_order[kk+1]*npoint**2 + point_order[ii]*npoint + point_order[jj],
								 point_order[kk+1]*npoint**2 + point_order[ii+1]*npoint + point_order[jj],
								 point_order[kk+1]*npoint**2 + point_order[ii+1]*npoint + point_order[jj+1],
								 point_order[kk+1]*npoint**2 + point_order[ii]*npoint + point_order[jj+1]]
								for kk in range(self._porder) for ii in range(self._porder) for jj in range(self._porder)], dtype=np.int32)
				for iel, elem in enumerate(self._lnods):
					lnods_vtk[iel * nelVTKxel:(iel + 1) * nelVTKxel, :] = elem[indices]
			return lnods_vtk

	@property
	def eltype_linear(self):
		nelVTKxel = self._porder**self._ndime
		if self.nel > 0:
			if self._ltype[0] == 10:
				return self._ltype
			else:
				linel     = linearizedElements[self._ltype[0]] ## SOD2D has only one type of element
				return linel*np.ones(self.nel*nelVTKxel,dtype=np.int32)
		else:
			return self._ltype
	
	@property
	def leinv_linear(self):
		nelVTKxel    = self._porder**self._ndime
		leinv_linear = np.zeros((self.leinv.shape[0]*nelVTKxel,),dtype=np.int32)
		base_linear  = np.arange(nelVTKxel, dtype=np.int32)
		for iel, el in enumerate(self.leinv):
			leinv_linear[iel*nelVTKxel:(iel+1)*nelVTKxel] = el*nelVTKxel + base_linear
		return leinv_linear
	
	@property
	def ndime(self):
		return self._ndime
	
	@property
	def porder(self):
		return self._porder
	
	@property 
	def vtk2ijk(self):
		i = np.array([0, self._porder] + list(range(1, self._porder)))
		j = np.array([0, self._porder] + list(range(1, self._porder)))
		k = np.array([0, self._porder] + list(range(1, self._porder)))
		point_indices = np.zeros(((self._porder+1)**self._ndime,), dtype=np.int32)
		ip = 0
		if self._ndime == 2:
			for ii in i:
				for jj in j:
					point_indices[ip] = vtkHigherOrderQuadrilateral_pointIndexFromIJ(self._porder, ii, jj) - 1 #-1 to account for python indexing starting at 0
					ip = ip + 1
		if self._ndime == 3:
			for kk in k:
				for ii in i:
					for jj in j:
						point_indices[ip] = vtkHigherOrderHexahedron_pointIndexFromIJK(self._porder, ii, jj, kk) - 1 #-1 to account for python indexing starting at 0
						ip = ip + 1
		return point_indices
	
	@property
	def ijk2vtk(self):
		return np.argsort(self.vtk2ijk)
		

def meshReadSOD2DHDF(casestr,basedir):
	'''
	Read mesh files for SOD2D in HDF5 format
	'''
	# Generate partition table
	ptable = PartitionTable.fromSOD2DHDF(casestr,basedir)
	# Read mesh
	coord, lnods, lninv, leinv, porder, boucodes, bouconec, lmast, ranksToComm, commsMemSize, nodesToComm = io.SOD2DHDF_read_mesh(casestr,ptable,basedir=basedir)
	# Account for python starting to count in 0
	lnods       -= 1
	lninv       -= 1
	leinv       -= 1
	bouconec    -= 1
	nodesToComm -= 1
	#Set up the ltype array (only one type of element is supported in SOD2D)
	ltype = 40*np.ones((leinv.shape[0],))
	# Return
	return coord, lnods, lninv, leinv, porder, ltype, boucodes, bouconec, lmast, ranksToComm, commsMemSize, nodesToComm, ptable

def meshReadMPIO(casestr,basedir,read_commu,read_massm,read_codno,read_exnor,read_lmast,alt_basedir,use_post):
	'''
	Read mesh files in MPIO format
	'''
	# Define format of the input
	auxfile_fmt = io.MPIO_AUXFILE_P_FMT if MPI_SIZE > 1 or use_post else io.MPIO_AUXFILE_S_FMT
	binfile_fmt = io.MPIO_BINFILE_P_FMT if MPI_SIZE > 1 or use_post else io.MPIO_BINFILE_S_FMT
	# Check if file exists, otherwise recover the parallel format
	if not os.path.exists(os.path.join(basedir, auxfile_fmt % (casestr,'COORD'))):
		auxfile_fmt = io.MPIO_AUXFILE_P_FMT
		binfile_fmt = io.MPIO_BINFILE_P_FMT
	alt_basedir = basedir if alt_basedir == None else alt_basedir
	# Create the filenames for the case
	partfile  = os.path.join(basedir,     io.MPIO_PARTFILE_FMT % casestr)
	coordfile = os.path.join(basedir,     auxfile_fmt % (casestr,'COORD'))
	lnodsfile = os.path.join(basedir,     auxfile_fmt % (casestr,'LNODS'))
	ltypefile = os.path.join(basedir,     auxfile_fmt % (casestr,'LTYPE'))
	lninvfile = os.path.join(basedir,     auxfile_fmt % (casestr,'LNINV'))
	leinvfile = os.path.join(basedir,     auxfile_fmt % (casestr,'LEINV'))
	lmastfile = os.path.join(basedir,     auxfile_fmt % (casestr,'LMAST'))
	codnofile = os.path.join(alt_basedir, auxfile_fmt % (casestr,'CODNO'))
	exnorfile = os.path.join(alt_basedir, binfile_fmt % (casestr,'EXNOR',0))
	commufile = os.path.join(alt_basedir, binfile_fmt % (casestr,'COMMU',0))
	massmfile = os.path.join(alt_basedir, binfile_fmt % (casestr,'MASSM',0))
	# Read the partition table
	ptable  = PartitionTable.fromAlya(casestr,basedir)
	# Read the mesh files
	coord,_ = io.AlyaMPIO_read(coordfile,partfile,force_partition_data=True)
	lnods,_ = io.AlyaMPIO_read(lnodsfile,partfile)
	ltype,_ = io.AlyaMPIO_read(ltypefile,partfile)
	lninv,_ = io.AlyaMPIO_read(lninvfile,partfile)
	leinv,_ = io.AlyaMPIO_read(leinvfile,partfile)
	# Account for python starting to count in 0
	lnods -= 1
	lninv -= 1
	leinv -= 1
	# Fix NaNs
	lnods[np.isnan(lnods)] = -1
	ltype[np.isnan(ltype)] = -1
	lninv[np.isnan(lninv)] = -1
	leinv[np.isnan(leinv)] = -1
	# Read LMAST file if it exists
	if read_lmast and os.path.exists(lmastfile):
		lmast,_ = io.AlyaMPIO_read(lmastfile,partfile)
	else:
		lmast = np.array([],np.int32)
	# Read communication array if needed
	if read_commu:
		if not os.path.exists(commufile): raiseError("COMMU file not found or not processed!")
		commu, _  = io.AlyaMPIO_read(commufile,partfile)
	else:
		commu = np.array([[]],np.int32)
	# Read mass matrix if needed
	if read_massm:
		if not os.path.exists(massmfile): raiseError("MASSM file not found or not processed!")
		massm, _  = io.AlyaMPIO_read(massmfile,partfile)
	else:
		massm = np.array([],np.double)
	# Read boundary conditions if requested
	if read_codno:
		if not os.path.exists(codnofile): raiseError("CODNO file not found or not processed!")
		codno, _  = io.AlyaMPIO_read(codnofile,partfile)
		codno[np.isnan(codno)] = -1
	else:
		codno = np.array([[]],np.int32)
	# Read exterior normal if requested
	if read_exnor:
		if not os.path.exists(exnorfile): raiseError("EXNOR file not found or not processed!")
		exnor, _  = io.AlyaMPIO_read(exnorfile,partfile)
		exnor[np.isnan(exnor)] = -1
	else:
		exnor = np.array([[]],np.double)
	# Return
	return coord, lnods, ltype, lninv, leinv, lmast, commu, massm, codno, exnor, ptable

def meshReadENSIGHT(casestr,basedir):
	'''
	Read mesh files in Ensight format
	'''
	if MPI_SIZE > 1: raiseWarning('Ensight format not supposed to be working in parallel!')
	# Create the filename for the case
	geofile = os.path.join(basedir,'%s.ensi.geo'%casestr)
	if not os.path.isfile(geofile): geofile = os.path.join(basedir,'%s.geo'%casestr)
	# Read the mesh file
	coord, lnods, header = io.Ensight_readGeo(geofile)
	# Build ltype
	t = -1
	if 'tria3'  in header['eltype']: t = 10 # TRI03
	if 'quad4'  in header['eltype']: t = 12 # QUA04
	if 'tetra4' in header['eltype']: t = 30 # TET04
	if 'penta6' in header['eltype']: t = 34 # PEN06
	if 'hexa8'  in header['eltype']: t = 37 # HEX08
	ltype = t*np.ones((lnods.shape[0],),dtype=np.int32)
	lninv = np.arange(1,coord.shape[0]+1,dtype=np.int32)
	leinv = np.arange(1,ltype.shape[0]+1,dtype=np.int32)
	# Account for python starting to count in 0
	lnods -= 1
	lninv -= 1
	leinv -= 1
	# Fix NaNs
	lnods[np.isnan(lnods)] = -1
	ltype[np.isnan(ltype)] = -1
	lninv[np.isnan(lninv)] = -1
	leinv[np.isnan(leinv)] = -1
	# Non existent arrays
	lmast  = np.zeros((coord.shape[0],),np.int32)
	commu  = np.array([[]],np.int32)
	massm  = np.array([],np.double)
	codno  = np.array([[]],np.int32)
	exnor  = np.array([[]],np.double)
	ptable = None
	# Return
	return coord, lnods, ltype, lninv, leinv, lmast, commu, massm, codno, exnor, ptable

def meshReadVTK(casestr,basedir):
	'''
	Read mesh files in VTK format
	'''
	filename = os.path.join(basedir,casestr) 
	vtkData  = io.vtkIO(filename=filename,mode='read',varlist=[])
	# Read from VTK
	coord = vtkData.points
	lnods = vtkData.connectivity
	ltype = vtkData.cellTypes
	lninv = np.arange(coord.shape[0],dtype=np.int32)
	leinv = np.arange(ltype.shape[0],dtype=np.int32)
	# Non existent arrays
	lmast  = np.zeros((coord.shape[0],),np.int32)
	commu  = np.array([[]],np.int32)
	massm  = np.array([],np.double)
	codno  = np.array([[]],np.int32)
	exnor  = np.array([[]],np.double)
	ptable = None
	# Return
	return coord, lnods, ltype, lninv, leinv, lmast, commu, massm, codno, exnor, ptable

def meshWriteMPIO(mesh,casestr,basedir,nsub,use_post):
	'''
	Write mesh in MPIO format
	'''
	do_write = not np.any(np.isnan(mesh.xyz)) # When not to write
	force_partition_data = True
	# Partition file format
	partfile  = os.path.join(basedir, io.MPIO_PARTFILE_FMT % casestr)
	# Check if the partition file exists, otherwise create it
	if not os.path.exists(partfile): mesh.partition_table.toAlya(casestr,basedir)
	# Alya MPIO binaries formats
	auxfile_fmt = io.MPIO_AUXFILE_P_FMT if MPI_SIZE > 1 or use_post else io.MPIO_AUXFILE_S_FMT
	binfile_fmt = io.MPIO_BINFILE_P_FMT if MPI_SIZE > 1 or use_post else io.MPIO_BINFILE_S_FMT
	# Alya MPIO binaries
	coordfile = os.path.join(basedir, auxfile_fmt % (casestr,'COORD'))
	lnodsfile = os.path.join(basedir, auxfile_fmt % (casestr,'LNODS'))
	ltypefile = os.path.join(basedir, auxfile_fmt % (casestr,'LTYPE'))
	lninvfile = os.path.join(basedir, auxfile_fmt % (casestr,'LNINV'))
	leinvfile = os.path.join(basedir, auxfile_fmt % (casestr,'LEINV'))
	lmastfile = os.path.join(basedir, auxfile_fmt % (casestr,'LMAST'))
	codnofile = os.path.join(basedir, auxfile_fmt % (casestr,'CODNO'))
	exnorfile = os.path.join(basedir, binfile_fmt % (casestr,'EXNOR',0))
	commufile = os.path.join(basedir, binfile_fmt % (casestr,'COMMU',0))
	massmfile = os.path.join(basedir, binfile_fmt % (casestr,'MASSM',0))
	npoints   = mesh.nnodT
	nelems    = mesh.nelT
	# Store node coordinates
	header = io.AlyaMPIO_header(
		fieldname   = 'COORD',
		dimension   = 'VECTO',
		association = 'NPOIN',
		npoints     = npoints,
		nsub        = nsub,
		ndims       = mesh._xyz.shape[1],
		itime       = 0,
		time        = 0
	)			
	header.dtype = mesh._xyz.dtype
	io.AlyaMPIO_write(coordfile,mesh._xyz,header,partfile,force_partition_data=force_partition_data,write=do_write)
	force_partition_data = False
	# Store element connectivity
	header = io.AlyaMPIO_header(
		fieldname   = 'LNODS',
		dimension   = 'VECTO',
		association = 'NELEM',
		npoints     = nelems,
		nsub        = nsub,
		ndims       = mesh._conec.shape[1],
		itime       = 0,
		time        = 0
	)			
	header.dtype = mesh._conec.dtype
	io.AlyaMPIO_write(lnodsfile,mesh._conec+1,header,partfile,force_partition_data=force_partition_data,write=do_write)
	# Store element type
	header = io.AlyaMPIO_header(
		fieldname   = 'LTYPE',
		dimension   = 'SCALA',
		association = 'NELEM',
		npoints     = nelems,
		nsub        = nsub,
		ndims       = 1,
		itime       = 0,
		time        = 0
	)			
	header.dtype = mesh._ltype.dtype
	io.AlyaMPIO_write(ltypefile,mesh._ltype,header,partfile,force_partition_data=force_partition_data,write=do_write)
	# Store global node ordering
	header = io.AlyaMPIO_header(
		fieldname   = 'LNINV',
		dimension   = 'SCALA',
		association = 'NPOIN',
		npoints     = npoints,
		nsub        = nsub,
		ndims       = 1,
		itime       = 0,
		time        = 0
	)			
	header.dtype = mesh._lninv.dtype
	io.AlyaMPIO_write(lninvfile,mesh._lninv+1,header,partfile,force_partition_data=force_partition_data,write=do_write)
	# Store global element ordering
	header = io.AlyaMPIO_header(
		fieldname   = 'LEINV',
		dimension   = 'SCALA',
		association = 'NELEM',
		npoints     = nelems,
		nsub        = nsub,
		ndims       = 1,
		itime       = 0,
		time        = 0
	)			
	header.dtype = mesh._leinv.dtype
	io.AlyaMPIO_write(leinvfile,mesh._leinv+1,header,partfile,force_partition_data=force_partition_data,write=do_write)
	# Store list of master nodes
	lmast_shape = mpi_reduce(mesh._lmast.shape[0],op='max',all=True)
	if lmast_shape > 1:
		header = io.AlyaMPIO_header(
			fieldname   = 'LMAST',
			dimension   = 'SCALA',
			association = 'NPOIN',
			npoints     = npoints,
			nsub        = nsub,
			ndims       = 1,
			itime       = 0,
			time        = 0
		)			
		header.dtype = mesh._lmast.dtype
		io.AlyaMPIO_write(leinvfile,mesh._lmast,header,partfile,force_partition_data=force_partition_data,write=do_write)
	# Store boundary codes
	codno_shape = mpi_reduce(mesh._codno.shape[0],op='max',all=True)
	if codno_shape > 1:
		header = io.AlyaMPIO_header(
			fieldname   = 'CODNO',
			dimension   = 'VECTO',
			association = 'NPOIN',
			npoints     = npoints,
			nsub        = nsub,
			ndims       = mesh._xyz.shape[1],
			itime       = 0,
			time        = 0
		)			
		header.dtype = mesh._codno.dtype
		io.AlyaMPIO_write(codnofile,mesh._codno,header,partfile,force_partition_data=force_partition_data,write=do_write)
	# Store element normals
	exnor_shape = mpi_reduce(mesh._exnor.shape[0],op='max',all=True)
	if exnor_shape > 1:
		header = io.AlyaMPIO_header(
			fieldname   = 'EXNOR',
			dimension   = 'VECTO',
			association = 'NPOIN',
			npoints     = npoints,
			nsub        = nsub,
			ndims       = mesh._xyz.shape[1],
			itime       = 0,
			time        = 0
		)			
		header.dtype = mesh._exnor.dtype
		io.AlyaMPIO_write(exnorfile,mesh._exnor,header,partfile,force_partition_data=force_partition_data,write=do_write)
	# Store communications matrix
	if mesh._comm is not None:
		commu = mesh._comm.to_commu(mesh.nnod) 
		header = io.AlyaMPIO_header(
			fieldname   = 'COMMU',
			dimension   = 'VECTO',
			association = 'NPOIN',
			npoints     = npoints,
			nsub        = nsub,
			ndims       = commu.shape[1],
			itime       = 0,
			time        = 0
		)			
		header.dtype = commu.dtype
		io.AlyaMPIO_write(commufile,commu,header,partfile,force_partition_data=force_partition_data,write=do_write)	
	# Store mass matrix
	header = io.AlyaMPIO_header(
		fieldname   = 'MASSM',
		dimension   = 'SCALA',
		association = 'NPOIN',
		npoints     = npoints,
		nsub        = nsub,
		ndims       = 1,
		itime       = 0,
		time        = 0
	)			
	header.dtype = mesh.volume.dtype
	io.AlyaMPIO_write(massmfile,mesh.volume,header,partfile,force_partition_data=force_partition_data,write=do_write)

def meshWriteENSIGHT(mesh,casestr,basedir):
	'''
	Write mesh in ENSIGHT format
	'''
	if MPI_SIZE > 1: raiseWarning('Ensight format not supposed to be working in parallel!')
	# Create the filename for the case
	geofile = os.path.join(basedir,'%s.ensi.geo'%casestr)
	header = {
		'descr'  : 'File created with pyQvarsi tool\nmesh file',
		'nodeID' : 'assign',
		'elemID' : 'assign',
		'partID' : 1,
		'partNM' : 'Volume Mesh',
		'eltype' : ''
	}
	# Build element type
	if mesh._ltype[0] == 10: header['eltype'] = 'tria3'
	if mesh._ltype[0] == 12: header['eltype'] = 'quad4'
	if mesh._ltype[0] == 30: header['eltype'] = 'tetra4'
	if mesh._ltype[0] == 34: header['eltype'] = 'penta6'
	if mesh._ltype[0] == 37: header['eltype'] = 'hexa8'
	# Write
	io.Ensight_writeGeo(geofile,mesh.xyz,mesh.connectivity+1,header)

def meshWriteVTKH5(mesh,casestr,basedir,linkfile):
	'''
	Write mesh in VTKH5 format
	'''
	# Create the filename for the case
	filename = os.path.join(basedir,io.VTKH5_FILE_FMT%casestr)
	linkname = os.path.join(basedir,io.VTKH5_FILE_FMT%linkfile)
	# Save in VTKH5
	if mesh.partition_table == None:
		write_master = False
	else:
		write_master=not mesh.partition_table.has_master
	if linkfile is None:
		io.vtkh5_save_mesh(filename,mesh.xyz,mesh.connectivity_vtk,mesh.eltype_linear,write_master=not mesh.partition_table.has_master)
	else:
		io.vtkh5_link_mesh(filename,linkname)

def vtkHigherOrderQuadrilateral_pointIndexFromIJ(mporder, i, j):
	"""
	Translates a point's indices in a higher-order quadrilateral to a point index.

	Parameters:
	mporder (int): The order of the quadrilateral.
	i (int): The i-th index of the point.
	j (int): The j-th index of the point.

	Returns:
	int: The point index.
	"""
	# Check if the indices are on the boundary
	ibdy = 1 if (i == 0 or i == mporder) else 0
	jbdy = 1 if (j == 0 or j == mporder) else 0

	nbdy = ibdy + jbdy

	pointIndex = 1

	# Vertex
	if nbdy == 2:
		if i != 0:
			if j != 0:
				pointIndex += 2
			else:
				pointIndex += 1
		else:
			if j != 0:
				pointIndex += 3

		return pointIndex

	pointIndex += 4

	# Edge
	if nbdy == 1:
		if ibdy == 0:  # i-axis
			pointIndex += (i - 1)
			if j != 0:
				pointIndex += (mporder * 2 - 2)
		else:  # j-axis
			pointIndex += (j - 1)
			if i != 0:
				pointIndex += (mporder - 1)
			else:
				pointIndex += (2 * (mporder - 1) + mporder - 1)

		return pointIndex

	# Body DOF
	pointIndex += (4 * mporder - 4)
	pointIndex += (i - 1) + (mporder - 1) * (j - 1)

	return pointIndex

def vtkHigherOrderHexahedron_pointIndexFromIJK(mporder, i, j, k):
	# Boundary flags
	ibdy = 1 if i == 0 or i == mporder else 0
	jbdy = 1 if j == 0 or j == mporder else 0
	kbdy = 1 if k == 0 or k == mporder else 0

	# Count how many boundaries are touched
	nbdy = ibdy + jbdy + kbdy

	# Initialize pointIndex
	pointIndex = 1

	# Vertex case
	if nbdy == 3:
		pointIndex += 2 if i != 0 and j != 0 else 1 if i != 0 else 3 if j != 0 else 0
		pointIndex += 4 if k != 0 else 0
		return pointIndex

	pointIndex += 8

	# Edge case
	if nbdy == 2:
		if ibdy == 0:  # i-axis
			pointIndex += (i - 1)
			pointIndex += (mporder * 2 - 2) if j != 0 else 0
			pointIndex += 2 * (mporder * 2 - 2) if k != 0 else 0
		elif jbdy == 0:  # j-axis
			pointIndex += (j - 1)
			pointIndex += (mporder - 1) if i != 0 else 2 * (mporder - 1) + mporder - 1
			pointIndex += 2 * (2 * mporder - 2) if k != 0 else 0
		else:  # k-axis
			pointIndex += 4 * (mporder - 1) + 4 * (mporder - 1)
			aux_pi = 2 if i != 0 and j != 0 else 1 if i != 0 else 3 if j != 0 else 0
			pointIndex += (k - 1) + (mporder - 1) * aux_pi
		return pointIndex

	# Face case
	pointIndex += 4 * (3 * mporder - 3)
	if nbdy == 1:
		if ibdy != 0:  # i-normal face
			pointIndex += (j - 1) + (mporder - 1) * (k - 1)
			pointIndex += (mporder - 1) ** 2 if i != 0 else 0
			return pointIndex
		pointIndex += 2*(mporder-1)**2
		if jbdy != 0:  # j-normal face
			pointIndex += (i - 1) + (mporder - 1) * (k - 1)
			pointIndex += (mporder - 1) ** 2 if j != 0 else 0
			return pointIndex
		# k-normal face
		pointIndex += 2 * (mporder - 1) ** 2
		pointIndex += (i - 1) + (mporder - 1) * (j - 1)
		pointIndex += (mporder - 1) ** 2 if k != 0 else 0
		return pointIndex

	# Body DOF case
	pointIndex += 2 * ((mporder - 1) ** 2 * 3)
	pointIndex += (i - 1) + (mporder - 1) * ((j - 1) + (mporder - 1) * (k - 1))

	return pointIndex

linearizedElements ={
	# Linear cells
	# Ref: https://github.com/Kitware/VTK/blob/master/Common/DataModel/vtkCellType.h
	15 : 12 , # Lagrangian quadrangle is now a Quadrangular cell
	40 : 37,  # Lagrangian hexahedron is now a linear hexahedron
}