#!/usr/bin/env python
#
# pyQvarsi, field.
#
# Field class, reader and reduction routines.
#
# Last rev: 03/03/2021
from __future__ import print_function, division

import os, copy, numpy as np

from .                import inp_out as io, vmath as math
from .cr              import cr
from .mem             import mem
from .partition_table import PartitionTable
from .utils.common    import raiseWarning, raiseError
from .utils.parallel  import MPI_RANK, MPI_SIZE, mpi_create_op, mpi_reduce


POS_KEYS = ['xyz','coords','pos']

# Fast reduction operator for nodes class
fieldFastReduce = mpi_create_op(lambda f1,f2,dtype : f1.join(f2).clean(), commute=True)

class Field(object):
	'''
	The Field class wraps the position of the nodes and a number of variables
	and relates them so that the operations in parallel are easier.
	'''
	@cr('field.init')
	def __init__(self, xyz=np.array([[]],np.double), instant=0, time=0., ptable=None, **kwargs):
		'''
		Class constructor (self, xyz=np.array([[]]), **kwargs)

		Inputs:
			> xyz:   position of the nodes as a numpy array of 3 dimensions.
			> kwags: dictionary containin the variable name and values as a
					 python dictionary.
		'''
		self._xyz     = xyz
		self._vardict = kwargs
		self._instant = instant
		self._time    = time
		self._ptable  = ptable

	def __len__(self):
		return self._xyz.shape[0]

	def __str__(self):
		'''
		String representation
		'''
		if self._xyz.shape[0] > 0:
			s   = 'Field of %d points:\n' % len(self)
			s  += '  > xyz  - max = ' + str(np.nanmax(self._xyz,axis=0)) + ', min = ' + str(np.nanmin(self._xyz,axis=0)) + '\n'
			for key in self.varnames:
				var = self[key]
				nanstr = ' (has NaNs) ' if np.any(np.isnan(var)) else ' '
				s  += '  > ' +  key + nanstr + '- max = ' + str(np.nanmax(var,axis=0)) \
											 + ', min = ' + str(np.nanmin(var,axis=0)) \
											 + ', avg = ' + str(np.nanmean(var,axis=0)) \
											 + '\n'
		else:
			s = 'Empty field'
		return s
		
	# Set and get functions
	def __getitem__(self,key):
		'''
		Field[key]

		Recover the value of a variable given its key
		'''
		return self._xyz if key in POS_KEYS else self._vardict[key]

	def __setitem__(self,key,value):
		'''
		Field[key] = value

		Set the value of a variable given its key
		'''
		if key in POS_KEYS:
			self._xyz = value
		else:
			self._vardict[key] = value

	# Operators
	def __eq__(self, other):
		'''
		Field1 == Field2

		Two nodes are equal if all the coordinates 
		are the same
		'''	
		return np.all(self._xyz == other._xyz)

	@cr('field.add',1)
	def __add__(self, other):
		'''
		Field1 + Field2

		In case of Field1 == Field2 the same order
		is expected.
		'''
		new = copy.deepcopy(self)
		if new == other:
			# Both are the same so we can simply sum
			# the variables
			for key in new._vardict.keys():
				new.var[key] += other.var[key]
		else:
			# Both are not the same 
			# sum those that are equal
			idx = []
			# Loop the points in the other partition (smaller or equal)
			for ii in range(len(other)):
				# Find if the point is on the first partition
				ieq = new.find(other.xyz[ii,:])
				if not len(ieq) == 0:
					for key in new.varnames:
						new.var[key][ieq,] += other.var[key][ii,]
				else:
					idx.append(ii)
			if not idx == []: new.join(other,idx=idx)
		return new.clearNaN()

	@cr('field.add',2)
	def __iadd__(self, other):
		'''
		Field1 += Field2

		In case of Field1 == Field2 the same order
		is expected.
		'''
		if self == other:
			# Both are the same so we can simply sum
			# the variables
			for key in self._vardict.keys():
				self.var[key] += other.var[key]
		else:
			# Both are not the same 
			# sum those that are equal
			idx = []
			# Loop the points in the other partition (smaller or equal)
			for ii in range(len(other)):
				# Find if the point is on the first partition
				ieq = self.find(other.xyz[ii,:])
				if not len(ieq) == 0:
					for key in self.varnames:
						self.var[key][ieq,] += other.var[key][ii,]
				else:
					idx.append(ii)
			if not idx == []: self.join(other,idx=idx)
		return self.clearNaN()

	def __sub__(self, other):
		'''
		Field1 - Field2

		In case of Field1 == Field2 the same order
		is expected.
		'''
		new = copy.deepcopy(self)
		if new == other:
			for key in new.varnames:
				new.var[key] -= other.var[key]
		else:
			raise NotImplementedError 
		return new

	def __mul__(self,other):
		'''
		Field*val
		'''
		new = copy.deepcopy(self)
		for key in new.varnames:
			new.var[key] *= other
		return new

	def __rmul__(self,other):
		'''
		val*Field
		'''
		return self.__mul__(other)

	def __imul__(self,other):
		'''
		Field *= val
		'''
		for key in self.varnames:
			self.var[key] *= other
		return self

	def __truediv__(self,other):
		'''
		Field/val
		'''
		new = copy.deepcopy(self)
		for vname in new.varnames:
			if len(new.var[vname].shape) > 1:
				for ii in range(new.var[vname].shape[1]):
					new.var[vname][:,ii] /= other
			else:
				new.var[vname] /= other
		return new

	def __itruediv__(self,other):
		'''
		Field /= val
		'''
		for vname in self.varnames:
			if len(self.var[vname].shape) > 1:
				for ii in range(self.var[vname].shape[1]):
					self.var[vname][:,ii] /= other
			else:
				self.var[vname] /= other
		return self

	# Functions
	def ndim(self,var):
		'''
		Return the dimensions of a variable on the field.
		'''
		return self.var[var][0].size

	@cr('field.find')
	def find(self,xyz):
		'''
		Return all the points where self._xyz == xyz
		'''
		return np.where(np.all(self._xyz == xyz,axis=1))[0]

	@cr('field.join')
	def join(self,other,idx=[],varnames=[]):
		'''
		Joins two classes with certain idx if provided
		'''
		if varnames == []: varnames = self.varnames
		# Joining algorithm
		if idx == []:
			self._xyz = np.append(self._xyz,other._xyz,axis=0)
			for key in varnames:
				self.var[key] = np.append(self.var[key],other.var[key],axis=0)
		else:
			self._xyz = np.append(self._xyz,other._xyz[idx],axis=0)
			for key in varnames:
				self.var[key] = np.append(self.var[key],other.var[key][idx],axis=0)
		return self

	@cr('field.sort')
	def sort(self,array='x',as_idx=False,**kwargs):
		'''
		Sort arrays according to a 
		direction or an array
		'''
		idx = []
		if isinstance(array,str):
			if array == 'x':
				idx = np.argsort(self.x,**kwargs)
			if array == 'y':
				idx = np.argsort(self.y,**kwargs)
			if array == 'z':
				idx = np.argsort(self.z,**kwargs)
			if array in POS_KEYS:
				idx = np.argsort(self.xyz,**kwargs)
			if array in self.varnames:
				idx = np.argsort(self.var[array],**kwargs)
		else:
			# Else use the provided array to obtain idx
			idx = np.argsort(array,**kwargs) if not as_idx else array

		# Sort
		self._xyz = self._xyz[idx]
		for vname in self.varnames:
			self.var[vname] = self.var[vname][idx]
		return self

	@cr('field.clean')
	def clean(self):
		'''
		Erase duplicated node values and its arrays
		'''
		self._xyz, idx = np.unique(self._xyz, axis=0, return_index=True)
		for vname in self.varnames:
			self.var[vname] = self.var[vname][idx]
		return self

	@cr('field.clearNaN')
	def clearNaN(self):
		'''
		Erases NaNs in the field
		'''
		idx = np.where(np.isnan(self._xyz))[0]
		self._xyz = np.delete(self._xyz,idx,axis=0)
		for vname in self.varnames:
			self.var[vname] = np.delete(self.var[vname],idx,axis=0)
		return self

	@cr('field.filter_bc')
	def filter_bc(self,mesh):
		'''
		Returns a field without repeated boundary
		nodes. 
		'''
		out = Field(xyz=mesh.filter_bc(self.xyz))
		for v in self.varnames:
			out[v] = mesh.filter_bc(self.var[v])
		return out

	def rename(self,new,old):
		'''
		Rename a variable inside a field.
		'''
		self.var[new] = self.var.pop(old)
		return self

	def delete(self,varname):
		'''
		Delete a variable inside a field.
		'''
		return self.var.pop(varname)

	@cr('field.rotate')
	def rotate(self,angles,center=np.array([0.,0.,0.],np.double)):
		'''
		Rotate the mesh coordinates given the Euler angles and a center.
		'''
		self._xyz = math.vecRotate(self._xyz,angles[0],angles[1],angles[2],center)
		# We only need to add the displacement of the center to the points
		# the arrays are only affected by the rotation
		for var in self._vardict.keys():
			if len(self[var].shape) == 1: continue # skip scalar arrays
			if self[var].shape[1] == 3: # Vectorial array
				self[var] = math.vecRotate(self[var],angles[0],angles[1],angles[2],np.array([0.,0.,0.],np.double))
			if self[var].shape[1] == 9: # Tensorial array
				self[var] = math.tensRotate(self[var],angles[0],angles[1],angles[2])

	@cr('field.reduce')
	def reduce(self,root=-1,op='sum'):
		'''
		Reduce a field using MPI_Allreduce if root is negative
		or MPI_Reduce if a root processor is selected.
		'''
		# Reduce returns None if root != rank
		out = mpi_reduce(self,root=root,op=op,all=True if root < 0 else False)
		# Only clean NaNs when we have actual data
		if root < 0 or root == MPI_RANK: out.clearNaN()
		return out

	@cr('field.selectMask')
	def selectMask(self,mask):
		'''
		Given a masking array, it returns a new Field with the
		cropped values that are inside the region defined by the
		mask.
		'''
		# If all the points are outside the defined region generate an
		# empty field.
		field_new = self.__class__(xyz=self.xyz[mask].copy() if np.any(mask) else np.empty((0,3)) if self.xyz.shape[0] > 1 else np.nan*np.ones((1,3),dtype=np.double))
		# Now store each of the variables inside the field
		for var in self.varnames:
			if np.any(np.isnan(field_new.xyz)):
				field_new[var] = np.nan*np.ones((1,self.var[var].shape[0]))
			else:
				field_new[var] = self.var[var][mask].copy() if np.any(mask) else np.empty((0,self.var[var].shape[1]) if len(self.var[var].shape) > 1 else (0,),dtype=np.double)
		# Return the new field
		return field_new

	@cr('field.select')
	def select(self,poly):
		'''
		Given an entity of pyQvarsi.Geom, it returns a new Field with the
		cropped values that are inside the region defined by the
		geometric element.
		'''
		# First check if the points contained in this field are inside
		# the defined region and obtain the mask.
		mask = poly.areinside(self.xyz)
		# Now we can generate a new Field
		field_new = self.selectMask(mask)
		# Return the new field
		return field_new

	@cr('field.clip')
	def clip(self,start=0,end=-1):
		'''
		Clip a field between two indices.
		'''
		mask = np.zeros((len(self),),dtype=bool)
		mask[start:end] = True
		return self.selectMask(mask)

	@cr('field.interp')
	def interpolate(self,poly):
		'''
		Given an entity of pyQvarsi.Geom, it returns a new Field with the
		interpolated values that are in the points defined by the
		geometric element.
		'''
		# First check if the points contained in this field are inside
		# the defined region and obtain the mask.
		mask = poly.areinside(self.xyz)
		# Create a new field with the points of the interpolating poly
		xyz = np.array([p.xyz for p in poly.points],dtype=np.double) if np.any(mask) else np.nan*np.array([[0.,0.,0.]],dtype=np.double)
		field_new = Field(xyz=xyz)
		# Interpolate variables to poly
		for var in self.varnames:
			field_new[var] = poly.interpolate(self.xyz[mask],self.var[var][mask]) if np.any(mask) else np.nan*np.zeros((1,self.var[var].shape[1]) if len(self.var[var].shape) > 1 else (1,),dtype=np.double)
		# Return the new field
		return field_new

	@cr('field.save')
	def save(self,fname,**kwargs):
		'''
		Store the field in various formats.
		'''
		# Guess format from extension
		fmt = os.path.splitext(fname)[1][1:] # skip the .
		# Pickle format
		if fmt.lower() == 'pkl': 
			io.pkl_save(fname,self)
		# H5 format
		if fmt.lower() == 'h5':
			# Set default parameters
			if not 'metadata'     in kwargs.keys(): kwargs['metadata']     = {}
			if not 'mpio'         in kwargs.keys(): kwargs['mpio']         = True
			if not 'write_master' in kwargs.keys(): kwargs['write_master'] = not self._ptable.has_master
			io.h5_save_field(fname,self.partition_table,self.xyz,self.instant,self.time,self.var,**kwargs)

	@cr('field.write')
	def write(self,casestr,basedir='./',exclude_vars=[],fmt='vtkhdf',nsub=max(MPI_SIZE-1,1),
		force_partition_data=False,use_post=True,tags=[-1,-1],write_rank=0):
		'''
		Store the data in the field using various formats.

		This method differs from save in the fact that save is used
		to recover the field, write only outputs the data.
		'''
		if fmt.lower() == 'mpio':
			fieldWriteMPIO(self,casestr,basedir,self.instant,self.time,nsub,exclude_vars,
				force_partition_data,use_post)
		elif fmt.lower() == 'serial':
			nsub = 1 # Forcing just one subdomain
			fieldWriteMPIOserial(self,casestr,basedir,self.instant,self.time,nsub,exclude_vars,
				force_partition_data,use_post,tags=tags,write_rank=write_rank)
		elif fmt.lower() in ['ensi','ensight']:
			fieldWriteENSIGHT(self,casestr,basedir,self.instant,self.time,nsub,exclude_vars)
		elif fmt.lower() in ['vtkhdf','vtkh5']:
			fieldWriteVTKH5(self,casestr,basedir,self.instant,self.time,exclude_vars)
		else:
			raiseError('Format <%s> not implemented!'%fmt)

	@classmethod
	@cr('field.like')
	def field_like(cls,f,idx=[],varnames=[],copy=True):

		'''
		Create another field from an existing one
		'''
		if varnames == []: varnames = f.varnames
		if idx == []:
			out = cls(ptable=f.partition_table,xyz=f.xyz.copy(),instant=f.instant,time=f.time)
			if copy:
				for v in varnames:
					out[v] = f[v].copy()
		else:
			out = cls(ptable=f.partition_table,xyz=f.xyz[idx].copy(),instant=f.instant,time=f.time)
			if copy:
				for v in varnames:
					out[v] = f[v][idx].copy()			
		return out

	# Properties
	@property
	def xyz(self):
		return self._xyz
	@xyz.setter
	def xyz(self,value):
		self._xyz = value

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
	def npoints(self):
		return self._xyz.shape[0]
	@property
	def instant(self):
		return self._instant
	@property
	def time(self):
		return self._time
	@property
	def partition_table(self):
		return self._ptable
	
	@property
	def var(self):
		return self._vardict
	@property
	def varnames(self):
		return list(self._vardict.keys())
	
class FieldAlya(Field):
	'''
	The Field class wraps the position of the nodes and a number of variables
	and relates them so that the operations in parallel are easier.
	'''
	def __init__(self, ptable=None, xyz=np.array([[]],np.double), instant=0, time=0., serial=False, **kwargs):
		'''
		Class constructor (self, xyz=np.array([[]]), **kwargs)

		Inputs:
			> xyz:   position of the nodes as a numpy array of 3 dimensions.
			> kwags: dictionary containin the variable name and values as a
					 python dictionary.
		'''
		super(FieldAlya,self).__init__(xyz,instant,time,ptable=ptable,**kwargs)
		self._ptable = self.generate_partition_table(ptable,serial=False) if self._ptable is None else self._ptable


	@cr('fieldAlya.ptable')
	def generate_partition_table(self,ptable=None,serial=False):
		'''
		Generate the partition table if it does not exist
		'''
		# Return the partition table if it is correct
		if ptable is not None: return ptable
		# Create a new partition table and scatter it
		# to all the processors
		return PartitionTable.fromField(self,has_master=True) if not serial else PartitionTable.new(1,npoints=self.xyz.shape[0])

	@classmethod	
	@cr('fieldAlya.read')
	def read(cls,casestr,varList,instant,coords,fmt='mpio',basedir='./',force_partition_data=False,use_post=True):
		'''
		Read mpio files and return a Field class.

		IN:
			> casestr: name of the Alya case.
			> varList: list of the variables to read.
			> instant: instant to read.
			> coords:  coordinates of the nodes.
			> basedir: (optional) main directory of the simulation.

		OUT:
			> field:   instance of the field class.
			> header:  header of the mpio file.
		'''
		if fmt.lower() == 'mpio':
			varDict, header, ptable = fieldReadMPIO(casestr,varList,instant,basedir,force_partition_data,use_post)
		elif fmt.lower() in ['ensi','ensight']:
			varDict, header, ptable = fieldReadENSIGHT(casestr,varList,instant,basedir,coords.shape[0],force_partition_data)
		elif fmt.lower() == 'vtk':
			varDict, header, ptable = fieldReadVTK(casestr,varList,instant,basedir,coords.shape[0],force_partition_data)
		else:
			raiseError('Format <%s> not implemented!'%fmt)
		return cls(ptable=ptable,xyz=coords,instant=instant,time=header.time,**varDict), header
	
	@classmethod
	@cr('field.load')
	def load(cls,fname,**kwargs):
		'''
		Load a field from various formats
		'''
		# Guess format from extension
		fmt = os.path.splitext(fname)[1][1:] # skip the .
		# Pickle format
		if fmt.lower() == 'pkl': 
			return io.pkl_load(fname)
		# H5 format
		if fmt.lower() == 'h5':
			ptable, xyz, instant, time, varDict = io.h5_load_field(fname,mpio=kwargs.get('mpio',True),vars=kwargs.get('vars',[]),inods=kwargs.get('inods',None))
			return cls(ptable=ptable,xyz=xyz,instant=instant,time=time,**varDict)
		raiseError('Cannot load file <%s>!'%fname)

class FieldSOD2D(Field):
	'''
	The Field class wraps the position of the nodes and a number of variables
	and relates them so that the operations in parallel are easier.
	'''
	def __init__(self, ptable=None, xyz=np.array([[]],np.double), instant=0, time=0., serial=False, **kwargs):
		'''
		Class constructor (self, xyz=np.array([[]]), **kwargs)

		Inputs:
			> xyz:   position of the nodes as a numpy array of 3 dimensions.
			> kwags: dictionary containin the variable name and values as a
					 python dictionary.
		'''
		super(FieldSOD2D,self).__init__(xyz,instant,time,ptable=ptable,**kwargs)
		self._ptable = self.generate_partition_table(ptable,serial=serial) if self._ptable is None else self._ptable


	@cr('fieldSOD2D.ptable')
	def generate_partition_table(self,ptable=None,serial=False):
		'''
		Generate the partition table if it does not exist
		'''
		# Return the partition table if it is correct
		if ptable is not None: return ptable
		# Create a new partition table and scatter it
		# to all the processors
		return PartitionTable.fromField(self,has_master=False) if not serial else PartitionTable.new(1,npoints=self.xyz.shape[0])

	@classmethod	
	@cr('fieldSOD2D.read')
	def read(cls,casestr,varList,instant,coords,nprocs=MPI_SIZE,basedir='./',force_partition_data=False):
		'''
		Read mpio files and return a Field class.

		IN:
			> casestr: name of the Alya case.
			> varList: list of the variables to read.
			> instant: instant to read.
			> coords:  coordinates of the nodes.
			> basedir: (optional) main directory of the simulation.

		OUT:
			> field:   instance of the field class.
			> header:  header of the mpio file.
		'''
		# SOD2D does not have a header, instead the time is returned
		ptable          = PartitionTable.fromSOD2DHDF(casestr,basedir)
		varDict, header = io.SOD2DHDF_read_results(casestr,varList,instant,nprocs,basedir=basedir,force_partition_data=force_partition_data)
		return cls(ptable=ptable,xyz=coords,instant=instant,time=header.time,**varDict)
	
	@cr('fieldSOD2D.write')
	def write(self,casestr,basedir='./',exclude_vars=[],nsub=max(MPI_SIZE-1,1),force_partition_data=False,use_post=True,tags=[-1,-1],write_rank=0):
		super(FieldSOD2D, self).write(casestr,fmt='vtkh5',basedir=basedir,exclude_vars=exclude_vars,nsub=nsub,force_partition_data=force_partition_data,use_post=use_post,tags=tags,write_rank=write_rank)


def fieldReadMPIO(casestr,varList,instant,basedir,force_partition_data,use_post):
	'''
	Read field in MPIO format.
	'''
#	auxfile_fmt = io.MPIO_AUXFILE_P_FMT if MPI_SIZE > 1 or use_post else io.MPIO_AUXFILE_S_FMT
	binfile_fmt = io.MPIO_BINFILE_P_FMT if MPI_SIZE > 1 or use_post else io.MPIO_BINFILE_S_FMT
	partfile    = os.path.join(basedir,io.MPIO_PARTFILE_FMT % casestr)
	varDict     = {}
	# Loop the requested variables
	for var in varList:
		ii           = instant if not var in io.MPIO_NOTEMP_VARS else 0
		filename     = os.path.join(basedir,binfile_fmt % (casestr,var,ii))
		data, header = io.AlyaMPIO_read(filename,partfile,force_partition_data=force_partition_data)
		varDict[var] = data
		force_partition_data = False
	# Partition table
	ptable  = PartitionTable.fromAlya(casestr,basedir)
	# Return
	return varDict, header, ptable

def fieldReadENSIGHT(casestr,varList,instant,basedir,nnod,force_partition_data):
	'''
	Read field in Ensight format.
	'''
	if MPI_SIZE > 1: raiseWarning('Ensight format not supposed to be working in parallel!')
	# Read CASE
	if force_partition_data or (not hasattr(fieldReadENSIGHT,'vars')):
		casefile = os.path.join(basedir,'%s.ensi.case' % casestr)
		if not os.path.isfile(casefile): casefile = os.path.join(basedir,'%s.case'%casestr)
		fieldReadENSIGHT.vars, fieldReadENSIGHT.tsteps = io.Ensight_readCase(casefile)
	varDict     = {}
	# Loop the requested variables
	for var in varList:
		vDict = {}
		# Find which variable in the casefile
		for v in fieldReadENSIGHT.vars:
			if v['name'] == var:
				vDict = v
				break
		# Build the filename
		#nstars       = vDict['file'].count('*')
		istar        = vDict['file'].find('*')
		filename_fmt = vDict['file'].replace(vDict['file'][istar:],'%%0%dd'%vDict['file'].count('*'))
		filename     = os.path.join(basedir,filename_fmt % instant)
		data, header = io.Ensight_readField(filename,vDict['dims'],nnod)
		varDict[var] = data
	header = io.AlyaMPIO_header()
	header.time  = fieldReadENSIGHT.tsteps[instant]
	header.itime = instant
	# Partition table
	ptable = None
	# Return
	return varDict, header, ptable

def fieldReadVTK(casestr,varList,instant,basedir,nnod,force_partition_data):
	'''
	Read field in VTK format.
	'''
	filename = os.path.join(basedir,casestr)
	# Get VTK data and print info
	dataVTK = io.vtkIO(filename=filename,mode='read',varlist=varList)
	# Convert cell data to point data
	io.vtkCelltoPointData(dataVTK,varList,keepdata=False)
	# Get VTK data in numpy format directly
	data = dataVTK.get_vars(varList)
	# Rearrange variables dictionary
	varDict = {}
	for var in varList:
		varDict[var] = data[var+'_pointdata']
	# Define header structure
	header       = io.AlyaMPIO_header()
	header.time  = 0.0 # TODO: recover in VTK
	header.itime = instant
	# Partition table
	ptable = None
	# Return
	return varDict, header, ptable

def fieldWriteMPIO(field,casestr,basedir,instant,time,nsub,exclude_vars,force_partition_data,use_post):
	'''
	Write field in MPIO format.
	'''
	do_write = not np.any(np.isnan(field.xyz)) # When not to write
	# Alya MPIO binaries - formats
	#auxfile_fmt = io.MPIO_AUXFILE_P_FMT if MPI_SIZE > 1 or use_post else io.MPIO_AUXFILE_S_FMT
	binfile_fmt = io.MPIO_BINFILE_P_FMT if MPI_SIZE > 1 or use_post else io.MPIO_BINFILE_S_FMT
	# Alya MPIO binaries
	partfile = os.path.join(basedir,io.MPIO_PARTFILE_FMT % casestr)
	npoints  = mpi_reduce(field.xyz.shape[0],op='sum',all=True) - 1 if nsub > 1 else field.xyz.shape[0] # To take out the master
	# Loop each variable on the field
	for var in field.varnames:
		if var in exclude_vars: continue # Skip if excluded
		# Generate the filename
		filename = os.path.join(basedir,binfile_fmt % (casestr,var,instant))
		# Compute dimension
		ndims     = field.var[var].shape[1] if len(field.var[var].shape) > 1 else 1
		dimension = 'SCALA' if ndims == 1 else 'VECTO'
#		dimension = 'MATRI'
#		if ndims == 1: dimension = 'SCALA'
#		if ndims == 3: dimension = 'VECTO'
		# Create the header for MPIO
		header = io.AlyaMPIO_header(
			fieldname   = var,
			dimension   = dimension,
			association = 'NPOIN', # For now...
			npoints     = npoints,
			nsub        = nsub,
			ndims       = ndims,
			itime       = instant,
			time        = time
		)
		header.dtype = field.var[var].dtype
		# Write MPIO file
		io.AlyaMPIO_write(filename,field.var[var],header,partfile,force_partition_data=force_partition_data,write=do_write)
		force_partition_data = False

def fieldWriteMPIOserial(field,casestr,basedir,instant,time,nsub,exclude_vars,
	force_partition_data,use_post,tags=[-1,-1],write_rank=0):
	'''
	Write field in MPIO format serial case.
	'''
	auxfile_fmt = io.MPIO_AUXFILE_P_FMT if use_post else io.MPIO_AUXFILE_S_FMT
	binfile_fmt = io.MPIO_BINFILE_P_FMT if use_post else io.MPIO_BINFILE_S_FMT
	# Alya MPIO binaries
#	partfile = os.path.join(basedir,io.MPIO_PARTFILE_FMT % casestr)
	npoints  = field.xyz.shape[0] # To take out the master
	# Loop each variable on the field
	for var in field.varnames:
		if var in exclude_vars: continue # Skip if excluded
		# Generate the filename
		filename = os.path.join(basedir,binfile_fmt % (casestr,var,instant))
		# Compute dimension
		ndims     = field.var[var].shape[1] if len(field.var[var].shape) > 1 else 1
		dimension = 'SCALA' if ndims == 1 else 'VECTO'
#		dimension = 'MATRI'
#		if ndims == 1: dimension = 'SCALA'
#		if ndims == 3: dimension = 'VECTO'
		# Create the header for MPIO
		header = io.AlyaMPIO_header(
			fieldname   = var,
			dimension   = dimension,
			sequence    = 'SEQUE',
			association = 'NPOIN', # For now...
			npoints     = npoints,
			nsub        = nsub,
			ndims       = ndims,
			itime       = instant,
			tag1        = tags[0],
			tag2        = tags[1],
			time        = time
		)
		header.dtype = field.var[var].dtype
		# Write MPIO file
		io.AlyaMPIO_write_serial(filename,field.var[var],header,rank=write_rank)
		force_partition_data = False

def fieldWriteENSIGHT(field,casestr,basedir,instant,time,nsub,exclude_vars):
	'''
	Write field in ENSIGHT format.
	'''
	if MPI_SIZE > 1: raiseWarning('Ensight format not supposed to be working in parallel!')
	binfile_fmt = '%s.ensi.%s-%06d'
	# Define Ensight header	
	header = {'descr':'File created with pyQvarsi tool','partID':1,'partNM':'part'}
	# Loop each variable on the field
	for var in field.varnames:
		if var in exclude_vars: continue # Skip if excluded
		# Generate the filename
		filename = os.path.join(basedir,binfile_fmt % (casestr,var,instant))
		# Write ENSIGHT file
		io.Ensight_writeField(filename,field.var[var],header)

def fieldWriteVTKH5(field,casestr,basedir,instant,time,exclude_vars):
	'''
	Write field in ENSIGHT format.
	'''
	filename = os.path.join(basedir,io.VTKH5_FILE_FMT%casestr)
	varDict  = {}
	# Loop each variable on the field
	for var in field.varnames:
		if var in exclude_vars: continue # Skip if excluded
		varDict[var] = field[var]
	# Write file
	io.vtkh5_save_field(filename,instant,time,varDict,mpio= not field.partition_table.is_serial,write_master=not field.partition_table.has_master)
