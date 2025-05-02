#!/usr/bin/env python
#
# Alya MPIO Input/Output
#
# Source: https://gitlab.com/bsc-alya/tools/alya-mpio-tools/-/wikis/MPI-IO-binary-format-specifications
#
# Last rev: 03/03/2021
from __future__ import print_function, division

import numpy as np, struct

from ..cr             import cr
from ..mem            import mem
from ..utils.common   import raiseError, raiseWarning
from ..utils.parallel import MPI_RANK, MPI_SIZE, MPI_COMM, MPI_RDONLY, MPI_WRONLY, MPI_CREATE
from ..utils.parallel import mpi_file_open, mpi_bcast, mpi_gather


ALYA_MPIO_ERROR_NSUBD = False # Raise an error if nsubd != MPI_SIZE
ALYA_MPIO_WARNG_NSUBD = True  # Raise a warning if MPI_SIZE > nsubd

ALYA_MPIO_MAGIC   = 27093
ALYA_MPIO_FMT     = 'MPIAL00'
ALYA_MPIO_VERS    = 'V000400'
ALYA_HEADER_BYTES = 256


class AlyaMPIO_header(object):
	'''
	Class to manage the header of a MPIO binary.
	++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	Magic number (signed int 64 bits):        27093
	Format (char* 64 bits):                   MPIAL00 
	Version (char* 64 bits):                  V000400
	Object (char* 64 bits):                   COORD00/LTYPE00/LNODS00/LTYPB00...
	Dimension (char* 64 bits)                 SCALA00/VECTO00
	Results on (char* 64 bits)                NELEM00/NPOIN00/NBOUN00
	Type int/real (char* 64 bits):            INTEG00/REAL000
	Size (char* 64 bits):                     4BYTE00/8BYTE00
	Seq/Parallel (char* 64 bits)              SEQUE00/PARAL00
	Filter or no filter (char* 64 bits)       FILTE00/NOFIL00
	Sorting (char* 64 bits):                  ASCEN00/DESCE00/NONE000
	Id (char* 64 bits):                       ID00000/NOID000
	--Alignment (char* 64 bits):              0000000
	Columns (signed int 64 bits):             ncolumns (id not counted)
	Lines (signed int 64 bits):               nlines
	Time step number (signed int 64 bits):    ittim
	Number of subdomains (signed int 64 bits):nsubd (1=SEQUENTIAL)
	Mesh division (signed int 64 bits):       divi
	Tag 1 (signed int 64 bits):               tag1
	Tag 2 (signed int 64 bits):               tag2
	Time (real 64 bits):                      time
	--Alignment (char* 64 bits):              0000000
	Option 1 (char* 64 bits):                 OPTION1 (each option is a character string of length 7+1)
	Option 2 (char* 64 bits):                 OPTION2 (each option is a character string of length 7+1)
	Option 3 (char* 64 bits):                 OPTION3 (each option is a character string of length 7+1)
	Option 4 (char* 64 bits):                 OPTION4 (each option is a character string of length 7+1)
	Option 5 (char* 64 bits):                 OPTION5 (each option is a character string of length 7+1)
	Option 6 (char* 64 bits):                 OPTION6 (each option is a character string of length 7+1)
	Option 7 (char* 64 bits):                 OPTION7 (each option is a character string of length 7+1)
	Option 8 (char* 64 bits):                 OPTION8 (each option is a character string of length 7+1)
	Option 9 (char* 64 bits):                 OPTION9 (each option is a character string of length 7+1)
	Option 10 (char* 64 bits):                OPTION10 (each option is a character string of length 7+1)
	++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	Total: 64+64*12+64*7+64+64+10*64 = 2048 bits = 256 bytes
	'''
#	@mem('AlyaMPIO.header')
	def __init__(self,
		fieldname   = 'UNKNO',
		dimension   = 'SCALA',
		association = 'NPOIN',
		dtype       = 'REAL',
		size        = '8BYTE',
		sequence    = 'PARAL' if MPI_SIZE > 1 else 'SEQUE',
		filt        = 'NOFIL',
		sorting     = 'ASCEN',
		id          = 'NOID',
		npoints     = 1,
		ndims       = 1,
		itime       = 0,
		nsub        = max(MPI_SIZE-1,1), # To account for the master
		division    = 0,
		tag1        = -1,
		tag2        = -1,
		time        = 0.,
		ignore_err  = False):
		'''
		Class constructor
		'''
		self._nbytes = ALYA_HEADER_BYTES
		# Check inputs
		if not dimension   in ['SCALA','VECTO','MATRI']: raiseError('Dimension <%s> not valid!'%dimension)
		if not association in ['NELEM','NPOIN','NBOUN']: raiseError('Association <%s> not valid!'%association)
		if not dtype       in ['INTEG','REAL']:          raiseError('Type <%s> not valid!'%dtype)
		if not size        in ['4BYTE','8BYTE']:         raiseError('Size <%s> not valid!'%size)
		if not sequence    in ['SEQUE','PARAL']:         raiseError('Sequence <%s> not valid!'%sequence)
		if not filt        in ['FILTE','NOFIL']:         raiseError('Filter <%s> not valid!'%filt)
		if not sorting     in ['ASCEN','DESCE','NONE']:  raiseError('Sorting <%s> not valid!'%sorting)
		if not id          in ['ID'   ,'NOID']:          raiseError('Id <%s> not valid!'%id)
		if not ignore_err:
			if ALYA_MPIO_ERROR_NSUBD and not nsub == max(MPI_SIZE-1,1): raiseError('Incorrect number of subdomains %d!'%nsub)
			if ALYA_MPIO_WARNG_NSUBD and not nsub == max(MPI_SIZE-1,1): raiseWarning('Inconsistent number of subdomains %d!'%nsub)
		# Set the header dict with the default parameters
		self._header = {
			'Magic'    : ALYA_MPIO_MAGIC,
			'Format'   : ALYA_MPIO_FMT,
			'Version'  : ALYA_MPIO_VERS,
			'Object'   : fieldname,
			'Dimension': dimension,   # SCALA00/VECTO00/MATRI00
			'ResultsOn': association, # NELEM00/NPOIN00/NBOUN00
			'Type'     : dtype,       # INTEG00/REAL000
			'Size'     : size,        # 4BYTE00/8BYTE00
			'Sequence' : sequence,    # SEQUE00/PARAL00
			'Filter'   : filt,        # FILTE00/NOFIL00
			'Sorting'  : sorting,     # ASCEN00/DESCE00/NONE000
			'Id'       : id,          # ID00000/NOID000
			'Columns'  : ndims,
			'Lines'    : npoints,
			'TstepNo'  : itime,
			'NSubdom'  : nsub,
			'Division' : division,
			'Tag1'     : tag1,
			'Tag2'     : tag2,
			'Time'     : time
		}

	@classmethod
	def from_file(cls,file):
		'''
		Read MPIO Header
		'''
		# Define a header class
		header = cls()
		# Read the entire header in a buffer
		buff = bytearray(header.nbytes)
		file.Read(buff)
		# Ask the header class to parse the buffer
		header.parse(buff)
		return header

	@classmethod
	def read(cls,filename):
		'''
		Read MPIO Header
		'''
		file   = mpi_file_open(MPI_COMM,filename,MPI_RDONLY)
		header = cls.from_file(file)
		file.Close()
		return header

	def __str__(self):
		retstr = ''
		for key in self._header:
			retstr += '%s : '%key + str(self._header[key]) + '\n'
		return retstr

	def __getitem__(self,key):
		if key in ['Format','Version','Object','Dimension','ResultsOn','Type','Size','Sequence','Filter','Sorting','Id']:
			return self.str_to_bin(self._header[key].ljust(7,'0')[:8])
		if key in ['Magic','Columns','Lines','TstepNo','NSubdom','Division','Tag1','Tag2']:
			return self.int_to_bin(self._header[key])
		if key in ['Time']:
			return self.real_to_bin(self._header[key])
		raiseError('Key <%s> not found!'%key)

	def __setitem__(self,key,value):
		if   key in ['Format','Version','Object','Dimension','ResultsOn','Type','Size','Sequence','Filter','Sorting','Id']:
			self._header[key] = self.bin_to_str(value).replace('0','')
		elif key in ['Magic','Columns','Lines','TstepNo','NSubdom','Division','Tag1','Tag2']:
			self._header[key] = self.bin_to_int(value)
		elif key in ['Time']:
			self._header[key] = self.bin_to_real(value)
		else:
			raiseError('Key <%s> not found!'%key)

	def set_dtype(self,dtype):
		'''
		Set header dtype using numpy dtypes.
		'''
		if   dtype == np.int32: 
			self._header['Type'], self._header['Size'] = 'INTEG','4BYTE'
		elif dtype == np.int64: 
			self._header['Type'], self._header['Size'] = 'INTEG','8BYTE'
		elif dtype == np.float32: 
			self._header['Type'], self._header['Size'] = 'REAL','4BYTE'
		elif dtype == np.float64: 
			self._header['Type'], self._header['Size'] = 'REAL','8BYTE'
		else:
			raiseError('Unsupported data type: %s' % dtype)

	def get_dtype(self):
		'''
		Get header dtype using numpy dtypes.
		'''
		if 'INT'  in self._header['Type'] and '4' in self._header['Size']: return np.int32
		if 'INT'  in self._header['Type'] and '8' in self._header['Size']: return np.int64
		if 'REAL' in self._header['Type'] and '4' in self._header['Size']: return np.float32
		if 'REAL' in self._header['Type'] and '8' in self._header['Size']: return np.float64	

	def parse(self,buff):
		'''
		Parse a buffer and generate the header dictionary
		'''
		# Magic number
		magic = self.bin_to_int(buff[:8])
		if not magic == ALYA_MPIO_MAGIC: raiseError('Not a valid Alya MPIO file!')
		# Format
		fmt = self.bin_to_str(buff[8:16])
		if not fmt == ALYA_MPIO_FMT: raiseError('Not a valid Alya MPIO format!')		
		# Version
		vers = self.bin_to_str(buff[16:24])
		if not vers == ALYA_MPIO_VERS: raiseError('Not a valid Alya MPIO version!')	
		# Keep parsin the header
		self['Object']    = buff[24:32]
		self['Dimension'] = buff[32:40]
		self['ResultsOn'] = buff[40:48]
		self['Type']      = buff[48:56]
		self['Size']      = buff[56:64]
		self['Sequence']  = buff[64:72]
		self['Filter']    = buff[72:80]
		self['Sorting']   = buff[80:88]
		self['Id']        = buff[88:96]
		# Some extra checks
		if not 'NOID'  in self._header['Id']:     raiseError('ID column is not supported!')
		if not 'NOFIL' in self._header['Filter']: raiseError('Filtered fields are not supported!')
		# Keep parsin the header
		self['Columns']   = buff[104:112]
		self['Lines']     = buff[112:120]
		self['TstepNo']   = buff[120:128]
		self['NSubdom']   = buff[128:136]
		self['Division']  = buff[136:144]
		self['Tag1']      = buff[144:152]
		self['Tag2']      = buff[152:160]
		self['Time']      = buff[160:168]
		# Some extra checks
		if MPI_SIZE > 1 and ALYA_MPIO_ERROR_NSUBD and not self._header['NSubdom'] == MPI_SIZE-1: raiseError('Incorrect number of subdomains %d, need %d!'%(MPI_SIZE-1,self._header['NSubdom']))
		if MPI_SIZE > 1 and ALYA_MPIO_ERROR_NSUBD and not self._header['NSubdom'] < MPI_SIZE:    raiseWarning('Using more subdomains %d than needed %d!'%(MPI_SIZE-1,self._header['NSubdom']))

	def to_buffer(self):
		'''
		Return the header as a binary array of 256 bytes
		'''
		# Define the buffer
		buff = bytearray(self.nbytes)
		# Write the header
		buff[:8]      = self['Magic']
		buff[8:16]    = self['Format']
		buff[16:24]   = self['Version']

		buff[24:32]   = self['Object']
		buff[32:40]   = self['Dimension']
		buff[40:48]   = self['ResultsOn']
		buff[48:56]   = self['Type']
		buff[56:64]   = self['Size']
		buff[64:72]   = self['Sequence']
		buff[72:80]   = self['Filter']
		buff[80:88]   = self['Sorting']
		buff[88:96]   = self['Id']

		buff[104:112] = self['Columns']
		buff[112:120] = self['Lines']
		buff[120:128] = self['TstepNo']
		buff[128:136] = self['NSubdom']
		buff[136:144] = self['Division']
		buff[144:152] = self['Tag1']
		buff[152:160] = self['Tag2']
		buff[160:168] = self['Time']

		return buff

	@staticmethod
	def str_to_bin(string):
		return ('%s\0'%(string)).encode('utf-8')

	@staticmethod
	def bin_to_str(binary):
		return binary[:-1].decode('utf-8')

	@staticmethod
	def int_to_bin(integer):
		return int(integer).to_bytes(8,'little',signed=True)

	@staticmethod
	def bin_to_int(integer):
		return int.from_bytes(integer,'little',signed=True)

	@staticmethod
	def real_to_bin(real):
		return struct.pack('d',real)

	@staticmethod
	def bin_to_real(real):
		return struct.unpack('d',real)[0]

	@property
	def nbytes(self):
		return self._nbytes

	@property
	def header(self):
		return self._header
	
	@property
	def association(self):
		return self._header['ResultsOn']

	@property
	def nsubd(self):
		return self._header['NSubdom']
	@nsubd.setter
	def nsubd(self,value):
		self._header['NSubdom'] = value

	@property
	def npoints(self):
		return self._header['Lines']
	@npoints.setter
	def npoints(self,value):
		self._header['Lines'] = value

	@property
	def ndims(self):
		return self._header['Columns']
	@ndims.setter
	def ndims(self,value):
		self._header['Columns'] = value

	@property
	def dtype(self):
		return self.get_dtype()
	@dtype.setter
	def dtype(self,value):
		self.set_dtype(value)

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


def readAlyaMPIOHeader(file):
	'''
	Read the header of an Alya MPIO file.
	'''
	return AlyaMPIO_header.from_file(file)


def readPartitions(filename):
	'''
	Read partitions file and return the partition table.	
	'''
	dtypes = [('id','<i8'),('Elements','<i8'),('Points','<i8'),('Boundaries','<i8')]
	return np.genfromtxt(filename,skip_header=1,dtype=dtypes) if MPI_SIZE > 1 else np.array([(0,-1,-1,-1)],dtype=dtypes)

def writePartitions(filename,partition_data):
	'''
	Write partitions table into a file.
	'''
	file = open(filename,'w')
	file.write('%d\n'%partition_data.shape[0])
	for ii in range(partition_data.shape[0]):
		file.write('%10d%10d%10d%10d\n'%(partition_data[ii,0],partition_data[ii,1],partition_data[ii,2],partition_data[ii,3]))
	file.close()

def partitionAlyaMPIO(ncols, dtype, association, nsubd, partition_data, rank=MPI_RANK):
	'''
	Return the rows that must be read and must be skipped on an
	Alya MPIO file according to the partition table.
	'''
	rows2skip, rows2read = -1, -1
	# Treat master in a different way that is compliant with the whole toolbox
	# Also if reading with more ranks than subdomains, skip those that are not needed
	# Skip this treatment if dealing with a serial MPIO
	if not (rank == 0 and MPI_SIZE > 1) and not rank > nsubd:
		# Get the subtable to calculate cumulative sum of the number of rows per partition
		cumsum = {'Elements':0,'Points':0,'Boundaries':0}
		if rank > 1:
			idx = partition_data['id'] < rank
			cumsum['Points']     = np.sum(partition_data['Points'][idx]    ,axis=0)
			cumsum['Elements']   = np.sum(partition_data['Elements'][idx]  ,axis=0)
			cumsum['Boundaries'] = np.sum(partition_data['Boundaries'][idx],axis=0)

		# Obtain this partition
		this_partition = partition_data[partition_data['id']==rank]
		
		# Skip to the block start depending association
		# header['DataType'] is created on line 79 of this file
		#
		# It substitutes  REAL for float and INT for int
		# and substitutes   4  for   32  and  8  for  64
		#
		# so...
		rows2skip, rows2read = 0, 0
		if 'ELEM' in association: rows2skip, rows2read = cumsum['Elements'],   this_partition['Elements'][0]
		if 'POIN' in association: rows2skip, rows2read = cumsum['Points'],     this_partition['Points'][0]
		if 'BOUN' in association: rows2skip, rows2read = cumsum['Boundaries'], this_partition['Boundaries'][0]

	return rows2skip, rows2read


def readAlyaMPIOArray(file, ncols, dtype, rows2skip, rows2read, header_offset, close=True):
	'''
	Read the array of an Alya MPIO file.
	'''
	bytes_per_number = np.dtype(dtype).itemsize
	# Treat master in a different way that is compliant with the whole toolbox
	# Also if reading with more ranks than subdomains, skip those that are not needed
	# Skip this treatment if dealing with a serial MPIO	
	if rows2skip==-1 or rows2read ==-1: 
		aux  = -1 if dtype == np.int32 or dtype == np.int64 else np.nan
		data = aux*np.ones((1,) if ncols == 1 else (1,ncols),dtype=dtype)
	else:                                                                    	
		# Allocate array
		data = np.ndarray((rows2read,ncols) if ncols > 1 else (rows2read,),dtype=dtype)
		# Read data
		file.Read_at(header_offset+rows2skip*ncols*bytes_per_number,data.ravel())
	if close: file.Close()
	return data


def writeAlyaMPIOHeader(file,header):
	'''
	Write the header of an Alya MPIO file.
	'''
	# Write the header in one go
	file.Write(header.to_buffer())

def writeAlyaMPIOArray(file, data, ncols, dtype, nsubd, rows2skip, rows2write, header_offset, rank=MPI_RANK):
	'''
	Write the array to an Alya MPIO file.

	Master does not store data, hence it shouldn't enter this function
	'''
	bytes_per_number = np.dtype(dtype).itemsize

	# Skip the ranks that are higher than the number of subdomain in case
	# we are reading with a higher number of subdomains than expected
	if rank > nsubd: return

	# Write data
	file.Write_at(header_offset+rows2skip*ncols*bytes_per_number,data.ravel())

def AlyaMPIO_readPartitionTable(partitionfile):
	'''
	Read the partition table for Alya MPIO files.
	'''
	table  = np.genfromtxt(partitionfile,skip_header=1,dtype=np.int32) 
	return table if len(table.shape) > 1 else np.expand_dims(table,0)

def AlyaMPIO_writePartitionTable(partitionfile,partition_data):
	'''
	Write the partition table for Alya MPIO files.
	'''
	writePartitions(partitionfile,partition_data)

@cr('AlyaMPIO.read')
def AlyaMPIO_read(filename,partitionfile,force_partition_data=False,rank=MPI_RANK):
	'''
	Read an Alya MPIO file.
	'''
	# Read binary file
	file = mpi_file_open(MPI_COMM,filename,MPI_RDONLY)
	# Read header, only master
	header = None
	if MPI_RANK == 0: header = readAlyaMPIOHeader(file)
	# Master broadcasts header to the other procs
	header = mpi_bcast(header,root=0)
	# Read partition file
	# We don't need the partitions file for a serial MPIO
	# For sequential use always generate the partition_data
	if force_partition_data or not hasattr(AlyaMPIO_read,'partition_data'):
		AlyaMPIO_read.partition_data = readPartitions(partitionfile)
	if MPI_SIZE == 1:
		AlyaMPIO_read.partition_data['Points']     = header.npoints
		AlyaMPIO_read.partition_data['Elements']   = header.npoints
		AlyaMPIO_read.partition_data['Boundaries'] = header.npoints
	rows2skip,rows2read = partitionAlyaMPIO(header.ndims,header.dtype, header.association,
		 header.nsubd, AlyaMPIO_read.partition_data, rank)
	# Read array
	return readAlyaMPIOArray(file,header.ndims,header.dtype, rows2skip, rows2read,header.nbytes), header

@cr('AlyaMPIO.read_serial')
def AlyaMPIO_read_serial(filename,rank=MPI_RANK):
	'''
	Read an Alya MPIO file in serial mode.
	'''
	# Read binary file
	file = mpi_file_open(MPI_COMM,filename,MPI_RDONLY)
	data, header = None, None
	# Read header
	if MPI_RANK == rank: 
		header = readAlyaMPIOHeader(file)
		# Partitions
		rows2skip, rows2read = 0, header.npoints
		# Read array
		data = readAlyaMPIOArray(file,header.ndims,header.dtype, rows2skip, rows2read,header.nbytes,close=False)
	# Close file
	file.Close()
	return data, header

@cr('AlyaMPIO.readByChunk')
def AlyaMPIO_readByChunk(filename,rows2read,rows2skip):
	'''
	Read an Alya MPIO file by chunks defining which
	rows are to be read and which to be skipped.
	'''
	# Read binary file
	file = mpi_file_open(MPI_COMM,filename,MPI_RDONLY)
	# Read header, only master
	header = None
	if MPI_RANK == 0: header = readAlyaMPIOHeader(file)
	# Master broadcasts header to the other procs
	header = mpi_bcast(header,root=0)
	return readAlyaMPIOArray(file,header.ndims,header.dtype,rows2skip,rows2read,header.nbytes), header

@cr('AlyaMPIO.readByChunk_serial')
def AlyaMPIO_readByChunk_serial(filename,rows2read,rows2skip,rank=MPI_RANK):
	'''
	Read an Alya MPIO file by chunks defining which
	rows are to be read and which to be skipped.
	'''
	# Read binary file
	file = mpi_file_open(MPI_COMM,filename,MPI_RDONLY)
	data, header = None, None
	if MPI_RANK == rank: 
		header = readAlyaMPIOHeader(file)
		data   = readAlyaMPIOArray(file,header.ndims,header.dtype,rows2skip,rows2read,header.nbytes,close=False)
	# Close file
	file.Close()
	return data, header

@cr('AlyaMPIO.write')
def AlyaMPIO_write(filename,data,header,partitionfile,force_partition_data=False,rank=MPI_RANK,write=True):
	'''
	Write an Alya MPIO file.
	'''
	# Read partition file
	if force_partition_data or not hasattr(AlyaMPIO_write,'partition_data'):
		AlyaMPIO_write.partition_data = readPartitions(partitionfile)
	if MPI_SIZE == 1:
		AlyaMPIO_write.partition_data['Points']     = header.npoints
		AlyaMPIO_write.partition_data['Elements']   = header.npoints
		AlyaMPIO_write.partition_data['Boundaries'] = header.npoints
	rows2skip,rows2write = partitionAlyaMPIO(header.ndims,header.dtype, header.association,
		 header.nsubd, AlyaMPIO_write.partition_data, rank)
	# Check if we have a correct partition id
	if MPI_RANK < 0 and MPI_SIZE > 1: raiseError("rank %d must be > 0" % MPI_RANK)
	# Write binary file
	file = mpi_file_open(MPI_COMM,filename,MPI_WRONLY|MPI_CREATE)
	# Write header, only master
	if MPI_RANK == 0: writeAlyaMPIOHeader(file,header)
	# Write data
	if (MPI_SIZE == 1 or MPI_RANK > 0) and write:
		writeAlyaMPIOArray(file,data,header.ndims,header.dtype,header.nsubd,rows2skip,rows2write,header.nbytes)
	# Close file
	file.Close()

@cr('AlyaMPIO.write_serial')
def AlyaMPIO_write_serial(filename,data,header,rank=MPI_RANK):
	'''
	Write an Alya MPIO file.
	'''
	rows2skip,rows2write = 0, header.npoints
	# Write binary file
	file = mpi_file_open(MPI_COMM,filename,MPI_WRONLY|MPI_CREATE)
	# Write
	if MPI_RANK == rank: 
		writeAlyaMPIOHeader(file,header)
		writeAlyaMPIOArray(file,data,header.ndims,header.dtype,header.nsubd,rows2skip,rows2write,header.nbytes)
	# Close file
	file.Close()

@cr('AlyaMPIO.writeByChunk')
def AlyaMPIO_writeByChunk(filename,data,header,rows2write,rows2skip):
	'''
	Write an Alya MPIO file.
	'''
	# Write binary file
	file = mpi_file_open(MPI_COMM,filename,MPI_WRONLY|MPI_CREATE)
	# Write header, only master
	if MPI_RANK == 0 and rows2skip == 0: writeAlyaMPIOHeader(file,header)
	# Write data
	if MPI_SIZE == 1 or MPI_RANK > 0:
		writeAlyaMPIOArray(file,data,header.ndims,header.dtype,header.nsubd,rows2skip,rows2write,header.nbytes)
	# Close file
	file.Close()

@cr('AlyaMPIO.writeByChunk_serial')
def AlyaMPIO_writeByChunk_serial(filename,data,header,rows2write,rows2skip,rank=MPI_RANK):
	'''
	Write an Alya MPIO file.
	'''
	# Write binary file
	file = mpi_file_open(MPI_COMM,filename,MPI_WRONLY|MPI_CREATE)
	# Write
	if MPI_RANK == rank: 
		writeAlyaMPIOHeader(file,header)
		writeAlyaMPIOArray(file,data,header.ndims,header.dtype,header.nsubd,rows2skip,rows2write,header.nbytes)
	# Close file
	file.Close()

@cr('AlyaMPIO.writeHeader')
def AlyaMPIO_writeHeader(filename,header):
	'''
	Write an Alya MPIO file.
	'''
	# Write binary file
	file = mpi_file_open(MPI_COMM,filename,MPI_WRONLY|MPI_CREATE)
	# Write header, only master
	if MPI_RANK == 0: writeAlyaMPIOHeader(file,header)
	# Close file
	file.Close()
