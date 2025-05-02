#!/usr/bin/env cpython
#
# pyQvarsi, utils.
#
# Periodic utilities.
#
# Last rev: 22/11/2022
# cython: legacy_implicit_noexcept=True
from __future__ import print_function, division

import os, numpy as np

cimport numpy as np
cimport cython

from libc.stdio      cimport FILE, fopen, fclose, fprintf, fgets, feof, fseek, ftell, SEEK_SET
from libc.stdlib     cimport atof
from libc.string     cimport strcmp, strtok, strlen
from libc.math       cimport round, sqrt
from libcpp.map      cimport map
from cython.operator cimport dereference, postincrement

from ..cr import cr

# Declare the class with cdef
cdef extern from "D3class.h":
	# D3 class
	cdef cppclass D3 "D3":
		D3() except +
		D3(const double x,const double y,const double z) except +
		D3(const double x,const double y,const double z,double gzero) except +
		D3(const D3 &p) except +
		D3(const double*) except +

		inline double norm2() const

		inline D3 operator-(const double &a) const 
		inline D3 operator+(const double &a) const 
		inline D3 operator-(const D3 &a) const 
		inline D3 operator+(const D3 &a) const  
		inline D3 operator^(const D3 &b) const
		inline D3 operator*(const double l) const 
		inline double operator*(const D3 &a) const
		inline D3& operator=(const D3 &a)  
		inline bint operator==(const D3 &) const
		inline bint operator<(const D3 &) const 
		inline double  operator[](int i) const


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
@cython.nogil
cdef double _truncate(double value,int precision):
	'''
	Truncate array by a certain precision
	'''
	cdef int ip
	cdef double fact = 1, out
	for ip in range(precision): fact *= 10.
	return round(value*fact)/fact

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
@cython.nogil
cdef char *_reads(char *line, int size, FILE *fin):
	cdef char *l
	cdef int ii
	return fgets(line,size,fin)


@cr('periodic.get_coord')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def get_coordinates(object geofile,object basedir='./',int precision=6,int ndim=3):
	'''
	Obtain the coordinates from an Alya geo file
	'''
	cdef char buff[1024]
	cdef char ret[512]
	cdef int inod, idim, nnod = 0
	cdef long int pos = 0
	cdef FILE *myfile
	cdef object fname = os.path.join(basedir,geofile)
	cdef np.ndarray[np.double_t,ndim=2] coord
	# Open file for reading
	myfile = fopen(fname.encode('utf-8'),"r")
	# Start reading the file until the COORDINATES
	while not feof(myfile):
		# Read a line
		_reads(buff,sizeof(buff),myfile)
		# Check if we have found the coordinates
		if strcmp(buff,"COORDINATES\n") == 0: break
	# Store the current pointer position
	pos = ftell(myfile)
	# Read until END_COORDINATES to figure out the size of the array
	while not feof(myfile):
		# Read a line
		_reads(buff,sizeof(buff),myfile)
		# Check if we have found the coordinates
		if strcmp(buff,"END_COORDINATES\n") == 0: break
		# Increase number of nodes
		nnod += 1
	# Rewind the pointer to the previous position and
	# allocate the necessary space to read the coordinates
	fseek(myfile,pos,SEEK_SET)
	coord = np.zeros((nnod,3),dtype=np.double)
	# Now read the nodes
	for inod in range(nnod):
		# Read a line
		_reads(buff,sizeof(buff),myfile)
		# Parse positions from line
		ret = strtok(buff," ")
		# Parse coordinates
		for idim in range(ndim):
			# Parse positions from line
			ret = strtok(NULL," ")
			coord[inod,idim] = _truncate(atof(ret),precision)
	fclose(myfile)
	return coord


@cr('periodic.get_bbox')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def get_bounding_box(double[:,:] coord):
	'''
	Obtain the bounding box of a given coordinate set
	'''
	cdef int inod, nnod = coord.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] bbox = np.zeros((6,),np.double)
	# Compute maximum and minimum
	bbox[0] = coord[0,0] # Min X
	bbox[1] = coord[0,0] # Max X
	bbox[2] = coord[0,1] # Min Y
	bbox[3] = coord[0,1] # Max Y
	bbox[4] = coord[0,2] # Min Z
	bbox[5] = coord[0,2] # Max Z
	for inod in range(nnod):
		bbox[0] = min(bbox[0],coord[inod,0]) # Min X 
		bbox[1] = max(bbox[1],coord[inod,0]) # Max X
		bbox[2] = min(bbox[2],coord[inod,1]) # Min Y 
		bbox[3] = max(bbox[3],coord[inod,1]) # Max Y
		bbox[4] = min(bbox[4],coord[inod,2]) # Min Z 
		bbox[5] = max(bbox[5],coord[inod,2]) # Max Z
	return bbox


@cr('periodic.get_per_nodes')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def get_per_nodes(double[:,:] coord,double value1=0.,double value2=0.,int perDim=0,double gzero2=1e-10):
	'''
	Obtain periodic nodes (getPerNodes.cpp)
	'''
	cdef int inod, nnod = coord.shape[0], np1 = 0, np2 = 0, npp = 0
	cdef double goal
	cdef bint crit1, crit2
	cdef D3 point
	cdef map[D3,int] no1, no2
	cdef map[D3,int].iterator it1, it2
	cdef np.ndarray[np.int32_t,ndim=2] perNodes
	# Loop coordinates and build maps
	for inod in range(nnod):
		goal  = coord[inod,perDim]
		point = D3(0.,coord[inod,1],coord[inod,2],gzero2)
		if perDim == 1: point = D3(coord[inod,0],0.,coord[inod,2],gzero2)
		if perDim == 2: point = D3(coord[inod,0],coord[inod,1],0.,gzero2)
		if abs(goal - value1) < gzero2: 
			no1[point] = inod + 1 # Because of python indexing
			np1 += 1
		if abs(goal - value2) < gzero2: 
			no2[point] = inod + 1 # Because of python indexing
			np2 += 1
	# Obtain periodic nodes
	inod = 0
	perNodes = np.zeros((np1,2),np.int32)
	it1 = no1.begin()
	while it1 != no1.end():
		it2 = no2.begin()
		while it2 != no2.end():
			if dereference(it1).first == dereference(it2).first:
				crit1 = abs((dereference(it1).first[perDim] - value1)) < gzero2
				crit2 = abs((dereference(it1).first[perDim] - value2)) < gzero2
				if not crit1 or not crit2:
					perNodes[inod,0] = dereference(it1).second
					perNodes[inod,1] = dereference(it2).second
					inod += 1
					no2.erase(it2)
					break
			postincrement(it2)
		postincrement(it1)
	# Return
	return perNodes[:inod,:] # Crop to number of periodics


@cr('periodic.unique')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def unique_periodic(int[:,:] perNodes):
	'''
	Clean the periodic nodes list and only obtain the
	ones that are unique combinations.
	'''
	cdef int ip, nper = perNodes.shape[0], mast, targ, shihan, nnper = 0
	cdef bint found
	cdef map[int,int] binding
	cdef map[int,int].iterator it, jt, kt
	# Build the binding map
	for ip in range(nper):
		mast  = perNodes[ip,0]
		targ  = perNodes[ip,1]
		it    = binding.begin()
		found = False
		while it != binding.end():
			if targ == dereference(it).first:
				found = True
				break
			postincrement(it)
		if found:
			binding[mast] = targ
		else:
			binding[targ] = mast
	# Clean
	it = binding.begin()
	while it != binding.end():
		# Recover master and target
		mast  = dereference(it).second
		targ  = dereference(it).first
		jt    = binding.begin()
		while jt != binding.end():
			if mast == dereference(jt).first:
				shihan = binding[mast]
				kt     = binding.begin()
				while kt != binding.end():
					if dereference(kt).second == mast:
						dereference(kt).second = shihan
					postincrement(kt)
				break
			postincrement(jt)
		postincrement(it)
		postincrement(nnper) # nper++
	# Write cleaned list
	perNodes = np.zeros((nnper,2),np.int32)
	it = binding.begin()
	ip = 0
	while it != binding.end():
		perNodes[ip,0] = dereference(it).second
		perNodes[ip,1] = dereference(it).first
		postincrement(it)
		postincrement(ip)
	# Return
	return np.array(perNodes,dtype=np.int32)


@cr('periodic.write')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def write_periodic(int[:,:] perNodes,object filename,object basedir='./',bint slave_master=True):
	'''
	Write periodic nodes
	'''
	cdef int ip, nper = perNodes.shape[0]
	cdef object fname = os.path.join(basedir,filename)
	cdef FILE *myfile
	# Open file for writing
	myfile = fopen(fname.encode('utf-8'),"w")
	# Write periodic nodes - slave/master
	if slave_master:
		for ip in range(nper):
			fprintf(myfile,"%d %d\n",perNodes[ip,1],perNodes[ip,0])
	else:
		for ip in range(nper):
			fprintf(myfile,"%d %d\n",perNodes[ip,0],perNodes[ip,1])
	fclose(myfile)