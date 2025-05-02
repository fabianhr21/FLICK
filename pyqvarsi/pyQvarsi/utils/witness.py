#!/usr/bin/env python
#
# pyQvarsi, utils.
#
# ALYA witness file reading routines.
#
# Last rev: 01/04/2022
from __future__ import print_function, division

import numpy as np


def readWitnessHeader(file):
	'''
	Reads a header line by line from an ALYA witness file.
	File needs to be previouly opened.
	'''
	# Variables
	header  = {
		'VARIABLES' : {},
		'NVARS'     : -1,
		'NWIT'      : -1,
		'NLINES'    : 0
	}
	do_skip = True
	# Start reading a file line by line
	for line in file:
		# Increase the number of lines read
		header['NLINES'] += 1
		# Stopping criteria when START is read
		if 'START' in line: break
		# Stop skipping lines up to when HEADER is read
		if not 'HEADER' in line and do_skip: continue
		if 'HEADER' in line:                 do_skip = False; continue 
		# Parse the kind of line in the header
		if 'NUMVARIABLES' in line: header['NVARS'] = int(line.split(':')[1]); continue
		if 'NUMSETS' in line:      header['NWIT']  = int(line.split(':')[1]); continue
		# Else we are heading variables
		linep = line.split()
		header['VARIABLES'][linep[1]] = int(linep[5]) - 1 # to account for python indexing
	# Function return
	return header

def readWitnessInstant(file,nwit):
	'''
	Reads an instant from an ALYA witness file.
	Cursor should be already positioned at the start of the instant
	'''
	# Read iterations
	line = file.readline()
	it   = int(line.split()[3])
	# Read time
	line = file.readline()
	time = float(line.split()[3])
	# Now read the data matrix
	data = np.genfromtxt(file,max_rows=nwit)
	# Function return
	return it, time, data


def witnessReadNByFront(filename,n):
	'''
	Reads N instants starting from the top of the witness file.
	Requires the ALYA witness file to have a header.

	Inputs:
		> filename: full path to the witness file
		> n: number of instants to be read from the front
	
	Outputs:
		> iter: an array containing the number of iterations of the case
		> time: an array containing the time vector
		> data: a dictionary containing the different data from the variables
			data[VARNAME].shape = (n,nwitness)
	'''
	# Open file for reading
	file = open(filename,'r')
	# Read witness file header
	header = readWitnessHeader(file)
	# Preallocate outputs
	iter = np.zeros((n,),dtype=np.double)
	time = np.zeros((n,),dtype=np.double)
	data = {}
	for v  in header['VARIABLES'].keys():
		data[v] = np.zeros((n,header['NWIT']),dtype=np.double)
	# Read a number of instants from the witness file starting 
	# from the front of the file
	for ii in range(n):
		# Read file
		it, t, d = readWitnessInstant(file,header['NWIT'])
		# Set data
		iter[ii] = it
		time[ii] = t
		for v  in header['VARIABLES']:
			data[v][ii,:] = d[:,header['VARIABLES'][v]]
	# Close file
	file.close()
	# Function return
	return iter, time, data


def witnessReadNByBehind(filename,n):
	'''
	Reads N instants starting from the top of the witness file.
	Requires the ALYA witness file to have a header.

	Inputs:
		> filename: full path to the witness file
		> n: number of instants to be read from the front
	
	Outputs:
		> iter: an array containing the number of iterations of the case
		> time: an array containing the time vector
		> data: a dictionary containing the different data from the variables
			data[VARNAME].shape = (n,nwitness)
	'''
	# Open file for reading
	file = open(filename,'r')
	# Read witness file header
	header = readWitnessHeader(file)
	# Preallocate outputs
	iter = np.zeros((n,),dtype=np.double)
	time = np.zeros((n,),dtype=np.double)
	data = {}
	for v  in header['VARIABLES'].keys():
		data[v] = np.zeros((n,header['NWIT']),dtype=np.double)
	# Set the cursor from the bottom of the file up a certain
	# number of lines
	file.seek(0,2)    # seek to end of file; f.seek(0, 2) is legal
	# First we compute how much of an offset an instant is
	# a variable is 17 characters
	# the witness id is 10 characters
	# thus a line is 27 characters - we have nwit lines
	# We need to add the lines on iterations and time
	# Iterations are 37 characters
	# Time are 32 characters
	offset = (10+17*(header['NVARS']-1)+1)*header['NWIT'] + 38 + 33
	# file.seek(0,2) moves the cursor to the end of the file
	# so the following should move the cursor from the bottom to
	# the beginning of the n instants that must be read
	file.seek(file.tell() - n*offset, os.SEEK_SET)   # go backwards 3 bytes
	# Read a number of instants from the witness file starting 
	# from the front of the file
	for ii in range(n):
		# Read file
		it, t, d = readWitnessInstant(file,header['NWIT'])
		# Set data
		iter[ii] = it
		time[ii] = t
		for v  in header['VARIABLES']:
			data[v][ii,:] = d[:,header['VARIABLES'][v]]
	# Close file
	file.close()
	# Function return
	return iter, time, data