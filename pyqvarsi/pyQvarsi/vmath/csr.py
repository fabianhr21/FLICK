#!/usr/bin/env python
#
# pyQvarsi, CSR matrix.
#
# Module to compute mathematical operations between
# CSR matrix formats.
#
# CSR operations.
#
# Last rev: 07/07/2022
from __future__ import print_function, division

import numpy as np, scipy


def dok_create(m,n,dtype=np.double):
	'''
	Return an empty DOK useful for initializing 
	sparse arrays
	'''
	return scipy.sparse.dok_matrix((m,n),dtype=dtype)

def csr_create(m,n,dtype=np.double):
	'''
	Return CSR matrix of size (m,n) and a certain dtype
	'''
	return scipy.sparse.csr_matrix((m,n),dtype=dtype)

def csr_tocsr(A):
	'''
	Create a CSR array from a given format
	'''
	return A.tocsr()

def csr_unpack(A):
	'''
	Returns the CSR arrays from a CSR matrix.
	The number of nonzero elements is the length of
	the C array.
	'''
	return A.data, A.index, A.indptr

def csr_pack(C,S,R,shape,dtype):
	'''
	Return a CSR matrix in function of the C,S,R and
	the shape of the matrix.
	'''
	return scipy.sparse.csr_matrix((C, S, R), shape=shape, dtype=dtype)

def csr_convert(A):
	'''
	Convert matrix A to CSR format
	'''
	return scipy.sparse.csr_matrix(A,dtype=A.dtype)

def csr_toarray(A):
	'''
	Convert CSR matrix A to array
	'''
	return np.array(A.toarray())

def csr_identity(A):
	'''
	Identity CSR matrix of the same size of A.
	'''
	return scipy.sparse.identity(A.shape[0],dtype=A.dtype,format='csr')

def csr_transpose(A):
	'''
	Transposes the CSR matrix A into A^t.
	'''
	return A.transpose()

def csr_trace(A):
	'''
	Computes the trace of a CSR matrix A
	'''
	return A.trace()

def csr_diagonal(A):
	'''
	Returns the diagonal of a CSR matrix A
	'''
	return A.diagonal()

def csr_spmv(A,v):
	'''
	CSR sparse matrix vector product. Returns an array.
	'''
	return A.dot(v)

def csr_spmm(A,B):
	'''
	CSR sparse matrix matrix product. Returns a matrix in CSR format.
	'''
	return A.dot(B)