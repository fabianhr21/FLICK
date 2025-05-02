import numpy as np

cimport numpy as np
cimport cython
from libc.math cimport pow, fabs, cos, pi

from ..cr import cr
from ..utils.common import raiseError

cdef inline long comb(long n, long k):
	cdef long r = 1, d = n - k
	if k < 0 or k > n:
		return 0
	if d < k:
		k = d
	for d in range(1, k+1):
		r *= n
		r //= d
		n -= 1
	return r

def halley(double x, fun, dfun, d2fun, double tol, int niter):
	cdef double y = fun(x)
	cdef double yp, ypp, xn
	cdef int ite

	if fabs(y) < tol:
		return x, fabs(y)

	for ite in range(niter):
		yp = dfun(x)
		ypp = d2fun(x)
		xn = x - 2.0 * y * yp / (2.0 * yp * yp - y * ypp)
		y = fun(xn)
		x = xn
		if fabs(y) < tol:
			break
	return x, fabs(y)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef double lagrange(double r, int kpoint, double[:] isopoints):
	cdef double prod = 1.0
	cdef int ipoint
	cdef double isopoint
	for ipoint in range(isopoints.shape[0]):
		if ipoint != kpoint:
			isopoint = isopoints[ipoint]
			prod *= (r - isopoint) / (isopoints[kpoint] - isopoint)

	return prod

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef double dlagrange(double r, int kpoint, double[:] isopoints):
	cdef double sum = 0.0
	cdef int ipoint, jpoint
	cdef double prod, isopoint, isopoint2
	for ipoint in range(isopoints.shape[0]):
		isopoint = isopoints[ipoint]
		if ipoint != kpoint:
			prod = 1.0
			for jpoint in range(isopoints.shape[0]):
				isopoint2 = isopoints[jpoint]
				if jpoint != kpoint and jpoint != ipoint:
					prod *= (r - isopoint2) / (isopoints[kpoint] - isopoint2)
			sum += prod / (isopoints[kpoint] - isopoint)

	return sum

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def legendre(int p, double x):
	cdef int k
	cdef double lp = 0.0
	for k in range(p+1):
		lp += pow(0.5*(x-1.0), k) * comb(p, k) * comb(p+k, k)
	return lp

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def dlegendre(int p, double x):
	# Avoid zero singularity
	if fabs(x - 1.0) < np.finfo(x).eps:
		x += np.finfo(x).eps
	cdef int k
	cdef double lp = 0.0, term
	for k in range(1, p+1):  # start from 1 since k=0 term is always 0
		term = 0.5 * k * pow(0.5 * (x - 1.0), k - 1)
		lp += comb(p, k) * comb(p + k, k) * term
	return lp

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def d2legendre(int p, double x):
	# Avoid zero singularity
	if fabs(x - 1.0) < np.finfo(x).eps:
		x += np.finfo(x).eps
	cdef int k
	cdef double lp = 0.0, term
	for k in range(2, p+1):  # start from 2 since k=0 and k=1 terms are always 0
		term = 0.25 * k * (k - 1) * pow(0.5 * (x - 1.0), k - 2)
		lp += comb(p, k) * comb(p + k, k) * term
	return lp

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def d3legendre(int p, double x):
	# Avoid zero singularity
	if fabs(x - 1.0) < np.finfo(x).eps:
		x += np.finfo(x).eps
	cdef int k
	cdef double lp = 0.0, term
	for k in range(3, p+1):  # start from 3 since k=0, k=1, and k=2 terms are always 0 or undefined
		term = 0.125 * k * (k - 1) * (k - 2) * pow(0.5 * (x - 1.0), k - 3)
		lp += comb(p, k) * comb(p + k, k) * term
	return lp

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def quadrature_GaussLobatto(int p, double tol=1e-16, niter=5):
	cdef int k, i
	cdef np.ndarray[np.double_t,ndim=1] xi = np.zeros(p, dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] wi = np.zeros(p, dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] ts = np.zeros(p, dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] xi_reorder = np.zeros(p, dtype=np.double)
	cdef np.ndarray[np.double_t,ndim=1] wi_reorder = np.zeros(p, dtype=np.double)
	cdef double Pxi
	cdef double eps = 1e-12

	for k in range(1, p-1):
		xi[k] = -(1.0 - (3.0 * (p - 2.0)) / (8.0 * (p - 1.0)**3)) * cos((4.0 * (k+1) - 3.0) * pi / (4.0 * (p - 1.0) + 1.0))
		xi[k], ts[k] = halley(xi[k], lambda x: dlegendre(p-1, x), lambda x: d2legendre(p-1, x), lambda x: d3legendre(p-1, x), tol, niter)
	for k in range(1,p-1):
		abxk = np.abs(xi[k])
		for i in range(1,p-1):
			if i != k:
				abxi = np.abs(xi[i])
				if np.round(abxi,10) == np.round(abxk,10):
					if ts[k] < ts[i]:
						xi[k] = -xi[i]
	for k in range(1,p-1): 
		Pxi = legendre(p - 1, xi[k])
		wi[k] = 2.0 / (p * (p - 1.0) * Pxi**2)
	# Set extremes
	xi[0] = -1.0
	wi[0] = 2.0 / (p * (p - 1.0))
	xi[p-1] = 1.0
	wi[p-1] = 2.0 / (p * (p - 1.0))
	# Round zeros and reorder
	for i in range(p):
		if fabs(xi[i]) < eps:
			xi[i] = 0.0
	xi_reorder[0] = xi[0]
	xi_reorder[1] = xi[p-1]
	wi_reorder[0] = wi[0]
	wi_reorder[1] = wi[p-1]
	for i in range(1, p-1):
		xi_reorder[i+1] = xi[i]
		wi_reorder[i+1] = wi[i]

	return xi_reorder, wi_reorder