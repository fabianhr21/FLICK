#!/usr/bin/env cython
#
# pyQvarsi, FEM lib.
#
# Small FEM module to compute derivatives and possible other
# simple stuff from Alya output for postprocessing purposes.
#
# Library of FEM elements according to Alya.
#
# Last rev: 06/06/2023
from __future__ import print_function, division
from itertools import product

import numpy as np

cimport numpy as np
cimport cython
from libc.math cimport sqrt, pow, floor, abs
from ..cr import cr


from ..utils.common import raiseError
from .quadratures import quadrature_GaussLobatto
from .quadratures cimport lagrange, dlagrange

cdef class Element1D:
	'''
	Basic FEM parent class that implements the
	basic operations for a 1D element.
	'''     
	cdef int  _ngp, _nnod

	cdef int[:] _nodeList

	cdef double[:]     _weigp
	cdef double[:,:]   _posnod, _posgp, _shapef
	cdef double[:,:,:] _gradi

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	def __init__(Element1D self, int[:] nodeList, int ngauss):
		'''
		Define an element given a list of nodes and 
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:                  number of Gauss points
		'''
		cdef int ii

		self._ngp      = ngauss
		self._nnod     = len(nodeList)
		self._nodeList = np.zeros((self.nnod,),dtype=np.int32)
		
		for ii in range(self._nnod):
			self._nodeList[ii] = nodeList[ii]

		self._posnod   = np.zeros((self.nnod,1),dtype=np.double)             # Nodes position in local coordinates
		self._posgp    = np.zeros((self.ngauss,1),dtype=np.double)           # Gauss points position
		self._weigp    = np.zeros((self.ngauss,),dtype=np.double)            # Gauss points weights
		self._shapef   = np.zeros((self.nnod,self.ngauss),dtype=np.double)   # Shape function
		self._gradi    = np.zeros((1,self.nnod,self.ngauss),dtype=np.double) # Gradient function (in local coordinates)
		
	def __str__(Element1D self):
		s = 'Element 1D nnod=%d\n' % self._nnod
		return s

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	def __len__(Element1D self):
		return self._nnod

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	def __eq__(Element1D self, Element1D other):
		'''
		Element1 == Element2
		'''
		if other is None:
			return type(self) == None
		else:
			return self._ngp == other._ngp and self._nnod == other._nnod

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def centroid(Element1D self,double[:,:] xyel):
		'''
		Element centroid
		'''
		cdef int inod, nnod = self._nnod
		cdef np.ndarray[np.double_t,ndim=1] cen = np.zeros((2,),dtype=np.double)
		# Compute center
		for inod in range(nnod):
			cen[0] += xyel[inod,0]
			cen[1] += xyel[inod,1]
		cen[0] /= nnod
		cen[1] /= nnod
		return cen

	@cython.boundscheck(False) # turn off bounds-checking for entizre function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def normal(Element1D self,double[:,:] xyel):
		'''
		Element normal, expects 2D coordinates.
		'''
		cdef int idim, inod, nnod = self._nnod
		cdef double[:] cen
		cdef np.ndarray[np.double_t,ndim=1] u   = np.zeros((3,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] v   = np.zeros((3,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] nor = np.zeros((2,),dtype=np.double)
		# Compute center
		cen = self.centroid(xyel)
		v[2] = 1.
		# Compute normal
		# Compute u, v
		for idim in range(3):
			u[idim] =  xyel[0,idim] - cen[idim]
		# Cross product
		nor[0] -= 0.5*(u[1]*v[2] - u[2]*v[1])
		nor[1] -= 0.5*(u[2]*v[0] - u[0]*v[2])
		for inod in range(1,nnod):
			# Compute u, v
			for idim in range(3):
				u[idim] =  xyel[inod,idim]       - cen[idim]
			# Cross product
			nor[0] -= 0.5*(u[1]*v[2] - u[2]*v[1])
			nor[1] -= 0.5*(u[2]*v[0] - u[0]*v[2])
		return nor

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def transform1D(Element1D self,double[:,:] xy, double[:,:] xyel): 
		'''
		Transforms the coordinate system of a surface element in a 1D mesh 
		to one with the following properties:
			- Origin: 1st node of the connectivity.
			- S axis: aligned with the edge that goes from the 1st to the 2nd node.
			- T axis: orthogonal and coplanar to the S axis. It points to the side where the 3rd node is located.
			- R axis: orthogonal to the plane defined by the triangle.
			- The determinant of the Jacobian of this transformation is always 1 (det(J) = 1)
		IN:
			> xy(nnod,2):   position of the points to transform
			> xyel(nnod,2): position of the nodes in cartesian coordinates
		OUT:
			> xel(nnod,): position of the nodes in cartesian coordinates
		'''
		cdef int idim, idim1, nnod = xy.shape[0] if xy.ndim > 1 else 0
		cdef double norm_r, norm_s
		cdef np.ndarray x

		cdef np.ndarray[np.double_t,ndim=1] r = np.zeros((2,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] s = np.zeros((2,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] M = np.zeros((2,2),dtype=np.double)

		# Compute RST axis
		r    = self.normal(xyel)
		s[0] = xyel[1,0] - xyel[0,0]
		s[1] = xyel[1,1] - xyel[0,1]
		# Normalize
		norm_r = sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])
		norm_s = sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2])
		# Coordinate change matrix
		for idim in range(2):
			M[0,idim] = s[idim]/norm_s
			M[1,idim] = r[idim]/norm_r
		# Project
		if nnod > 0:
			x = np.zeros((nnod,),dtype=np.double)
			for inod in range(nnod):
				for idim1 in range(2):
					x[inod] += M[0,idim1]*xy[inod,idim1]
		else:
			x = 0.
			for idim1 in range(3):
				x += M[0,idim1]*xy[0,idim1]
		return x

	@cython.boundscheck(False) # turn off bounds-checking for entizre function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def nodes2gp(Element1D self,np.ndarray elfield):
		'''
		Transforms an entry field on the nodes
		(or their position) to the Gauss point 
		equivalent.
		'''
		cdef int         igauss, inod, idim, nnod = self._nnod, ngp = self._ngp
		cdef np.ndarray  field_gp
		cdef double[:,:] shapef = self._shapef
		# Compute dot
		if elfield.ndim > 1:
			ndim = elfield.shape[1] 
			field_gp = np.zeros((ngp,ndim),dtype=np.double)      
			
			for igauss in range(ngp):
				for inod in range(nnod):
					for idim in range(ndim):
						field_gp[igauss,idim] += shapef[inod,igauss]*elfield[inod,idim]
		else:
			ndim = 0
			field_gp = np.zeros((ngp,),dtype=np.double)      
			
			for igauss in range(ngp):
				for inod in range(nnod):
						field_gp[igauss] += shapef[inod,igauss]*elfield[inod]
		return field_gp

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def derivative(Element1D self,double[:,:] xyel):
		'''
		Derivative of the element:
		IN:
			> xyel(nnod,2):            position of the nodes in cartesian coordinates
		OUT:
			> deri(1,nnod,ngauss):  derivatives per each gauss point
			> vol(ngauss):                  volume per each gauss point
			> mle(nnod):                    lumped mass matrix per each node
		'''
		cdef int           igp, inod
		cdef double        J, Jinv
		cdef double[:]     xel, weigp  = self._weigp
		cdef double[:,:,:] gradi  = self._gradi

		cdef np.ndarray[np.double_t,ndim=1] vol  = np.zeros((self.ngauss,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=3] deri = np.zeros((1,self.nnod,self.ngauss),dtype=np.double)

		# Ensure dealing with a 2D array
		xel = xyel if xyel.shape[1] == 1 else self.transform1D(xyel,xyel)
		for igp in range(self._ngp):
			J = 0.
			# Compute Jacobian
			for inod in range(self._nnod):
				J += xel[inod] * gradi[0,inod,igp]
			# Determinant and inverse of Jacobian
			Jinv = 1./J
			# Element derivatives & lumped mass
			for inod in range(self._nnod):
				deri[0,inod,igp] = Jinv*gradi[0,inod,igp]
			# Element volume
			vol[igp] = J*weigp[igp]

		return deri, vol

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def integrative(Element1D self, double[:,:] xyel):
		'''
		Integral of the element:
		IN:
			> xyel(nnod,2):          position of the nodes in cartesian coordinates
		OUT:
			> integ(nnod,ngauss): integral per each gauss point
		'''
		cdef int           igp, inod
		cdef double        J
		cdef double[:]     weigp  = self._weigp, xel
		cdef double[:,:]   shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi

		cdef np.ndarray[np.double_t,ndim=2] integ = np.zeros((self.nnod, self.ngauss),dtype=np.double)

		# Ensure dealing with a 2D array
		xel = xyel if xyel.shape[1] == 1 else self.transform1D(xyel,xyel)
		# Integral computation
		for igp in range(self._ngp):
			J = 0.
			# Compute Jacobian
			for inod in range(self._nnod):
				J += xel[inod] * gradi[0,inod,igp]
			# Element integral
			for inod in range(self._nnod):
				integ[inod,igp] =  weigp[igp]*shapef[inod,igp]*J

		return integ

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def consistent(Element1D self, double[:,:] xyel):
		'''
		Consistent mass matrix of the element:
		IN:
			> xyel(nnod,2):   position of the nodes in cartesian coordinates
		OUT:
			> mle(nnod,nnod): consistent mass matrix over the Gauss points	
		'''
		cdef int           igp, inod, jnod
		cdef double        J
		cdef double[:]     weigp  = self._weigp, xel
		cdef double[:,:]   shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi

		cdef np.ndarray[np.double_t,ndim=2] mle = np.zeros((self._nnod,self._nnod),np.double)
		
		# Ensure dealing with a 2D array
		xel = xyel if xyel.shape[1] == 1 else self.transform1D(xyel,xyel)
		# Computation
		for igp in range(self._ngp):
			J = 0.
			# Compute Jacobian
			for inod in range(self._nnod):
				J += xel[inod] * gradi[0,inod,igp]
			# Element mass matrix
			for inod in range(self._nnod):
				for jnod in range(self._nnod):
					mle[inod,jnod] += weigp[igp]*J*shapef[inod,igp]*shapef[jnod,igp]

		return mle

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def find_stz(Element1D self, double[:] xy, double[:,:] xyel, int max_iter=20, double tol=1e-10):
		'''
		Find a position of the point in xyz coordinates
		in element local coordinates stz:
		IN:
			> xy(1,2):              position of the point
			> xyel(nnod,2): position of the element nodes
		OUT:
			> stz(1,):               position of the point in local coordinates
		'''
		cdef int it, inod
		cdef double r, stz, delta, x, aux, T, Tinv, f
		cdef double[:]  shapef = self._shapef[:,0]
		cdef double[:,:] gradi = self._gradi[:,:,0]

		cdef np.ndarray[np.double_t,ndim=1] xel  = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] xy1  = np.zeros((1,2),dtype=np.double)
		stz = self._posgp[0,0]

		# Ensure dealing with a 2D array
		xy1[0,0] = xy[0]
		xy1[0,1] = xy[1]
		x   = xy   if xyel.shape[1] == 1 else self.transform1D(xy1,xyel)
		xel = xyel if xyel.shape[1] == 1 else self.transform1D(xyel,xyel)

		# Compute residual
		aux = 0.
		for inod in range(self._nnod):
			aux += shapef[inod]*xel[inod]
		f = x - aux
		r = f

		# Newton-Raphson method
		for it in range(max_iter):
			T = 0.
			# Compute T
			for inod in range(self._nnod):
				T -= gradi[0,inod] * xel[inod]
			# Inverse of T
			Tinv = 1./T
			# New guess
			delta = -(Tinv*f)
			stz  += delta
			# New shape functions and gradients
			shapef, gradi = self.shape_func(stz)
			# Compute new residual
			aux = 0.
			for inod in range(self._nnod):
				aux += shapef[inod]*xel[inod]
			f = x - aux
			r = f
			# Exit criteria
			if r < tol: break
		return stz

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def interpolate(Element1D self, double[:] stz, np.ndarray elfield):
		'''
		Interpolate a variable on a point inside the element:
		IN:
			> stz(1):                         position of the point in local coordinates
			> elfield(nnod,ndim): variable to be interpolated
		OUT:
			> out(1,ndim):            interpolated variable at stz
		'''
		cdef int ii, inod
		cdef int n = elfield.shape[0], m = elfield.shape[1] if elfield.ndim > 1 else 0
		cdef double[:]  shapef
		cdef np.ndarray out

		# Recover the shape function at the point stz
		shapef,_ = self.shape_func(stz)

		# Compute output array
		if m > 0:
			# Allocate output array
			out = np.zeros((1,m),dtype=np.double)
			# Compute dot           
			for inod in range(self._nnod):
				for ii in range(m):
					out[0,ii] += shapef[inod]*elfield[inod,ii]
		else:
			# Allocate output array
			out = np.zeros((1,),dtype=np.double)
			# Compute dot           
			for inod in range(self._nnod):
				out[0] += shapef[inod]*elfield[inod]
		# Return output
		return out

	@property
	def nnod(Element1D self):
		return self._nnod
	
	@property
	def ngauss(Element1D self):
		return self._ngp

	@property
	def ndim(Element1D self):
		return 1

	@property
	def nodes(Element1D self):
		return self._nodeList

	@property
	def posnod(Element1D self):
		return self._posnod

	@property
	def posgp(Element1D self):
		return self._posgp

	@property
	def shape(Element1D self):
		return self._shapef
	
	@property
	def grad(Element1D self):
		return self._gradi


cdef class Element2D:
	'''
	Basic FEM parent class that implements the
	basic operations for a 2D element.
	'''     
	cdef int  _ngp, _nnod

	cdef int[:] _nodeList

	cdef double[:]     _weigp
	cdef double[:,:]   _posnod, _posgp, _shapef, _integ
	cdef double[:,:,:] _gradi

	cdef bint _sem

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	def __init__(Element2D self, int[:] nodeList, int ngauss, bint SEM=False):
		'''
		Define an element given a list of nodes and 
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:                  number of Gauss points
		'''
		cdef int ii

		self._ngp      = ngauss
		self._nnod     = len(nodeList)
		self._nodeList = np.zeros((self.nnod,),dtype=np.int32)
		self._sem      = SEM

		for ii in range(self._nnod):
			self._nodeList[ii] = nodeList[ii]

		self._posnod   = np.zeros((self.nnod,2),dtype=np.double)   			 # Nodes position in local coordinates
		if not self._sem:
			self._posgp = np.zeros((self.ngauss,2),dtype=np.double)			 # Gauss points position
			self._weigp    = np.zeros((self.ngauss,),dtype=np.double)            # Gauss points weights
			self._shapef   = np.zeros((self.nnod,self.ngauss),dtype=np.double)   # Shape function
			self._gradi    = np.zeros((2,self.nnod,self.ngauss),dtype=np.double) # Gradient function (in local coordinates)

	def __str__(Element2D self):
		s = 'Element 2D nnod=%d\n' % self._nnod
		return s

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	def __len__(Element2D self):
		return self._nnod

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	def __eq__(Element2D self, Element2D other):
		'''
		Element1 == Element2
		'''
		if other is None:
			return type(self) == None
		else:
			return self._ngp == other._ngp and self._nnod == other._nnod

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def centroid(Element2D self,double[:,:] xyzel):
		'''
		Element centroid
		'''
		cdef int inod, nnod = self._nnod
		cdef np.ndarray[np.double_t,ndim=1] cen = np.zeros((3,),dtype=np.double)
		# Compute center
		for inod in range(nnod):
			cen[0] += xyzel[inod,0]
			cen[1] += xyzel[inod,1]
			cen[2] += xyzel[inod,2]
		cen[0] /= nnod
		cen[1] /= nnod
		cen[2] /= nnod
		return cen

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def normal(Element2D self,double[:,:] xyzel):
		'''
		Normal in the nodes of the element, expects 3D coordinates.
		TODO: Talk a lot with Arnau and Lucas!
		'''
		cdef int igp, inod
		cdef double[3] gradi_xi, gradi_et, J
		cdef double detJ, val
		cdef np.ndarray[np.double_t,ndim=2] normal = np.zeros((self.nnod,3),dtype=np.double)
		cdef double[:,:,:] gradi  = self._gradi
		for igp in range(self._ngp):
			# Reset vectors for each gauss point
			gradi_xi[0] = 0.
			gradi_xi[1] = 0.
			gradi_xi[2] = 0.
			gradi_et[0] = 0.
			gradi_et[1] = 0.
			gradi_et[2] = 0.
			# Compute Jacobian contributions
			for inod in range(self.nnod):
				gradi_xi[0] += gradi[0, inod, igp] * xyzel[inod, 0]
				gradi_xi[1] += gradi[0, inod, igp] * xyzel[inod, 1]
				gradi_xi[2] += gradi[0, inod, igp] * xyzel[inod, 2] if xyzel.shape[1] == 3 else 0
				gradi_et[0] += gradi[1, inod, igp] * xyzel[inod, 0]
				gradi_et[1] += gradi[1, inod, igp] * xyzel[inod, 1]
				gradi_et[2] += gradi[1, inod, igp] * xyzel[inod, 2] if xyzel.shape[1] == 3 else 0
			# Calculate cross product manually
			J[0] = gradi_xi[1] * gradi_et[2] - gradi_xi[2] * gradi_et[1]
			J[1] = gradi_xi[2] * gradi_et[0] - gradi_xi[0] * gradi_et[2]
			J[2] = gradi_xi[0] * gradi_et[1] - gradi_xi[1] * gradi_et[0]
			detJ = sqrt(J[0]**2 + J[1]**2 + J[2]**2)
			normal[igp,0] = J[0]/detJ
			normal[igp,1] = J[1]/detJ
			normal[igp,2] = J[2]/detJ
		return normal

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def transform2D(Element2D self,double[:,:] xyz, double[:,:] xyzel): 
		'''
		Transforms the coordinate system of a surface element in a 3D mesh 
		to one with the following properties:
			- Origin: 1st node of the connectivity.
			- S axis: aligned with the edge that goes from the 1st to the 2nd node.
			- T axis: orthogonal and coplanar to the S axis. It points to the side where the 3rd node is located.
			- R axis: orthogonal to the plane defined by the triangle.
			- The determinant of the Jacobian of this transformation is always 1 (det(J) = 1)
		IN:
			> xyz(nnod,3):   position of the points to transform
			> xyzel(nnod,3): position of the nodes in cartesian coordinates
		OUT:
			> xyzel(nnod,2): position of the nodes in cartesian coordinates
		'''
		cdef int idim, idim1, nnod = xyz.shape[0] if xyz.ndim > 1 else 0
		cdef double norm_r, norm_s, norm_t
		cdef np.ndarray xy

		cdef np.ndarray[np.double_t,ndim=2] r = np.zeros((self.nnod,3),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] s = np.zeros((3,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] t = np.zeros((3,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] M = np.zeros((3,3),dtype=np.double)

		# Compute RST axis
		r    = self.normal(xyzel)
		s[0] = xyzel[1,0] - xyzel[0,0]
		s[1] = xyzel[1,1] - xyzel[0,1]
		s[2] = xyzel[1,2] - xyzel[0,2]
		t[0] = r[0,1]*s[2] - r[0,2]*s[1]
		t[1] = r[0,2]*s[0] - r[0,0]*s[2]
		t[2] = r[0,0]*s[1] - r[0,1]*s[0]
		# Normalize
		norm_r = sqrt(r[0,0]*r[0,0]+r[0,1]*r[0,1]+r[0,2]*r[0,2])
		norm_s = sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2])
		norm_t = sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2])
		# Coordinate change matrix
		for idim in range(3):
			M[0,idim] = s[idim]/norm_s
			M[1,idim] = t[idim]/norm_t
			M[2,idim] = r[0,idim]/norm_r
		# Project
		if nnod > 0:
			xy = np.zeros((nnod,2),dtype=np.double)
			for inod in range(nnod):
				for idim in range(2):
					for idim1 in range(3):
						xy[inod,idim] += M[idim,idim1]*xyz[inod,idim1]
		else:
			xy = np.zeros((2,),dtype=np.double)
			for idim in range(2):
				for idim1 in range(3):
					xy[idim] += M[idim,idim1]*xyz[0,idim1]
		return xy

	@cython.boundscheck(False) # turn off bounds-checking for entizre function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def nodes2gp(Element2D self,np.ndarray elfield):
		'''
		Transforms an entry field on the nodes
		(or their position) to the Gauss point 
		equivalent.
		'''
		cdef int         igauss, inod, idim, nnod = self._nnod, ngp = self._ngp
		cdef np.ndarray  field_gp
		cdef double[:,:] shapef = self._shapef
		# Compute dot
		if elfield.ndim > 1:
			ndim = elfield.shape[1] 
			field_gp = np.zeros((ngp,ndim),dtype=np.double)      
			
			for igauss in range(ngp):
				for inod in range(nnod):
					for idim in range(ndim):
						field_gp[igauss,idim] += shapef[inod,igauss]*elfield[inod,idim]
		else:
			ndim = 0
			field_gp = np.zeros((ngp,),dtype=np.double)      
			
			for igauss in range(ngp):
				for inod in range(nnod):
						field_gp[igauss] += shapef[inod,igauss]*elfield[inod]
		return field_gp

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def derivative(Element2D self,double[:,:] xyzel):
		'''
		Derivative of the element:
		IN:
			> xyzel(nnod,2):        position of the nodes in cartesian coordinates
		OUT:
			> deri(2,nnod,ngauss):  derivatives per each gauss point
			> vol(ngauss):          volume per each gauss point
			> mle(nnod):            lumped mass matrix per each node
		'''
		cdef int        igp, inod
		cdef double detJ
		cdef double[:]     weigp  = self._weigp
		cdef double[:,:]   xyel#, shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi

		cdef np.ndarray[np.double_t,ndim=2] J    = np.zeros((2,2),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] Jinv = np.zeros((2,2),dtype=np.double)

		cdef np.ndarray[np.double_t,ndim=1] vol  = np.zeros((self.ngauss,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=3] deri = np.zeros((2,self.nnod,self.ngauss),dtype=np.double)

		# Ensure dealing with a 2D array
		#xyel = xyzel if xyzel.shape[1] == 2 else self.transform2D(xyzel,xyzel)
		for igp in range(self._ngp):
			J[0,0] = 0.
			J[0,1] = 0.
			J[1,0] = 0.
			J[1,1] = 0.
			# Compute Jacobian
			for inod in range(self._nnod):
				J[0,0] += xyzel[inod,0] * gradi[0,inod,igp]
				J[0,1] += xyzel[inod,0] * gradi[1,inod,igp]
				J[1,0] += xyzel[inod,1] * gradi[0,inod,igp]
				J[1,1] += xyzel[inod,1] * gradi[1,inod,igp]
			# Determinant and inverse of Jacobian
			detJ = J[0,0]*J[1,1] - J[1,0]*J[0,1]
			Jinv[0,0] =  J[1,1]/detJ
			Jinv[0,1] = -J[0,1]/detJ
			Jinv[1,0] = -J[1,0]/detJ
			Jinv[1,1] =  J[0,0]/detJ
			# Element derivatives & lumped mass
			for inod in range(self._nnod):
				deri[0,inod,igp] = Jinv[0,0]*gradi[0,inod,igp] + Jinv[1,0]*gradi[1,inod,igp]
				deri[1,inod,igp] = Jinv[0,1]*gradi[0,inod,igp] + Jinv[1,1]*gradi[1,inod,igp]
			# Element volume
			vol[igp] = detJ*weigp[igp]

		return deri, vol

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def integrative(Element2D self, double[:,:] xyzel):
		'''
		Integral of the element:
		IN:
			> xyzel(nnod,3):      position of the nodes in cartesian coordinates
		OUT:
			> integ(nnod,ngauss): integral per each gauss point
		'''
		cdef int igp, inod
		cdef double[3] gradi_xi, gradi_et, J
		cdef double detJ, val
		cdef np.ndarray[np.double_t,ndim=2] integ = np.zeros((self.nnod,self._ngp),dtype=np.double)
		cdef double[:]     weigp  = self._weigp
		cdef double[:,:,:] gradi  = self._gradi
		cdef double[:,:]   shapef = self._shapef
		for igp in range(self._ngp):
			# Reset vectors for each gauss point
			gradi_xi[0] = 0.
			gradi_xi[1] = 0.
			gradi_xi[2] = 0.
			gradi_et[0] = 0.
			gradi_et[1] = 0.
			gradi_et[2] = 0.
			# Compute Jacobian contributions
			for inod in range(self.nnod):
				gradi_xi[0] += gradi[0, inod, igp] * xyzel[inod, 0]
				gradi_xi[1] += gradi[0, inod, igp] * xyzel[inod, 1]
				gradi_xi[2] += gradi[0, inod, igp] * xyzel[inod, 2] if xyzel.shape[1] == 3 else 0
				gradi_et[0] += gradi[1, inod, igp] * xyzel[inod, 0]
				gradi_et[1] += gradi[1, inod, igp] * xyzel[inod, 1]
				gradi_et[2] += gradi[1, inod, igp] * xyzel[inod, 2] if xyzel.shape[1] == 3 else 0
			# Calculate cross product manually
			J[0] = gradi_xi[1] * gradi_et[2] - gradi_xi[2] * gradi_et[1]
			J[1] = gradi_xi[2] * gradi_et[0] - gradi_xi[0] * gradi_et[2]
			J[2] = gradi_xi[0] * gradi_et[1] - gradi_xi[1] * gradi_et[0]
			# Calculate norm (determinant) manually
			detJ = sqrt(J[0]**2 + J[1]**2 + J[2]**2)
			# Update integration values
			for inod in range(self._nnod):  # Assuming _integ's first dimension size
				integ[inod,igp] = weigp[igp]*shapef[inod,igp]*detJ

		return integ

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def consistent(Element2D self, double[:,:] xyzel):
		'''
		Consistent mass matrix of the element:
		IN:
			> xyzel(nnod,2):  position of the nodes in cartesian coordinates
		OUT:
			> mle(nnod,nnod): consistent mass matrix over the Gauss points	
		'''
		cdef int           igp, inod, jnod
		cdef double        detJ
		cdef double[:]     weigp  = self._weigp
		cdef double[:,:]   xyel, shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi

		cdef np.ndarray[np.double_t,ndim=2] J   = np.zeros((2,2),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] mle = np.zeros((self._nnod,self._nnod),np.double)

		# Ensure dealing with a 2D array
		xyel = xyzel if xyzel.shape[1] == 2 else self.transform2D(xyzel,xyzel)
		# Computation
		for igp in range(self._ngp):
			J[0,0] = 0.
			J[0,1] = 0.
			J[1,0] = 0.
			J[1,1] = 0.
			# Compute Jacobian
			for inod in range(self._nnod):
				J[0,0] += xyel[inod,0] * gradi[0,inod,igp]
				J[0,1] += xyel[inod,0] * gradi[1,inod,igp]
				J[1,0] += xyel[inod,1] * gradi[0,inod,igp]
				J[1,1] += xyel[inod,1] * gradi[1,inod,igp]
			# Determinant and inverse of Jacobian
			detJ = J[0,0]*J[1,1] - J[1,0]*J[0,1]
			# Element integral
			for inod in range(self._nnod):
				for jnod in range(self._nnod):
					mle[inod,jnod] +=  weigp[igp]*detJ*shapef[inod,igp]*shapef[jnod,igp]

		return mle

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def find_stz(Element2D self, double[:] xyz, double[:,:] xyzel, int max_iter=20, double tol=1e-10):
		'''
		Find a position of the point in xyz coordinates
		in element local coordinates stz:
		IN:
			> xyz(1,3):              position of the point
			> xyzel(nnod,3): position of the element nodes
		OUT:
			> stz(3,):               position of the point in local coordinates
		'''
		cdef int it, inod
		cdef double r, detT
		cdef double[:]  shapef = self._shapef[:,0]
		cdef double[:,:] gradi = self._gradi[:,:,0]

		cdef np.ndarray[np.double_t,ndim=1] stz   = np.zeros((2,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] delta = np.zeros((2,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] f     = np.zeros((2,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] aux   = np.zeros((2,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] T     = np.zeros((2,2),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] Tinv  = np.zeros((2,2),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] xy    = np.zeros((1,2),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] xyel  = np.zeros((self._nnod,2),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] xyz1  = np.zeros((1,3),dtype=np.double)
		stz[0] = self._posgp[0,0]
		stz[1] = self._posgp[0,1]

		# Ensure dealing with a 2D array
		xyz1[0,0] = xyz[0]
		xyz1[0,1] = xyz[1]
		xyz1[0,2] = xyz[2]
		xy   = xyz   if xyzel.shape[1] == 2 else self.transform2D(xyz1,xyzel)
		xyel = xyzel if xyzel.shape[1] == 2 else self.transform2D(xyzel,xyzel)

		# Compute residual
		for inod in range(self._nnod):
			aux[0] += shapef[inod]*xyel[inod,0]
			aux[1] += shapef[inod]*xyel[inod,1]
		f[0] = xy[0,0] - aux[0]
		f[1] = xy[0,1] - aux[1]
		r = sqrt(f[0]*f[0]+f[1]*f[1])

		# Newton-Raphson method
		for it in range(max_iter):
			T[0,0] = 0.
			T[0,1] = 0.
			T[1,0] = 0.
			T[1,1] = 0.
			# Compute T
			for inod in range(self._nnod):
				T[0,0] -= gradi[0,inod] * xyel[inod,0]
				T[0,1] -= gradi[1,inod] * xyel[inod,0]
				T[1,0] -= gradi[0,inod] * xyel[inod,1]
				T[1,1] -= gradi[1,inod] * xyel[inod,1]
			# Determinant 
			detT = T[0,0]*T[1,1] - T[1,0]*T[0,1]
			# Inverse of T
			Tinv[0,0] =  T[1,1]/detT
			Tinv[0,1] = -T[0,1]/detT
			Tinv[1,0] = -T[1,0]/detT
			Tinv[1,1] =  T[0,0]/detT
			# New guess
			delta[0] = -(Tinv[0,0]*f[0] + Tinv[0,1]*f[1])
			delta[1] = -(Tinv[1,0]*f[0] + Tinv[1,1]*f[1])
			stz[0]  += delta[0]
			stz[1]  += delta[1]
			# New shape functions and gradients
			shapef, gradi = self.shape_func(stz)
			# Compute new residual
			aux[0] = 0.
			aux[1] = 0.
			for inod in range(self._nnod):
				aux[0] += shapef[inod]*xyel[inod,0]
				aux[1] += shapef[inod]*xyel[inod,1]
			f[0] = xy[0,0] - aux[0]
			f[1] = xy[0,1] - aux[1]
			r = sqrt(f[0]*f[0]+f[1]*f[1])
			# Exit criteria
			if r < tol: break
		return stz

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def interpolate(Element2D self, double[:] stz, np.ndarray elfield):
		'''
		Interpolate a variable on a point inside the element:
		IN:
			> stz(2):                         position of the point in local coordinates
			> elfield(nnod,ndim): variable to be interpolated
		OUT:
			> out(1,ndim):            interpolated variable at stz
		'''
		cdef int ii, inod
		cdef int n = elfield.shape[0], m = elfield.shape[1] if elfield.ndim > 1 else 0
		cdef double[:]  shapef
		cdef np.ndarray out

		# Recover the shape function at the point stz
		shapef,_ = self.shape_func(stz)

		# Compute output array
		if m > 0:
			# Allocate output array
			out = np.zeros((1,m),dtype=np.double)
			# Compute dot           
			for inod in range(self._nnod):
				for ii in range(m):
					out[0,ii] += shapef[inod]*elfield[inod,ii]
		else:
			# Allocate output array
			out = np.zeros((1,),dtype=np.double)
			# Compute dot           
			for inod in range(self._nnod):
				out[0] += shapef[inod]*elfield[inod]
		# Return output
		return out

	@property
	def nnod(Element2D self):
		return self._nnod
	
	@property
	def ngauss(Element2D self):
		return self._ngp

	@property
	def ndim(Element2D self):
		return 2

	@property
	def nodes(Element2D self):
		return self._nodeList

	@property
	def posnod(Element2D self):
		return self._posnod

	@property
	def posgp(Element2D self):
		return self._posgp

	@property
	def shape(Element2D self):
		return self._shapef
	
	@property
	def grad(Element2D self):
		return self._gradi


cdef class Element3D:
	'''
	Basic FEM parent class that implements the
	basic operations for a 3D element.
	'''
	cdef int  _ngp, _nnod
	cdef bint _d_comp, _i_comp

	cdef int[:] _nodeList

	cdef double[:]     _weigp
	cdef double[:,:]   _posnod, _posgp, _shapef
	cdef double[:,:,:] _gradi

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)        
	def __init__(Element3D self, int[:] nodeList, int ngauss, bint SEM=False):
		'''
		Define an element given a list of nodes and 
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:                  number of Gauss points
		'''
		cdef int ii

		self._ngp      = ngauss
		self._nnod     = len(nodeList)
		self._nodeList = np.zeros((self.nnod,),dtype=np.int32)

		for ii in range(self._nnod):
			self._nodeList[ii] = nodeList[ii] 

		if not SEM:
			self._posnod   = np.zeros((self.nnod,3),dtype=np.double)             # Nodes position in local coordinates
		self._posgp    = np.zeros((self.ngauss,3),dtype=np.double)           # Gauss points position
		self._weigp    = np.zeros((self.ngauss,),dtype=np.double)            # Gauss points weights
		self._shapef   = np.zeros((self.nnod,self.ngauss),dtype=np.double)   # Shape function
		self._gradi    = np.zeros((3,self.nnod,self.ngauss),dtype=np.double) # Gradient function (in local coordinates)

	def __str__(Element3D self):
		s = 'Element 3D nnod=%d\n' % self._nnod
		return s

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	def __len__(Element3D self):
		return self._nnod

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	def __eq__(Element3D self, Element3D other):
		'''
		Element1 == Element2
		'''
		if other is None:
			return type(self) == None
		else:
			return self._ngp == other._ngp and self._nnod == other._nnod

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def centroid(Element3D self, double[:,:] xyzel):
		'''
		Element centroid
		'''
		cdef int inod, nnod = self._nnod
		cdef np.ndarray[np.double_t,ndim=1] cen = np.zeros((3,),dtype=np.double)
		# Compute center
		for inod in range(nnod):
			cen[0] += xyzel[inod,0]
			cen[1] += xyzel[inod,1]
			cen[2] += xyzel[inod,2]
		cen[0] /= nnod
		cen[1] /= nnod
		cen[2] /= nnod
		return cen

	@cython.boundscheck(False) # turn off bounds-checking for entizre function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def nodes2gp(Element3D self,np.ndarray elfield):
		'''
		Transforms an entry field on the nodes
		(or their position) to the Gauss point 
		equivalent.
		'''
		cdef int         igauss, inod, idim, nnod = self._nnod, ngp = self._ngp
		cdef np.ndarray  field_gp
		cdef double[:,:] shapef = self._shapef
		# Compute dot
		if elfield.ndim > 1:
			ndim = elfield.shape[1] 
			field_gp = np.zeros((ngp,ndim),dtype=np.double)      
			
			for igauss in range(ngp):
				for inod in range(nnod):
					for idim in range(ndim):
						field_gp[igauss,idim] += shapef[inod,igauss]*elfield[inod,idim]
		else:
			ndim = 0
			field_gp = np.zeros((ngp,),dtype=np.double)      
			
			for igauss in range(ngp):
				for inod in range(nnod):
						field_gp[igauss] += shapef[inod,igauss]*elfield[inod]
		return field_gp

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def derivative(Element3D self, double[:,:] xyzel):
		'''
		Derivative of the element:
		IN:
			> xyzel(nnod,3):                position of the nodes in cartesian coordinates
		OUT:
			> deri(3,nnod,ngauss):  derivatives per each gauss point
			> vol(ngauss):                  volume per each gauss point
			> mle(nnod):                    lumped mass matrix per each node
		'''
		cdef int        igp, inod
		cdef double detJ, t1, t2, t3
		cdef double[:]     weigp  = self._weigp
		cdef double[:,:,:] gradi  = self._gradi

		cdef np.ndarray[np.double_t,ndim=2] J    = np.zeros((3,3),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] Jinv = np.zeros((3,3),dtype=np.double)

		cdef np.ndarray[np.double_t,ndim=1] vol  = np.zeros((self.ngauss,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=3] deri = np.zeros((3,self.nnod,self.ngauss),dtype=np.double)

		for igp in range(self._ngp):
			J[0,0] = 0.
			J[0,1] = 0.
			J[0,2] = 0.
			J[1,0] = 0.
			J[1,1] = 0.
			J[1,2] = 0.
			J[2,0] = 0.
			J[2,1] = 0.
			J[2,2] = 0.
			# Compute Jacobian
			for inod in range(self._nnod):
				J[0,0] += xyzel[inod,0] * gradi[0,inod,igp]
				J[0,1] += xyzel[inod,0] * gradi[1,inod,igp]
				J[0,2] += xyzel[inod,0] * gradi[2,inod,igp]
				J[1,0] += xyzel[inod,1] * gradi[0,inod,igp]
				J[1,1] += xyzel[inod,1] * gradi[1,inod,igp]
				J[1,2] += xyzel[inod,1] * gradi[2,inod,igp]
				J[2,0] += xyzel[inod,2] * gradi[0,inod,igp]
				J[2,1] += xyzel[inod,2] * gradi[1,inod,igp]
				J[2,2] += xyzel[inod,2] * gradi[2,inod,igp]
			# Determinant 
			t1   =  J[1,1]*J[2,2] - J[2,1]*J[1,2]
			t2   = -J[1,0]*J[2,2] + J[2,0]*J[1,2]
			t3   =  J[1,0]*J[2,1] - J[2,0]*J[1,1]
			detJ =  J[0,0]*t1 + J[0,1]*t2 + J[0,2]*t3
			# Inverse of Jacobian
			Jinv[0,0] = t1/detJ
			Jinv[1,0] = t2/detJ
			Jinv[2,0] = t3/detJ
			Jinv[1,1] = ( J[0,0]*J[2,2] - J[2,0]*J[0,2])/detJ
			Jinv[2,1] = (-J[0,0]*J[2,1] + J[0,1]*J[2,0])/detJ
			Jinv[2,2] = ( J[0,0]*J[1,1] - J[1,0]*J[0,1])/detJ
			Jinv[0,1] = (-J[0,1]*J[2,2] + J[2,1]*J[0,2])/detJ
			Jinv[0,2] = ( J[0,1]*J[1,2] - J[1,1]*J[0,2])/detJ
			Jinv[1,2] = (-J[0,0]*J[1,2] + J[1,0]*J[0,2])/detJ
			# Element derivatives & lumped mass
			for inod in range(self._nnod):
				deri[0,inod,igp] = Jinv[0,0]*gradi[0,inod,igp] + Jinv[1,0]*gradi[1,inod,igp] + Jinv[2,0]*gradi[2,inod,igp]
				deri[1,inod,igp] = Jinv[0,1]*gradi[0,inod,igp] + Jinv[1,1]*gradi[1,inod,igp] + Jinv[2,1]*gradi[2,inod,igp]
				deri[2,inod,igp] = Jinv[0,2]*gradi[0,inod,igp] + Jinv[1,2]*gradi[1,inod,igp] + Jinv[2,2]*gradi[2,inod,igp]
			# Element volume
			vol[igp] = detJ*weigp[igp]

		return deri, vol

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def integrative(Element3D self, double[:,:] xyzel):
		'''
		Integral of the element:
		IN:
			> xyzel(nnod,3):          position of the nodes in cartesian coordinates
		OUT:
			> integ(nnod,ngauss): integral per each gauss point
		'''
		cdef int           igp, inod
		cdef double        detJ, t1, t2, t3
		cdef double[:]     weigp  = self._weigp
		cdef double[:,:]   shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi

		cdef np.ndarray[np.double_t,ndim=2] J     = np.zeros((3,3),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] integ = np.zeros((self.nnod, self.ngauss),dtype=np.double)

		if not self._i_comp:
			for igp in range(self._ngp):
				J[0,0] = 0.
				J[0,1] = 0.
				J[0,2] = 0.
				J[1,0] = 0.
				J[1,1] = 0.
				J[1,2] = 0.
				J[2,0] = 0.
				J[2,1] = 0.
				J[2,2] = 0.
				# Compute Jacobian
				for inod in range(self._nnod):
					J[0,0] += xyzel[inod,0] * gradi[0,inod,igp]
					J[0,1] += xyzel[inod,0] * gradi[1,inod,igp]
					J[0,2] += xyzel[inod,0] * gradi[2,inod,igp]
					J[1,0] += xyzel[inod,1] * gradi[0,inod,igp]
					J[1,1] += xyzel[inod,1] * gradi[1,inod,igp]
					J[1,2] += xyzel[inod,1] * gradi[2,inod,igp]
					J[2,0] += xyzel[inod,2] * gradi[0,inod,igp]
					J[2,1] += xyzel[inod,2] * gradi[1,inod,igp]
					J[2,2] += xyzel[inod,2] * gradi[2,inod,igp]
				# Determinant 
				t1   =  J[1,1]*J[2,2] - J[2,1]*J[1,2]
				t2   = -J[1,0]*J[2,2] + J[2,0]*J[1,2]
				t3   =  J[1,0]*J[2,1] - J[2,0]*J[1,1]
				detJ =  J[0,0]*t1 + J[0,1]*t2 + J[0,2]*t3
				# Element integral
				for inod in range(self._nnod):
					integ[inod,igp] = weigp[igp]*shapef[inod,igp]*detJ
		
		return integ

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def consistent(Element3D self, double[:,:] xyzel):
		'''
		Consistent mass matrix of the element:
		IN:
			> xyel(nnod,3):   position of the nodes in cartesian coordinates
		OUT:
			> mle(nnod,nnod): consistent mass matrix over the Gauss points	
		'''
		cdef int           igp, inod, jnod
		cdef double        detJ, t1, t2, t3
		cdef double[:]     weigp  = self._weigp
		cdef double[:,:]   shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi

		cdef np.ndarray[np.double_t,ndim=2] J   = np.zeros((3,3),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] mle = np.zeros((self._nnod,self._nnod),np.double)

		for igp in range(self._ngp):
			J[0,0] = 0.
			J[0,1] = 0.
			J[0,2] = 0.
			J[1,0] = 0.
			J[1,1] = 0.
			J[1,2] = 0.
			J[2,0] = 0.
			J[2,1] = 0.
			J[2,2] = 0.
			# Compute Jacobian
			for inod in range(self._nnod):
				J[0,0] += xyzel[inod,0] * gradi[0,inod,igp]
				J[0,1] += xyzel[inod,0] * gradi[1,inod,igp]
				J[0,2] += xyzel[inod,0] * gradi[2,inod,igp]
				J[1,0] += xyzel[inod,1] * gradi[0,inod,igp]
				J[1,1] += xyzel[inod,1] * gradi[1,inod,igp]
				J[1,2] += xyzel[inod,1] * gradi[2,inod,igp]
				J[2,0] += xyzel[inod,2] * gradi[0,inod,igp]
				J[2,1] += xyzel[inod,2] * gradi[1,inod,igp]
				J[2,2] += xyzel[inod,2] * gradi[2,inod,igp]
			# Determinant 
			t1   =  J[1,1]*J[2,2] - J[2,1]*J[1,2]
			t2   = -J[1,0]*J[2,2] + J[2,0]*J[1,2]
			t3   =  J[1,0]*J[2,1] - J[2,0]*J[1,1]
			detJ =  J[0,0]*t1 + J[0,1]*t2 + J[0,2]*t3
			# Element mass matrix
			for inod in range(self._nnod):
				for jnod in range(self._nnod):
					mle[inod,jnod] += weigp[igp]*detJ*shapef[inod,igp]*shapef[jnod,igp]
		
		return mle

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def find_stz(Element3D self, double[:] xyz, double[:,:] xyzel, int max_iter=20, double tol=1e-10):
		'''
		Find a position of the point in xyz coordinates
		in element local coordinates stz:
		IN:
			> xyz(1,3):              position of the point
			> xyzel(nnod,3): position of the element nodes
		OUT:
			> stz(3,):               position of the point in local coordinates
		'''
		cdef int it, inod
		cdef double r, t1, t2, t3, detT
		cdef double[:]  shapef = self._shapef[:,0]
		cdef double[:,:] gradi = self._gradi[:,:,0]

		cdef np.ndarray[np.double_t,ndim=1] stz   = np.zeros((3,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] delta = np.zeros((3,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] f     = np.zeros((3,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] T     = np.zeros((3,3),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] Tinv  = np.zeros((3,3),dtype=np.double)
		stz[0] = self._posgp[0,0]
		stz[1] = self._posgp[0,1]
		stz[2] = self._posgp[0,2]

		# Compute residual
		for inod in range(self._nnod):
			f[0] += shapef[inod]*xyzel[inod,0]
			f[1] += shapef[inod]*xyzel[inod,1]
			f[2] += shapef[inod]*xyzel[inod,2]
		f[0] = xyz[0] - f[0]
		f[1] = xyz[1] - f[1]
		f[2] = xyz[2] - f[2]
		r = sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2])

		# Newton-Raphson method
		for it in range(max_iter):
			T[0,0] = 0.
			T[0,1] = 0.
			T[0,2] = 0.
			T[1,0] = 0.
			T[1,1] = 0.
			T[1,2] = 0.
			T[2,0] = 0.
			T[2,1] = 0.
			T[2,2] = 0.
			# Compute T
			for inod in range(self._nnod):
				T[0,0] -= xyzel[inod,0] * gradi[0,inod]
				T[0,1] -= xyzel[inod,0] * gradi[1,inod]
				T[0,2] -= xyzel[inod,0] * gradi[2,inod]
				T[1,0] -= xyzel[inod,1] * gradi[0,inod]
				T[1,1] -= xyzel[inod,1] * gradi[1,inod]
				T[1,2] -= xyzel[inod,1] * gradi[2,inod]
				T[2,0] -= xyzel[inod,2] * gradi[0,inod]
				T[2,1] -= xyzel[inod,2] * gradi[1,inod]
				T[2,2] -= xyzel[inod,2] * gradi[2,inod]
			# Determinant 
			t1   =  T[1,1]*T[2,2] - T[2,1]*T[1,2]
			t2   = -T[1,0]*T[2,2] + T[2,0]*T[1,2]
			t3   =  T[1,0]*T[2,1] - T[2,0]*T[1,1]
			detT =  T[0,0]*t1 + T[0,1]*t2 + T[0,2]*t3
			# Inverse of T
			Tinv[0,0] = t1/detT
			Tinv[1,0] = t2/detT
			Tinv[2,0] = t3/detT
			Tinv[1,1] = ( T[0,0]*T[2,2] - T[2,0]*T[0,2])/detT
			Tinv[2,1] = (-T[0,0]*T[2,1] + T[0,1]*T[2,0])/detT
			Tinv[2,2] = ( T[0,0]*T[1,1] - T[1,0]*T[0,1])/detT
			Tinv[0,1] = (-T[0,1]*T[2,2] + T[2,1]*T[0,2])/detT
			Tinv[0,2] = ( T[0,1]*T[1,2] - T[1,1]*T[0,2])/detT
			Tinv[1,2] = (-T[0,0]*T[1,2] + T[1,0]*T[0,2])/detT
			# New guess
			delta[0] = -(Tinv[0,0]*f[0] + Tinv[0,1]*f[1] + Tinv[0,2]*f[2])
			delta[1] = -(Tinv[1,0]*f[0] + Tinv[1,1]*f[1] + Tinv[1,2]*f[2])
			delta[2] = -(Tinv[2,0]*f[0] + Tinv[2,1]*f[1] + Tinv[2,2]*f[2])
			stz[0]  += delta[0]
			stz[1]  += delta[1]
			stz[2]  += delta[2]

			# New shape functions and gradients
			shapef, gradi = self.shape_func(stz)
			# Compute new residual
			f[0] = 0.
			f[1] = 0.
			f[2] = 0.
			for inod in range(self._nnod):
				f[0] += shapef[inod]*xyzel[inod,0]
				f[1] += shapef[inod]*xyzel[inod,1]
				f[2] += shapef[inod]*xyzel[inod,2]
			f[0] = xyz[0] - f[0]
			f[1] = xyz[1] - f[1]
			f[2] = xyz[2] - f[2]
			r = sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2])
			# Exit criteria
			if r < tol: break
		return stz

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def interpolate(Element3D self, double[:] stz, np.ndarray elfield):
		'''
		Interpolate a variable on a point inside the element:
		IN:
			> stz(3):                         position of the point in local coordinates
			> elfield(nnod,ndim): variable to be interpolated
		OUT:
			> out(1,ndim):            interpolated variable at stz
		'''
		cdef int ii, inod
		cdef int n = elfield.shape[0], m = elfield.shape[1] if elfield.ndim > 1 else 0
		cdef double[:]  shapef
		cdef np.ndarray out

		# Recover the shape function at the point stz
		shapef,_ = self.shape_func(stz)

		# Compute output array
		if m > 0:
			# Allocate output array
			out = np.zeros((1,m),dtype=np.double)
			# Compute dot           
			for inod in range(n):
				for ii in range(m):
					out[0,ii] += shapef[inod]*elfield[inod,ii]
		else:
			# Allocate output array
			out = np.zeros((1,),dtype=np.double)
			# Compute dot           
			for inod in range(n):
				out[0] += shapef[inod]*elfield[inod]
		# Return output
		return out

	@property
	def nnod(Element3D self):
		return self._nnod
	
	@property
	def ngauss(Element3D self):
		return self._ngp

	@property
	def ndim(Element3D self):
		return 3

	@property
	def nodes(Element3D self):
		return self._nodeList

	@property
	def posnod(Element3D self):
		return self._posnod

	@property
	def posgp(Element3D self):
		return self._posgp

	@property
	def shape(Element3D self):
		return self._shapef

	@property
	def grad(Element3D self):
		return self._gradi


cdef class Bar(Element1D):
	'''
	Bar element
	'''
	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check       
	def __init__(Bar self, int[:] nodeList, int ngauss=2):
		'''
		Define a bar element given a list of nodes and
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:                  number of Gauss points
		'''
		# check number of Gauss points
		if not ngauss == 1 and not ngauss == 2 and not ngauss == 3 and not ngauss==4:
			raiseError('Invalid number of Gauss points (%d)!'%ngauss)
		if nodeList[1] == -1:
			raiseError('Invalid Bar (%d)!'%len(nodeList))
		# Allocate memory by initializing the parent class
		super(Bar,self).__init__(nodeList,ngauss)
		# Gauss points positions and weights
		cdef double[:]   weigp = self._weigp
		cdef double[:,:] posgp = self._posgp, posnod = self._posnod
		posnod[0,0] = -1.
		posnod[1,0] =  1.
		if ngauss == 1:
			posgp[0,0] = 0.
			weigp[0]   = 2.
		if ngauss == 2:
			posgp[0,0] = -0.577350269189625764509148780502
			posgp[1,0] =  0.577350269189625764509148780502
			weigp[0]   =  1.
			weigp[1]   =  1.
		if ngauss == 3:
			posgp[0,0] = -0.774596669241483377035853079956
			posgp[1,0] =  0.
			posgp[2,0] =  0.774596669241483377035853079956
			weigp[0]   =  5./9.
			weigp[1]   =  8./9.
			weigp[2]   =  5./9.
		if ngauss == 4:
			posgp[0,0] = -0.861136311594052575223946488893
			posgp[1,0] = -0.339981043584856264802665759103
			posgp[2,0] =  0.339981043584856264802665759103
			posgp[3,0] =  0.861136311594052575223946488893
			weigp[0]   =  0.347854845137453857373063949222
			weigp[1]   =  0.652145154862546142626936050778
			weigp[2]   =  0.652145154862546142626936050778
			weigp[3]   =  0.347854845137453857373063949222
		self._posnod = posnod
		self._posgp  = posgp
		self._weigp  = weigp
		# Compute shape function and derivatives
		cdef int igp, inod
		cdef double[:,:]   shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi
		cdef np.ndarray[np.double_t,ndim=1] posgp_aux = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] shapef_aux
		cdef np.ndarray[np.double_t,ndim=2] gradi_aux
		for igp in range(ngauss):
			for inod in range(self._nnod):
				posgp_aux[inod] = posgp[igp,inod]
			shapef_aux, gradi_aux = self.shape_func(posgp_aux)
			for inod in range(self._nnod):
				shapef[inod,igp]  = shapef_aux[inod]
				gradi[0,inod,igp] = gradi_aux[0,inod]
				gradi[1,inod,igp] = gradi_aux[1,inod]
		self._shapef = shapef
		self._gradi  = gradi

	def __str__(Bar self):
		s = 'Bar nnod=%d' % self._nnod
		return s

	@property
	def type(Bar self):
		return 2

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)  
	def new(Bar self,int ngauss=2):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return Bar(self.nodes,ngauss)

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)                
	def shape_func(Bar self, double[:] stz):
		'''
		Shape function and gradient for a set of
		coordinates.
		'''
		cdef np.ndarray[np.double_t,ndim=1] shapef = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] gradi  = np.zeros((2,self._nnod),dtype=np.double)
		# Define the shape function in local coordinates
		shapef[0]       = 0.5*(1.-stz[0])
		shapef[1]       = 0.5*(1.+stz[0])
		# Define the gradient in local coordinates
		return shapef, gradi

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)
	def isinside(Bar self, double[:] xy, double[:,:] xyel, double epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		cdef int inod
		cdef double x, J, Jinv, rh_side, point_loc
		cdef double min_loc = 1 - epsi, max_loc = 1 + epsi

		cdef double[:,:,:] gradi  = self._gradi
		
		cdef np.ndarray[np.double_t,ndim=1] xel = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] xy1 = np.zeros((1,2),dtype=np.double)

		# Ensure dealing with a 2D array
		xy1[0,0] = xy[0]
		xy1[0,1] = xy[1]
		x   = xy   if xyel.shape[1] == 1 else self.transform1D(xy1, xyel)
		xel = xyel if xyel.shape[1] == 1 else self.transform1D(xyel,xyel)

		# Compute Jacobian
		J = 0.
		for inod in range(self._nnod):
			J += xel[inod] * gradi[0,inod,0]

		# Inverse of Jacobian
		Jinv = 1./J

		# Find the point in natural coordinates
		rh_side   = x - xel[0]
		point_loc = Jinv*rh_side

		# The point has to be between -1 and 1
		if point_loc >= min_loc and point_loc <= max_loc:
			return True
		return False


cdef class LinearTriangle(Element2D):
	'''
	Linear Triangle:
		|  3
		|       
		|
		|
		|  1       2
	'''
	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check       
	def __init__(LinearTriangle self, int[:] nodeList, int ngauss=3):
		'''
		Define a linear tetrahedron given a list of nodes and
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:                  number of Gauss points
		'''
		# check number of Gauss points
		if not ngauss == 1 and not ngauss == 3:
			raiseError('Invalid number of Gauss points (%d)!'%ngauss)
		if nodeList[2] == -1:
			raiseError('Invalid Linear Tiangle (%d)!'%len(nodeList))
		# Allocate memory by initializing the parent class
		super(LinearTriangle,self).__init__(nodeList,ngauss)
		# Gauss points positions and weights
		cdef double[:]   weigp = self._weigp
		cdef double[:,:] posgp = self._posgp, posnod = self._posnod
		posnod[0,0] = 0.
		posnod[0,1] = 0.
		posnod[1,0] = 1.
		posnod[1,1] = 0.
		posnod[2,0] = 0.
		posnod[2,1] = 1.
		if ngauss == 1:
			posgp[0,0] = 1./3.
			posgp[0,1] = 1./3.
			weigp[0]   = 1./2.
		if ngauss == 3:
			posgp[0,0] = 1./6.
			posgp[0,1] = 1./6.
			posgp[1,0] = 2./3.
			posgp[1,1] = 1./6.
			posgp[2,0] = 1./6.
			posgp[2,1] = 2./3.
			weigp[0]   = 1./6.
			weigp[1]   = 1./6.
			weigp[2]   = 1./6.
		self._posnod = posnod
		self._posgp  = posgp
		self._weigp  = weigp
		# Compute shape function and derivatives
		cdef int igp, inod
		cdef double[:,:]   shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi
		cdef np.ndarray[np.double_t,ndim=1] posgp_aux = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] shapef_aux
		cdef np.ndarray[np.double_t,ndim=2] gradi_aux
		for igp in range(ngauss):
			for inod in range(self._nnod):
				posgp_aux[inod] = posgp[igp,inod]
			shapef_aux, gradi_aux = self.shape_func(posgp_aux)
			for inod in range(self._nnod):
				shapef[inod,igp]  = shapef_aux[inod]
				gradi[0,inod,igp] = gradi_aux[0,inod]
				gradi[1,inod,igp] = gradi_aux[1,inod]
		self._shapef = shapef
		self._gradi  = gradi

	def __str__(LinearTriangle self):
		s = 'Linear triangle nnod=%d' % self._nnod
		return s

	@property
	def type(LinearTriangle self):
		return 10

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)  
	def new(LinearTriangle self,int ngauss=3):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return LinearTriangle(self.nodes,ngauss)

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	def shape_func(LinearTriangle self, double[:] stz):
		'''
		Shape function and gradient for a set of
		coordinates.
		'''
		cdef np.ndarray[np.double_t,ndim=1] shapef = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] gradi  = np.zeros((2,self._nnod),dtype=np.double)
		# Define the shape function in local coordinates
		shapef[0]  = 1. - stz[0] - stz[1]
		shapef[1]  = stz[0]
		shapef[2]  = stz[1]
		# Define the gradient in local coordinates
		gradi[0,0] =-1.
		gradi[1,0] =-1.
		gradi[0,1] = 1.
		gradi[1,1] = 0.
		gradi[0,2] = 0.
		gradi[1,2] = 1.

		return shapef, gradi

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)
	def isinside(LinearTriangle self, double[:] xyz, double[:,:] xyzel, double epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		cdef int inod
		cdef double detJ, ezzzt
		cdef double min_loc = -epsi, max_loc = 1 + epsi

		cdef double[:,:,:] gradi  = self._gradi
		
		cdef np.ndarray[np.double_t,ndim=1] rh_side   = np.zeros((2,) ,dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] point_loc = np.zeros((2,) ,dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] J         = np.zeros((2,2),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] Jinv      = np.zeros((2,2),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] xy        = np.zeros((1,2),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] xyel      = np.zeros((self._nnod,2),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] xyz1      = np.zeros((1,3),dtype=np.double)

		# Ensure dealing with a 2D array
		xyz1[0,0] = xyz[0]
		xyz1[0,1] = xyz[1]
		xyz1[0,2] = xyz[2]
		xy   = xyz   if xyzel.shape[1] == 2 else self.transform2D(xyz1, xyzel)
		xyel = xyzel if xyzel.shape[1] == 2 else self.transform2D(xyzel,xyzel)

		# Compute Jacobian
		for inod in range(self._nnod):
			J[0,0] += xyel[inod,0] * gradi[0,inod,0]
			J[0,1] += xyel[inod,0] * gradi[1,inod,0]
			J[1,0] += xyel[inod,1] * gradi[0,inod,0]
			J[1,1] += xyel[inod,1] * gradi[1,inod,0]

		# Determinant 
		detJ = J[0,0]*J[1,1] - J[1,0]*J[0,1]

		# Inverse of Jacobian
		Jinv[0,0] =  J[1,1]/detJ
		Jinv[0,1] = -J[0,1]/detJ
		Jinv[1,0] = -J[1,0]/detJ
		Jinv[1,1] =  J[0,0]/detJ

		# Find the point in natural coordinates
		rh_side[0]   = xy[0,0] - xyel[0,0]
		rh_side[1]   = xy[0,1] - xyel[0,1]
		point_loc[0] = Jinv[0,0]*rh_side[0] + Jinv[0,1]*rh_side[1]
		point_loc[1] = Jinv[1,0]*rh_side[0] + Jinv[1,1]*rh_side[1]

		# The point has to be between 0 and 1
		if point_loc[0] >= min_loc and point_loc[0] <= max_loc:
			if point_loc[1] >= min_loc and point_loc[1] <= max_loc:
				# The sum of the points in natural coordinates also
				# has to be between 0 and 1
				ezzzt = 1 - (point_loc[0]+point_loc[1])
				if ezzzt >= min_loc and ezzzt <= max_loc:
					return True
		return False


cdef class LinearQuadrangle(Element2D):
	'''
	Linear Quadrangle:
		|  3       4
		|
		|
		|
		|  1       2
	'''
	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check       
	def __init__(LinearQuadrangle self, int[:] nodeList, int ngauss=4):
		'''
		Define a linear Quadrangle given a list of nodes and 
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:                  number of Gauss points
		'''
		# check number of Gauss points
		if not ngauss == 1 and not ngauss == 4:
			raiseError('Invalid number of Gauss points (%d)!'%ngauss)
		if nodeList[3] == -1:
			raiseError('Invalid Linear Tiangle (%d)!'%len(nodeList))
		# Allocate memory by initializing the parent class
		super(LinearQuadrangle,self).__init__(nodeList,ngauss)
		# Gauss points positions and weights
		cdef double[:]   weigp = self._weigp
		cdef double[:,:] posgp = self._posgp, posnod = self._posnod
		posnod[0,0] = -1.0
		posnod[0,1] = -1.0
		posnod[1,0] =  1.0
		posnod[1,1] = -1.0
		posnod[2,0] =  1.0
		posnod[2,1] =  1.0
		posnod[3,0] = -1.0
		posnod[3,1] =  1.0
		if ngauss == 1:
			posgp[0,0] = 0.
			posgp[0,1] = 0.
			weigp[0]   = 4.
		if ngauss == 4:
			q = 1.0/sqrt(3.0)
			posgp[0,0] = -q
			posgp[0,1] = -q
			posgp[1,0] =  q
			posgp[1,1] = -q
			posgp[2,0] =  q
			posgp[2,1] =  q
			posgp[3,0] = -q
			posgp[3,1] =  q
			weigp[0]   =  1.0
			weigp[1]   =  1.0
			weigp[2]   =  1.0
			weigp[3]   =  1.0
		self._posnod = posnod
		self._posgp  = posgp
		self._weigp  = weigp
		# Compute shape function and derivatives
		cdef int igp, inod
		cdef double[:,:]   shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi
		cdef np.ndarray[np.double_t,ndim=1] posgp_aux = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] shapef_aux
		cdef np.ndarray[np.double_t,ndim=2] gradi_aux
		for igp in range(ngauss):
			for inod in range(self._nnod):
				posgp_aux[inod] = posgp[igp,inod]
			shapef_aux, gradi_aux = self.shape_func(posgp_aux)
			for inod in range(self._nnod):
				shapef[inod,igp]  = shapef_aux[inod]
				gradi[0,inod,igp] = gradi_aux[0,inod]
				gradi[1,inod,igp] = gradi_aux[1,inod]
		self._shapef = shapef
		self._gradi  = gradi

	def __str__(LinearQuadrangle self):
		s = 'Linear quadrangle nnod=%d' % self._nnod
		return s

	@property
	def type(LinearQuadrangle self):
		return 12

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)  
	def new(LinearQuadrangle self,int ngauss=4):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return LinearQuadrangle(self.nodes,ngauss)

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)                
	def shape_func(LinearQuadrangle self, double[:] stz):
		'''
		Shape function and gradient for a set of 
		coordinates.
		'''
		cdef np.ndarray[np.double_t,ndim=1] shapef = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] gradi  = np.zeros((2,self._nnod),dtype=np.double)
		# Define the shape function in local coordinates
		shapef[0]  = 0.25*(1.0-stz[0])*(1.0-stz[1])
		shapef[1]  = 0.25*(1.0+stz[0])*(1.0-stz[1])
		shapef[2]  = 0.25*(1.0+stz[0])*(1.0+stz[1])
		shapef[3]  = 0.25*(1.0-stz[0])*(1.0+stz[1])
		# Define the gradient in local coordinates
		gradi[0,0] = 0.25*(-1.0+stz[1])
		gradi[1,0] = 0.25*(-1.0+stz[0])
		gradi[0,1] = 0.25*( 1.0-stz[1])
		gradi[1,1] = 0.25*(-1.0-stz[0])
		gradi[0,2] = 0.25*( 1.0+stz[1])
		gradi[1,2] = 0.25*( 1.0+stz[0])
		gradi[0,3] = 0.25*(-1.0-stz[1])
		gradi[1,3] = 0.25*( 1.0-stz[0])

		return shapef, gradi

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)
	def isinside(LinearQuadrangle self, double[:] xyz, double[:,:] xyzel, double epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		cdef int ii
		cdef LinearTriangle triangle
		cdef np.ndarray[np.int32_t,ndim=1]  nodeList = np.zeros((3,),dtype=np.int32)
		cdef np.ndarray[np.double_t,ndim=2] xyzel1   = np.zeros((3,3),dtype=np.double)
		# Split the element into tetras
		nodeList[0] = self._nodeList[0]
		nodeList[1] = self._nodeList[1]
		nodeList[2] = self._nodeList[3]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[0,ii]
			xyzel1[1,ii] = xyzel[1,ii]
			xyzel1[2,ii] = xyzel[3,ii]
		triangle = LinearTriangle(nodeList,1)
		if triangle.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[3]
		nodeList[1] = self._nodeList[2]
		nodeList[2] = self._nodeList[1]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[3,ii]
			xyzel1[1,ii] = xyzel[2,ii]
			xyzel1[2,ii] = xyzel[1,ii]
		triangle = LinearTriangle(nodeList,1)
		if triangle.isinside(xyz,xyzel1,epsi): return True
		return False

cdef class pOrderQuadrangle(Element2D):
	'''
	SEM pOrder Quadrangle: gauss points = nodes. GLL quadrature.
	'''
	cdef int _porder
	cdef double[:] _pnodes
	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check       
	def __init__(pOrderQuadrangle self, int[:] nodeList, int ngauss, double[:] xi, double[:,:] posnod, double[:] weigp, double[:,:] shapef, double[:,:,:] gradi):
		# check number of Gauss points
		self._porder = np.sqrt(ngauss) - 1
		if not len(nodeList) == ngauss:
			raiseError('Invalid pOrder Quadrangle! Number of nodes (%d) is different to pOrder (%d)' % (len(nodeList),self._porder))
		# Allocate memory by initializing the parent class
		super(pOrderQuadrangle, self).__init__(nodeList, ngauss, SEM=True)
		# Nodes/Gauss points positions and weights
		self._pnodes = xi
		self._posnod = posnod
		self._weigp  = weigp
		# Shape function and derivatives in the Gauss Points
		self._shapef = shapef
		self._gradi  = gradi
		
	@property
	def type(pOrderQuadrangle self):
		return 15

cdef class LinearTetrahedron(Element3D):
	'''
	Linear tetrahedron: s=[0:1], t=[0:1], z=[0:1]
	'''
	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check       
	def __init__(LinearTetrahedron self, int[:] nodeList, int ngauss=4):
		'''
		Define a linear tetrahedron given a list of nodes and 
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:                  number of Gauss points
		'''
		# check number of Gauss points
		if not ngauss == 1 and not ngauss == 4:
			raiseError('Invalid number of Gauss points (%d)!'%ngauss)
		if nodeList[3] == -1:
			raiseError('Invalid Linear Tetrahedron (%d)!'%len(nodeList))
		# Allocate memory by initializing the parent class
		super(LinearTetrahedron,self).__init__(nodeList,ngauss)
		# Gauss points positions and weights
		cdef double[:]   weigp = self._weigp
		cdef double[:,:] posgp = self._posgp, posnod = self._posnod
		posnod[0,0] = 0.
		posnod[0,1] = 0.
		posnod[0,2] = 0.
		posnod[1,0] = 1.
		posnod[1,1] = 0.
		posnod[1,2] = 0.
		posnod[2,0] = 0.
		posnod[2,1] = 1.
		posnod[2,2] = 0.
		posnod[3,0] = 0.
		posnod[3,1] = 0.
		posnod[3,2] = 1.
		if ngauss == 1:
			posgp[0,0] = 1./4.
			posgp[0,1] = 1./4.
			posgp[0,2] = 1./4.
			weigp[0]   = 1./6.
		if ngauss == 4:
			a = 0.5854101966249685
			b = 0.1381966011250105
			posgp[0,0] = b
			posgp[0,1] = b
			posgp[0,2] = b
			posgp[1,0] = a
			posgp[1,1] = b
			posgp[1,2] = b
			posgp[2,0] = b
			posgp[2,1] = a
			posgp[2,2] = b
			posgp[3,0] = b
			posgp[3,1] = b
			posgp[3,2] = a
			weigp[0]   = 1./24.
			weigp[1]   = 1./24.
			weigp[2]   = 1./24.
			weigp[3]   = 1./24.
		self._posnod = posnod
		self._posgp  = posgp
		self._weigp  = weigp
		# Compute shape function and derivatives
		cdef int igp, inod
		cdef double[:,:]   shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi
		cdef np.ndarray[np.double_t,ndim=1] posgp_aux = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] shapef_aux
		cdef np.ndarray[np.double_t,ndim=2] gradi_aux
		for igp in range(ngauss):
			for inod in range(self._nnod):
				posgp_aux[inod] = posgp[igp,inod]
			shapef_aux, gradi_aux = self.shape_func(posgp_aux)
			for inod in range(self._nnod):
				shapef[inod,igp]  = shapef_aux[inod]
				gradi[0,inod,igp] = gradi_aux[0,inod]
				gradi[1,inod,igp] = gradi_aux[1,inod]
				gradi[2,inod,igp] = gradi_aux[2,inod]
		self._shapef = shapef
		self._gradi  = gradi

	def __str__(LinearTetrahedron self):
		s = 'Linear tetrahedron nnod=%d' % self._nnod
		return s

	@property
	def type(LinearTetrahedron self):
		return 30

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)  
	def new(LinearTetrahedron self,int ngauss=4):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return LinearTetrahedron(self.nodes,ngauss)

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	def shape_func(LinearTetrahedron self, double[:] stz):
		'''
		Shape function and gradient for a set of 
		coordinates.
		'''
		cdef np.ndarray[np.double_t,ndim=1] shapef = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] gradi  = np.zeros((3,self._nnod),dtype=np.double)
		# Define the shape function in local coordinates
		shapef[0]  = 1. - stz[0] - stz[1] - stz[2]
		shapef[1]  = stz[0]
		shapef[2]  = stz[1]
		shapef[3]  = stz[2]
		# Define the gradient in local coordinates
		gradi[0,0] = -1.
		gradi[1,0] = -1.
		gradi[2,0] = -1.
		gradi[0,1] =  1.
		gradi[1,2] =  1.
		gradi[2,3] =  1.

		return shapef, gradi

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)
	def isinside(LinearTetrahedron self, double[:] xyz, double[:,:] xyzel, double epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		cdef int inod
		cdef double t1, t2, t3, detJ, ezzzt
		cdef double min_loc = -epsi, max_loc = 1 + epsi

		cdef double[:,:,:] gradi  = self._gradi
		
		cdef np.ndarray[np.double_t,ndim=1] rh_side = np.zeros((3,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] point_loc = np.zeros((3,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] J    = np.zeros((3,3),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] Jinv = np.zeros((3,3),dtype=np.double)
	
		# Compute Jacobian
		for inod in range(self._nnod):
			J[0,0] += xyzel[inod,0] * gradi[0,inod,0]
			J[0,1] += xyzel[inod,0] * gradi[1,inod,0]
			J[0,2] += xyzel[inod,0] * gradi[2,inod,0]
			J[1,0] += xyzel[inod,1] * gradi[0,inod,0]
			J[1,1] += xyzel[inod,1] * gradi[1,inod,0]
			J[1,2] += xyzel[inod,1] * gradi[2,inod,0]
			J[2,0] += xyzel[inod,2] * gradi[0,inod,0]
			J[2,1] += xyzel[inod,2] * gradi[1,inod,0]
			J[2,2] += xyzel[inod,2] * gradi[2,inod,0]

		# Determinant 
		t1   =  J[1,1]*J[2,2] - J[2,1]*J[1,2]
		t2   = -J[1,0]*J[2,2] + J[2,0]*J[1,2]
		t3   =  J[1,0]*J[2,1] - J[2,0]*J[1,1]
		detJ =  J[0,0]*t1 + J[0,1]*t2 + J[0,2]*t3

		# Inverse of Jacobian
		Jinv[0,0] = t1/detJ
		Jinv[1,0] = t2/detJ
		Jinv[2,0] = t3/detJ
		Jinv[1,1] = ( J[0,0]*J[2,2] - J[2,0]*J[0,2])/detJ
		Jinv[2,1] = (-J[0,0]*J[2,1] + J[0,1]*J[2,0])/detJ
		Jinv[2,2] = ( J[0,0]*J[1,1] - J[1,0]*J[0,1])/detJ
		Jinv[0,1] = (-J[0,1]*J[2,2] + J[2,1]*J[0,2])/detJ
		Jinv[0,2] = ( J[0,1]*J[1,2] - J[1,1]*J[0,2])/detJ
		Jinv[1,2] = (-J[0,0]*J[1,2] + J[1,0]*J[0,2])/detJ

		# Find the point in natural coordinates
		rh_side[0]   = xyz[0] - xyzel[0,0]
		rh_side[1]   = xyz[1] - xyzel[0,1]
		rh_side[2]   = xyz[2] - xyzel[0,2]
		point_loc[0] = Jinv[0,0]*rh_side[0] + Jinv[0,1]*rh_side[1] + Jinv[0,2]*rh_side[2]
		point_loc[1] = Jinv[1,0]*rh_side[0] + Jinv[1,1]*rh_side[1] + Jinv[1,2]*rh_side[2]
		point_loc[2] = Jinv[2,0]*rh_side[0] + Jinv[2,1]*rh_side[1] + Jinv[2,2]*rh_side[2]

		# The point has to be between 0 and 1
		if point_loc[0] >= min_loc and point_loc[0] <= max_loc:
			if point_loc[1] >= min_loc and point_loc[1] <= max_loc:
				if point_loc[2] >= min_loc and point_loc[2] <= max_loc:
					# The sum of the points in natural coordinates also
					# has to be between 0 and 1
					ezzzt = 1 - (point_loc[0]+point_loc[1]+point_loc[2])
					if ezzzt >= min_loc and ezzzt <= max_loc:
						return True
		return False


cdef class LinearPyramid(Element3D):
	'''
	Linear Pyramid: s=[-1:1], t=[-1:1], z=[-1:1]
	'''
	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check       
	def __init__(LinearPyramid self, int[:] nodeList, int ngauss=5):
		'''
		Define a linear pyramid given a list of nodes and 
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:                  number of Gauss points
		'''
		# check number of Gauss points
		if not ngauss==1 and not ngauss==5:
			raiseError('Invalid number of Gauss points (%d)!'%ngauss)
		if nodeList[4] == -1:
			raiseError('Invalid Linear Pyramid (%d)!'%len(nodeList))
		# Allocate memory by initializing the parent class
		super(LinearPyramid,self).__init__(nodeList,ngauss)
		# Gauss points positions and weights
		cdef int ii
		cdef double g1, j, k
		cdef double[:]   weigp = self._weigp
		cdef double[:,:] posgp = self._posgp, posnod = self._posnod
		posnod[0,0] = -1.
		posnod[0,1] = -1.
		posnod[0,2] = -1.
		posnod[1,0] =  1.
		posnod[1,1] = -1.
		posnod[1,2] = -1.
		posnod[2,0] =  1.
		posnod[2,1] =  1.
		posnod[2,2] = -1.
		posnod[3,0] = -1.
		posnod[3,1] =  1.
		posnod[3,2] = -1.
		posnod[4,0] =  0.
		posnod[4,1] =  0.
		posnod[4,2] =  1.
		cdef np.ndarray[np.double_t,ndim=2] jk = np.zeros((4,2),dtype=np.double)
		jk[0,0]  = -1.
		jk[0,1]  = -1.
		jk[1,0]  =  1.
		jk[1,1]  = -1.
		jk[2,0]  =  1.
		jk[2,1]  =  1.
		jk[3,0]  = -1.
		jk[3,1]  =  1.
		# Gauss points positions and weights
		if ngauss == 1:
			posgp[0,0] = 0.
			posgp[0,1] = 0.
			posgp[0,2] = 0.5
			weigp[0]   = 128./27.
		if ngauss == 5:
			g1 = 8.*sqrt(2./15.)/5.
			for ii in range(4):
				j = jk[ii,0]
				k = jk[ii,1]
				posgp[ii,0] = j*g1
				posgp[ii,1] = k*g1
				posgp[ii,2] = -2./3.
				weigp[ii]   = 81./100.
			posgp[4,0] = 0.
			posgp[4,1] = 0.
			posgp[4,2] = 2./5.
			weigp[4]   = 125./27. 
		self._posnod = posnod
		self._posgp  = posgp
		self._weigp  = weigp
		# Compute shape function and derivatives
		cdef int igp, inod
		cdef double[:,:]   shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi
		cdef np.ndarray[np.double_t,ndim=1] posgp_aux = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] shapef_aux
		cdef np.ndarray[np.double_t,ndim=2] gradi_aux
		for igp in range(ngauss):
			for inod in range(self._nnod):
				posgp_aux[inod] = posgp[igp,inod]
			shapef_aux, gradi_aux = self.shape_func(posgp_aux)
			for inod in range(self._nnod):
				shapef[inod,igp]  = shapef_aux[inod]
				gradi[0,inod,igp] = gradi_aux[0,inod]
				gradi[1,inod,igp] = gradi_aux[1,inod]
				gradi[2,inod,igp] = gradi_aux[2,inod]
		self._shapef = shapef
		self._gradi  = gradi

	def __str__(LinearPyramid self):
		s = 'Linear pyramid nnod=%d' % self._nnod
		return s

	@property
	def type(LinearPyramid self):
		return 32

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)  
	def new(LinearPyramid self,int ngauss=5):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return LinearPyramid(self.nodes,ngauss)

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)                
	def shape_func(LinearPyramid self, double[:] stz):
		'''
		Shape function and gradient for a set of 
		coordinates.
		'''
		cdef np.ndarray[np.double_t,ndim=1] shapef = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] gradi  = np.zeros((3,self._nnod),dtype=np.double)
		# Define the shape function in local coordinates
		cdef double one8 = 0.125
		shapef[0] = one8*(1. - stz[0])*(1. - stz[1])*(1. - stz[2])
		shapef[1] = one8*(1. + stz[0])*(1. - stz[1])*(1. - stz[2])
		shapef[2] = one8*(1. + stz[0])*(1. + stz[1])*(1. - stz[2])
		shapef[3] = one8*(1. - stz[0])*(1. + stz[1])*(1. - stz[2])
		shapef[4] = 0.5*(1. + stz[2])           
		# Define the gradient in local coordinates
		gradi[0,0] = -one8*(1. - stz[1])*(1. - stz[2])
		gradi[1,0] = -one8*(1. - stz[0])*(1. - stz[2])
		gradi[2,0] = -one8*(1. - stz[0])*(1. - stz[1])
		gradi[0,1] =  one8*(1. - stz[1])*(1. - stz[2])
		gradi[1,1] = -one8*(1. + stz[0])*(1. - stz[2])
		gradi[2,1] = -one8*(1. + stz[0])*(1. - stz[1])
		gradi[0,2] =  one8*(1. + stz[1])*(1. - stz[2]) 
		gradi[1,2] =  one8*(1. + stz[0])*(1. - stz[2]) 
		gradi[2,2] = -one8*(1. + stz[0])*(1. + stz[1])
		gradi[0,3] = -one8*(1. + stz[1])*(1. - stz[2])
		gradi[1,3] =  one8*(1. - stz[0])*(1. - stz[2])
		gradi[2,3] = -one8*(1. - stz[0])*(1. + stz[1])
		gradi[0,4] = 0.
		gradi[1,4] = 0.
		gradi[2,4] = 0.5
		return shapef, gradi

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)                
	def isinside(LinearPyramid self, double[:] xyz, double[:,:] xyzel, double epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		cdef int ii
		cdef LinearTetrahedron tetra
		cdef np.ndarray[np.int32_t,ndim=1]  nodeList = np.zeros((4,),dtype=np.int32)
		cdef np.ndarray[np.double_t,ndim=2] xyzel1   = np.zeros((4,3),dtype=np.double)
		# Split the element into tetras
		nodeList[0] = self._nodeList[0]
		nodeList[1] = self._nodeList[1]
		nodeList[2] = self._nodeList[2]
		nodeList[3] = self._nodeList[4]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[0,ii]
			xyzel1[1,ii] = xyzel[1,ii]
			xyzel1[2,ii] = xyzel[2,ii]
			xyzel1[3,ii] = xyzel[4,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[0]
		nodeList[1] = self._nodeList[2]
		nodeList[2] = self._nodeList[3]
		nodeList[3] = self._nodeList[4]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[0,ii]
			xyzel1[1,ii] = xyzel[2,ii]
			xyzel1[2,ii] = xyzel[3,ii]
			xyzel1[3,ii] = xyzel[4,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		return False


cdef class LinearPrism(Element3D):
	'''
	Linear Prism: s=[0:1], t=[0:1], z=[0:1]
	'''
	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def __init__(LinearPrism self, int[:] nodeList, int ngauss=6):
		'''
		Define a linear prism given a list of nodes and 
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:                  number of Gauss points
		'''
		# check number of Gauss points
		if not ngauss == 1 and not ngauss==6:
			raiseError('Invalid number of Gauss points (%d)!'%ngauss)
		if nodeList[5] == -1:
			raiseError('Invalid Linear Prism (%d)!'%len(nodeList))
		# Allocate memory by initializing the parent class
		super(LinearPrism,self).__init__(nodeList,ngauss)       
		# Gauss points positions and weights
		cdef double[:]   weigp = self._weigp
		cdef double[:,:] posgp = self._posgp, posnod = self._posnod
		posnod[0,0] = 0.
		posnod[0,1] = 0.
		posnod[0,2] = 0.
		posnod[1,0] = 1.
		posnod[1,1] = 0.
		posnod[1,2] = 0.
		posnod[2,0] = 0.
		posnod[2,1] = 1.
		posnod[2,2] = 0.
		posnod[3,0] = 0.
		posnod[3,1] = 0.
		posnod[3,2] = 1.
		posnod[4,0] = 1.
		posnod[4,1] = 0.
		posnod[4,2] = 1.
		posnod[5,0] = 0.
		posnod[5,1] = 1.
		posnod[5,2] = 1.
		if ngauss == 1:
			posgp[0,0] = 1./3.
			posgp[0,1] = 1./3.
			posgp[0,2] = 1./2.
			weigp[0]   = 1./2.
		if ngauss == 6:
			posgp[0,0] = 2./3.
			posgp[0,1] = 1./6.
			posgp[0,2] = 0.21132486540518711774542560974902
			posgp[1,0] = 1./6.
			posgp[1,1] = 2./3.
			posgp[1,2] = 0.21132486540518711774542560974902
			posgp[2,0] = 1./6.
			posgp[2,1] = 1./6.
			posgp[2,2] = 0.21132486540518711774542560974902
			posgp[3,0] = 2./3.
			posgp[3,1] = 1./6.
			posgp[3,2] = 0.78867513459481288225457439025098
			posgp[4,0] = 1./6.
			posgp[4,1] = 2./3.
			posgp[4,2] = 0.78867513459481288225457439025098
			posgp[5,0] = 1./6.
			posgp[5,1] = 1./6.
			posgp[5,2] = 0.78867513459481288225457439025098
			weigp[0]   = 1./12.
			weigp[1]   = 1./12.
			weigp[2]   = 1./12.
			weigp[3]   = 1./12.
			weigp[4]   = 1./12.
			weigp[5]   = 1./12.
		self._posnod = posnod
		self._posgp  = posgp
		self._weigp  = weigp
		# Compute shape function and derivatives
		cdef int igp, inod
		cdef double[:,:]   shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi
		cdef np.ndarray[np.double_t,ndim=1] posgp_aux = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] shapef_aux
		cdef np.ndarray[np.double_t,ndim=2] gradi_aux
		for igp in range(ngauss):
			for inod in range(self._nnod):
				posgp_aux[inod] = posgp[igp,inod]
			shapef_aux, gradi_aux = self.shape_func(posgp_aux)
			for inod in range(self._nnod):
				shapef[inod,igp]  = shapef_aux[inod]
				gradi[0,inod,igp] = gradi_aux[0,inod]
				gradi[1,inod,igp] = gradi_aux[1,inod]
				gradi[2,inod,igp] = gradi_aux[2,inod]
		self._shapef = shapef
		self._gradi  = gradi

	def __str__(self):
		s = 'Linear prism nnod=%d' % self._nnod
		return s

	@property
	def type(LinearPrism self):
		return 34

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)  
	def new(LinearPrism self,int ngauss=6):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return LinearPrism(self.nodes,ngauss)

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)                
	def shape_func(LinearPrism self, double[:] stz):
		'''
		Shape function and gradient for a set of 
		coordinates.
		'''
		cdef np.ndarray[np.double_t,ndim=1] shapef = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] gradi  = np.zeros((3,self._nnod),dtype=np.double)
		# Define the shape function in local coordinates
		shapef[0] = (1. - stz[0] - stz[1])*(1. - stz[2])
		shapef[1] = stz[0]*(1. - stz[2])
		shapef[2] = stz[1]*(1. - stz[2])
		shapef[3] = (1. - stz[0] - stz[1])*stz[2]
		shapef[4] = stz[0]*stz[2]
		shapef[5] = stz[1]*stz[2]
		# Define the gradient in local coordinates
		gradi[0,0] = stz[2] - 1.
		gradi[1,0] = stz[2] - 1.
		gradi[2,0] = stz[0] + stz[1] - 1.
		gradi[0,1] = 1. - stz[2]
		gradi[2,1] = -stz[0]
		gradi[1,2] = 1. - stz[2]
		gradi[2,2] = -stz[1]
		gradi[0,3] = -stz[2]
		gradi[1,3] = -stz[2]
		gradi[2,3] = 1. - stz[0] - stz[1]
		gradi[0,4] = stz[2]                     
		gradi[2,4] = stz[0]
		gradi[1,5] = stz[2]
		gradi[2,5] = stz[1]
		return shapef, gradi

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)                
	def isinside(LinearPrism self, double[:] xyz, double[:,:] xyzel, double epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		tetras = [[0, 1, 2, 3], [4, 1, 3, 5], [5, 3, 2, 1]]
		cdef int ii
		cdef LinearTetrahedron tetra
		cdef np.ndarray[np.int32_t,ndim=1]  nodeList = np.zeros((4,),dtype=np.int32)
		cdef np.ndarray[np.double_t,ndim=2] xyzel1   = np.zeros((4,3),dtype=np.double)
		# Split the element into tetras
		nodeList[0] = self._nodeList[0]
		nodeList[1] = self._nodeList[1]
		nodeList[2] = self._nodeList[2]
		nodeList[3] = self._nodeList[3]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[0,ii]
			xyzel1[1,ii] = xyzel[1,ii]
			xyzel1[2,ii] = xyzel[2,ii]
			xyzel1[3,ii] = xyzel[3,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[4]
		nodeList[1] = self._nodeList[1]
		nodeList[2] = self._nodeList[3]
		nodeList[3] = self._nodeList[5]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[4,ii]
			xyzel1[1,ii] = xyzel[1,ii]
			xyzel1[2,ii] = xyzel[3,ii]
			xyzel1[3,ii] = xyzel[5,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[5]
		nodeList[1] = self._nodeList[3]
		nodeList[2] = self._nodeList[2]
		nodeList[3] = self._nodeList[1]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[5,ii]
			xyzel1[1,ii] = xyzel[3,ii]
			xyzel1[2,ii] = xyzel[2,ii]
			xyzel1[3,ii] = xyzel[1,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		return False


cdef class TrilinearBrick(Element3D):
	'''
	Trilinear brick
	'''
	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def __init__(TrilinearBrick self, int[:] nodeList, int ngauss=8):
		'''
		Define a trilinear brick given a list of nodes and 
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:                  number of Gauss points
		'''
		if nodeList[7] == -1:
			raiseError('Invalid Trilinear Brick (%d)!'%len(nodeList))
		# Allocate memory by initializing the parent class
		super(TrilinearBrick,self).__init__(nodeList,ngauss)
		# Gauss points positions and weights
		cdef double[:]   weigp = self._weigp
		cdef double[:,:] posgp = self._posgp, posnod = self._posnod
		cdef int igauss, ilocs, jlocs, klocs, nlocs = int( ngauss**(1./3.) )
		cdef np.ndarray[np.double_t,ndim=1] posgl = np.zeros((nlocs,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] weigl = np.zeros((nlocs,),dtype=np.double)
		posnod[0,0] = -1.
		posnod[0,1] = -1.
		posnod[0,2] = -1.
		posnod[1,0] =  1.
		posnod[1,1] = -1.
		posnod[1,2] = -1.
		posnod[2,0] =  1.
		posnod[2,1] =  1.
		posnod[2,2] = -1.
		posnod[3,0] = -1.
		posnod[3,1] =  1.
		posnod[3,2] = -1.
		posnod[4,0] = -1.
		posnod[4,1] = -1.
		posnod[4,2] =  1.
		posnod[5,0] =  1.
		posnod[5,1] = -1.
		posnod[5,2] =  1.
		posnod[6,0] =  1.
		posnod[6,1] =  1.
		posnod[6,2] =  1.
		posnod[7,0] = -1.
		posnod[7,1] =  1.
		posnod[7,2] =  1.
		if nlocs == 1:
			posgl[0] = 0.
			weigl[0] = 2.
		elif nlocs == 2:
			posgl[0] = -0.577350269189626
			posgl[1] =  0.577350269189626
			weigl[0] = 1.
			weigl[1] = 1.
		else:
			raiseError('Invalid number of Gauss points (%d)!'%ngauss)

		igauss = 0
		for ilocs in range(nlocs):
			for jlocs in range(nlocs):
				for klocs in range(nlocs):
					weigp[igauss] = weigl[ilocs]*weigl[jlocs]*weigl[klocs]
					posgp[igauss,0] = posgl[ilocs]
					posgp[igauss,1] = posgl[jlocs]
					posgp[igauss,2] = posgl[klocs]
					igauss += 1
		self._posnod = posnod
		self._posgp  = posgp
		self._weigp  = weigp
		# Compute shape function and derivatives
		cdef int igp, inod
		cdef double[:,:]   shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi
		cdef np.ndarray[np.double_t,ndim=1] posgp_aux = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] shapef_aux
		cdef np.ndarray[np.double_t,ndim=2] gradi_aux
		for igp in range(ngauss):
			for inod in range(self._nnod):
				posgp_aux[inod] = posgp[igp,inod]
			shapef_aux, gradi_aux = self.shape_func(posgp_aux)
			for inod in range(self._nnod):
				shapef[inod,igp]  = shapef_aux[inod]
				gradi[0,inod,igp] = gradi_aux[0,inod]
				gradi[1,inod,igp] = gradi_aux[1,inod]
				gradi[2,inod,igp] = gradi_aux[2,inod]
		self._shapef = shapef
		self._gradi  = gradi

	def __str__(TrilinearBrick self):
		s = 'Trilinear brick nnod=%d' % self._nnod
		return s

	@property
	def type(TrilinearBrick self):
		return 37

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)  
	def new(TrilinearBrick self,int ngauss=8):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return TrilinearBrick(self.nodes,ngauss)

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)                
	def shape_func(TrilinearBrick self, double[:] stz):
		'''
		Shape function and gradient for a set of 
		coordinates.
		'''
		cdef double sm, tm, zm, sq, tp, zp
		cdef np.ndarray[np.double_t,ndim=1] shapef = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] gradi  = np.zeros((3,self._nnod),dtype=np.double)
		sm = 0.5*(1. - stz[0])
		tm = 0.5*(1. - stz[1])
		zm = 0.5*(1. - stz[2])
		sq = 0.5*(1. + stz[0])
		tp = 0.5*(1. + stz[1])
		zp = 0.5*(1. + stz[2])
		# Define the shape function in local coordinates
		shapef[0] = sm*tm*zm
		shapef[1] = sq*tm*zm
		shapef[2] = sq*tp*zm
		shapef[3] = sm*tp*zm
		shapef[4] = sm*tm*zp
		shapef[5] = sq*tm*zp
		shapef[6] = sq*tp*zp
		shapef[7] = sm*tp*zp
		# Define the gradient in local coordinates
		gradi[0,0] = -0.5*tm*zm
		gradi[1,0] = -0.5*sm*zm
		gradi[2,0] = -0.5*sm*tm
		gradi[0,1] =  0.5*tm*zm
		gradi[1,1] = -0.5*sq*zm
		gradi[2,1] = -0.5*sq*tm
		gradi[0,2] =  0.5*tp*zm
		gradi[1,2] =  0.5*sq*zm
		gradi[2,2] = -0.5*sq*tp
		gradi[0,3] = -0.5*tp*zm
		gradi[1,3] =  0.5*sm*zm
		gradi[2,3] = -0.5*sm*tp
		gradi[0,4] = -0.5*tm*zp
		gradi[1,4] = -0.5*sm*zp
		gradi[2,4] =  0.5*sm*tm
		gradi[0,5] =  0.5*tm*zp
		gradi[1,5] = -0.5*sq*zp
		gradi[2,5] =  0.5*sq*tm
		gradi[0,6] =  0.5*tp*zp
		gradi[1,6] =  0.5*sq*zp
		gradi[2,6] =  0.5*sq*tp
		gradi[0,7] = -0.5*tp*zp
		gradi[1,7] =  0.5*sm*zp
		gradi[2,7] =  0.5*sm*tp
		return shapef, gradi

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)                
	def isinside(TrilinearBrick self, double[:] xyz, double[:,:] xyzel, double epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		cdef int ii
		cdef LinearTetrahedron tetra
		cdef np.ndarray[np.int32_t,ndim=1]  nodeList = np.zeros((4,),dtype=np.int32)
		cdef np.ndarray[np.double_t,ndim=2] xyzel1   = np.zeros((4,3),dtype=np.double)
		# Split the element into tetras
		nodeList[0] = self._nodeList[0]
		nodeList[1] = self._nodeList[1]
		nodeList[2] = self._nodeList[3]
		nodeList[3] = self._nodeList[4]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[0,ii]
			xyzel1[1,ii] = xyzel[1,ii]
			xyzel1[2,ii] = xyzel[3,ii]
			xyzel1[3,ii] = xyzel[4,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[5]
		nodeList[1] = self._nodeList[1]
		nodeList[2] = self._nodeList[4]
		nodeList[3] = self._nodeList[7]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[5,ii]
			xyzel1[1,ii] = xyzel[1,ii]
			xyzel1[2,ii] = xyzel[4,ii]
			xyzel1[3,ii] = xyzel[7,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[7]
		nodeList[1] = self._nodeList[4]
		nodeList[2] = self._nodeList[3]
		nodeList[3] = self._nodeList[1]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[7,ii]
			xyzel1[1,ii] = xyzel[4,ii]
			xyzel1[2,ii] = xyzel[3,ii]
			xyzel1[3,ii] = xyzel[1,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[2]
		nodeList[1] = self._nodeList[3]
		nodeList[2] = self._nodeList[1]
		nodeList[3] = self._nodeList[6]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[2,ii]
			xyzel1[1,ii] = xyzel[3,ii]
			xyzel1[2,ii] = xyzel[1,ii]
			xyzel1[3,ii] = xyzel[6,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[7]
		nodeList[1] = self._nodeList[3]
		nodeList[2] = self._nodeList[6]
		nodeList[3] = self._nodeList[5]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[7,ii]
			xyzel1[1,ii] = xyzel[3,ii]
			xyzel1[2,ii] = xyzel[6,ii]
			xyzel1[3,ii] = xyzel[5,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[5]
		nodeList[1] = self._nodeList[6]
		nodeList[2] = self._nodeList[1]
		nodeList[3] = self._nodeList[3]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[5,ii]
			xyzel1[1,ii] = xyzel[6,ii]
			xyzel1[2,ii] = xyzel[1,ii]
			xyzel1[3,ii] = xyzel[3,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		return False

cdef class TriQuadraticBrick(Element3D):
	'''
	TriQuadratic brick
	'''
	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check
	def __init__(TriQuadraticBrick self, int[:] nodeList, int ngauss=27):
		'''
		Define a triquadratic brick given a list of nodes and 
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:                  number of Gauss points
		'''
		if nodeList[26] == -1:
			raiseError('Invalid TriQuadratic Brick (%d)!'%len(nodeList))
		# Allocate memory by initializing the parent class
		super(TriQuadraticBrick,self).__init__(nodeList,ngauss)
		# Gauss points positions and weights
		cdef double[:]   weigp = self._weigp
		cdef double[:,:] posgp = self._posgp, posnod = self._posnod
		cdef int igauss, ilocs, jlocs, klocs, nlocs = int( ngauss**(1./3.) )
		cdef np.ndarray[np.double_t,ndim=1] posgl = np.zeros((nlocs,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] weigl = np.zeros((nlocs,),dtype=np.double)
		posnod[0 ,0] = -1. # s (left to right)
		posnod[0 ,1] = -1. # t (bottom to top)
		posnod[0 ,2] = -1. # z (back to front)
		posnod[1 ,0] =  1.
		posnod[1 ,1] = -1.
		posnod[1 ,2] = -1.
		posnod[2 ,0] =  1.
		posnod[2 ,1] =  1.
		posnod[2 ,2] = -1.
		posnod[3 ,0] = -1.
		posnod[3 ,1] =  1.
		posnod[3 ,2] = -1.
		posnod[4 ,0] = -1.
		posnod[4 ,1] = -1.
		posnod[4 ,2] =  1.
		posnod[5 ,0] =  1.
		posnod[5 ,1] = -1.
		posnod[5 ,2] =  1.
		posnod[6 ,0] =  1.
		posnod[6 ,1] =  1.
		posnod[6 ,2] =  1.
		posnod[7 ,0] = -1.
		posnod[7 ,1] =  1.
		posnod[7 ,2] =  1.
		# High-order nodes
		posnod[8 ,0] =  0. # s (left to right)
		posnod[8 ,1] = -1. # t (bottom to top)
		posnod[8 ,2] = -1. # z (back to front)
		posnod[9 ,0] =  1.
		posnod[9 ,1] =  0.
		posnod[9 ,2] = -1.
		posnod[10,0] =  0.
		posnod[10,1] =  1.
		posnod[10,2] = -1.
		posnod[11,0] = -1.
		posnod[11,1] =  0.
		posnod[11,2] = -1.
		posnod[12,0] = -1. # s (left to right)
		posnod[12,1] = -1. # t (bottom to top)
		posnod[12,2] =  0. # z (back to front)
		posnod[13,0] =  1.
		posnod[13,1] = -1.
		posnod[13,2] =  0.
		posnod[14,0] =  1.
		posnod[14,1] =  1.
		posnod[14,2] =  0.
		posnod[15,0] = -1.
		posnod[15,1] =  1.
		posnod[15,2] =  0.
		posnod[16,0] =  0. # s (left to right)
		posnod[16,1] = -1. # t (bottom to top)
		posnod[16,2] =  1. # z (back to front)
		posnod[17,0] =  1.
		posnod[17,1] =  0.
		posnod[17,2] =  1.
		posnod[18,0] =  0.
		posnod[18,1] =  1.
		posnod[18,2] =  1.
		posnod[19,0] = -1.
		posnod[19,1] =  0.
		posnod[19,2] =  1.
		posnod[20,0] =  0. # s (left to right)
		posnod[20,1] =  0. # t (bottom to top)
		posnod[20,2] = -1. # z (back to front)
		posnod[21,0] =  0.
		posnod[21,1] = -1.
		posnod[21,2] =  0.
		posnod[22,0] =  1.
		posnod[22,1] =  0.
		posnod[22,2] =  0.
		posnod[23,0] =  0.
		posnod[23,1] =  1.
		posnod[23,2] =  0.
		posnod[24,0] = -1.
		posnod[24,1] =  0.
		posnod[24,2] =  0.
		posnod[25,0] =  0.
		posnod[25,1] =  0.
		posnod[25,2] =  1.
		posnod[26,0] =  0.
		posnod[26,1] =  0.
		posnod[26,2] =  0.
		if nlocs == 1:
			posgl[0] = 0.
			weigl[0] = 2.
		elif nlocs == 2:
			posgl[0] = -0.577350269189626
			posgl[1] =  0.577350269189626
			weigl[0] = 1.
			weigl[1] = 1.
		elif nlocs == 3:
			posgl[0] = -0.774596669241483377035853079956
			posgl[1] =  0.0
			posgl[2] =  0.774596669241483377035853079956
			weigl[0] = 5./9.
			weigl[1] = 8./9.
			weigl[2] = 5./9.
		else:
			raiseError('Invalid number of Gauss points (%d)!'%ngauss)

		igauss = 0
		for ilocs in range(nlocs):
			for jlocs in range(nlocs):
				for klocs in range(nlocs):
					weigp[igauss] = weigl[ilocs]*weigl[jlocs]*weigl[klocs]
					posgp[igauss,0] = posgl[ilocs]
					posgp[igauss,1] = posgl[jlocs]
					posgp[igauss,2] = posgl[klocs]
					igauss += 1
		self._posnod = posnod
		self._posgp  = posgp
		self._weigp  = weigp
		# Compute shape function and derivatives
		cdef int igp, inod
		cdef double[:,:]   shapef = self._shapef
		cdef double[:,:,:] gradi  = self._gradi
		cdef np.ndarray[np.double_t,ndim=1] posgp_aux = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=1] shapef_aux
		cdef np.ndarray[np.double_t,ndim=2] gradi_aux
		for igp in range(ngauss):
			for inod in range(self._nnod):
				posgp_aux[inod] = posgp[igp,inod]
			shapef_aux, gradi_aux = self.shape_func(posgp_aux)
			for inod in range(self._nnod):
				shapef[inod,igp]  = shapef_aux[inod]
				gradi[0,inod,igp] = gradi_aux[0,inod]
				gradi[1,inod,igp] = gradi_aux[1,inod]
				gradi[2,inod,igp] = gradi_aux[2,inod]
		self._shapef = shapef
		self._gradi  = gradi

	def __str__(TriQuadraticBrick self):
		s = 'TriQuadratic brick nnod=%d' % self._nnod
		return s

	@property
	def type(TriQuadraticBrick self):
		return 39

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)  
	def new(TriQuadraticBrick self,int ngauss=27):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return TriQuadraticBrick(self.nodes,ngauss)

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)                
	def shape_func(TriQuadraticBrick self, double[:] stz):
		'''
		Shape function and gradient for a set of 
		coordinates.
		'''
		cdef double sl, tl, zl, sq, tp, zp
		cdef double s1, t1, z1
		cdef double s2, t2, z2
		cdef double s3, t3, z3
		cdef double s4, t4, z4
		cdef np.ndarray[np.double_t,ndim=1] shapef = np.zeros((self._nnod,),dtype=np.double)
		cdef np.ndarray[np.double_t,ndim=2] gradi  = np.zeros((3,self._nnod),dtype=np.double)
		sl=stz[0]*(stz[0]-1.)
		tl=stz[1]*(stz[1]-1.)
		zl=stz[2]*(stz[2]-1.)
		sq=stz[0]*(stz[0]+1.)
		tp=stz[1]*(stz[1]+1.)
		zp=stz[2]*(stz[2]+1.)
		s1= 2.*stz[0]-1.
		t1= 2.*stz[1]-1.
		z1= 2.*stz[2]-1.
		s2= 1.-stz[0]*stz[0]
		t2= 1.-stz[1]*stz[1]
		z2= 1.-stz[2]*stz[2]
		s3= 1.+2.*stz[0]
		t3= 1.+2.*stz[1]
		z3= 1.+2.*stz[2]
		s4=-2.*stz[0]
		t4=-2.*stz[1]
		z4=-2.*stz[2]

		shapef[ 0] = 0.125*sl*tl*zl
		shapef[ 1] = 0.125*sq*tl*zl
		shapef[ 2] = 0.125*sq*tp*zl
		shapef[ 3] = 0.125*sl*tp*zl
		shapef[ 4] = 0.125*sl*tl*zp
		shapef[ 5] = 0.125*sq*tl*zp
		shapef[ 6] = 0.125*sq*tp*zp
		shapef[ 7] = 0.125*sl*tp*zp
		shapef[ 8] = 0.25*s2*tl*zl
		shapef[ 9] = 0.25*sq*t2*zl
		shapef[10] = 0.25*s2*tp*zl
		shapef[11] = 0.25*sl*t2*zl
		shapef[12] = 0.25*sl*tl*z2
		shapef[13] = 0.25*sq*tl*z2
		shapef[14] = 0.25*sq*tp*z2
		shapef[15] = 0.25*sl*tp*z2
		shapef[16] = 0.25*s2*tl*zp
		shapef[17] = 0.25*sq*t2*zp
		shapef[18] = 0.25*s2*tp*zp
		shapef[19] = 0.25*sl*t2*zp
		shapef[20] = 0.5*s2*t2*zl
		shapef[21] = 0.5*s2*tl*z2
		shapef[22] = 0.5*sq*t2*z2
		shapef[23] = 0.5*s2*tp*z2
		shapef[24] = 0.5*sl*t2*z2
		shapef[25] = 0.5*s2*t2*zp
		shapef[26] = s2*t2*z2

		gradi[0, 0] = 0.125*s1*tl*zl
		gradi[1, 0] = 0.125*sl*t1*zl
		gradi[2, 0] = 0.125*sl*tl*z1
		gradi[0, 1] = 0.125*s3*tl*zl
		gradi[1, 1] = 0.125*sq*t1*zl
		gradi[2, 1] = 0.125*sq*tl*z1
		gradi[0, 2] = 0.125*s3*tp*zl
		gradi[1, 2] = 0.125*sq*t3*zl
		gradi[2, 2] = 0.125*sq*tp*z1
		gradi[0, 3] = 0.125*s1*tp*zl
		gradi[1, 3] = 0.125*sl*t3*zl
		gradi[2, 3] = 0.125*sl*tp*z1
		gradi[0, 4] = 0.125*s1*tl*zp
		gradi[1, 4] = 0.125*sl*t1*zp
		gradi[2, 4] = 0.125*sl*tl*z3
		gradi[0, 5] = 0.125*s3*tl*zp
		gradi[1, 5] = 0.125*sq*t1*zp
		gradi[2, 5] = 0.125*sq*tl*z3
		gradi[0, 6] = 0.125*s3*tp*zp
		gradi[1, 6] = 0.125*sq*t3*zp
		gradi[2, 6] = 0.125*sq*tp*z3
		gradi[0, 7] = 0.125*s1*tp*zp
		gradi[1, 7] = 0.125*sl*t3*zp
		gradi[2, 7] = 0.125*sl*tp*z3
		gradi[0, 8] = 0.25*s4*tl*zl
		gradi[1, 8] = 0.25*s2*t1*zl
		gradi[2, 8] = 0.25*s2*tl*z1
		gradi[0, 9] = 0.25*s3*t2*zl
		gradi[1, 9] = 0.25*sq*t4*zl
		gradi[2, 9] = 0.25*sq*t2*z1
		gradi[0,10] = 0.25*s4*tp*zl
		gradi[1,10] = 0.25*s2*t3*zl
		gradi[2,10] = 0.25*s2*tp*z1
		gradi[0,11] = 0.25*s1*t2*zl
		gradi[1,11] = 0.25*sl*t4*zl
		gradi[2,11] = 0.25*sl*t2*z1
		gradi[0,12] = 0.25*s1*tl*z2
		gradi[1,12] = 0.25*sl*t1*z2
		gradi[2,12] = 0.25*sl*tl*z4
		gradi[0,13] = 0.25*s3*tl*z2
		gradi[1,13] = 0.25*sq*t1*z2
		gradi[2,13] = 0.25*sq*tl*z4
		gradi[0,14] = 0.25*s3*tp*z2
		gradi[1,14] = 0.25*sq*t3*z2
		gradi[2,14] = 0.25*sq*tp*z4
		gradi[0,15] = 0.25*s1*tp*z2
		gradi[1,15] = 0.25*sl*t3*z2
		gradi[2,15] = 0.25*sl*tp*z4
		gradi[0,16] = 0.25*s4*tl*zp
		gradi[1,16] = 0.25*s2*t1*zp
		gradi[2,16] = 0.25*s2*tl*z3
		gradi[0,17] = 0.25*s3*t2*zp
		gradi[1,17] = 0.25*sq*t4*zp
		gradi[2,17] = 0.25*sq*t2*z3
		gradi[0,18] = 0.25*s4*tp*zp
		gradi[1,18] = 0.25*s2*t3*zp
		gradi[2,18] = 0.25*s2*tp*z3
		gradi[0,19] = 0.25*s1*t2*zp
		gradi[1,19] = 0.25*sl*t4*zp
		gradi[2,19] = 0.25*sl*t2*z3
		gradi[0,20] = 0.5*s4*t2*zl
		gradi[1,20] = 0.5*s2*t4*zl
		gradi[2,20] = 0.5*s2*t2*z1
		gradi[0,21] = 0.5*s4*tl*z2
		gradi[1,21] = 0.5*s2*t1*z2
		gradi[2,21] = 0.5*s2*tl*z4
		gradi[0,22] = 0.5*s3*t2*z2
		gradi[1,22] = 0.5*sq*t4*z2
		gradi[2,22] = 0.5*sq*t2*z4
		gradi[0,23] = 0.5*s4*tp*z2
		gradi[1,23] = 0.5*s2*t3*z2
		gradi[2,23] = 0.5*s2*tp*z4
		gradi[0,24] = 0.5*s1*t2*z2
		gradi[1,24] = 0.5*sl*t4*z2
		gradi[2,24] = 0.5*sl*t2*z4
		gradi[0,25] = 0.5*s4*t2*zp
		gradi[1,25] = 0.5*s2*t4*zp
		gradi[2,25] = 0.5*s2*t2*z3
		gradi[0,26] = s4*t2*z2
		gradi[1,26] = s2*t4*z2
		gradi[2,26] = s2*t2*z4
		return shapef, gradi

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)                
	def isinside(TriQuadraticBrick self, double[:] xyz, double[:,:] xyzel, double epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		cdef int ii
		cdef LinearTetrahedron tetra
		cdef np.ndarray[np.int32_t,ndim=1]  nodeList = np.zeros((4,),dtype=np.int32)
		cdef np.ndarray[np.double_t,ndim=2] xyzel1   = np.zeros((4,3),dtype=np.double)
		# Split the element into tetras
		nodeList[0] = self._nodeList[0]
		nodeList[1] = self._nodeList[1]
		nodeList[2] = self._nodeList[3]
		nodeList[3] = self._nodeList[4]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[0,ii]
			xyzel1[1,ii] = xyzel[1,ii]
			xyzel1[2,ii] = xyzel[3,ii]
			xyzel1[3,ii] = xyzel[4,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[5]
		nodeList[1] = self._nodeList[1]
		nodeList[2] = self._nodeList[4]
		nodeList[3] = self._nodeList[7]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[5,ii]
			xyzel1[1,ii] = xyzel[1,ii]
			xyzel1[2,ii] = xyzel[4,ii]
			xyzel1[3,ii] = xyzel[7,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[7]
		nodeList[1] = self._nodeList[4]
		nodeList[2] = self._nodeList[3]
		nodeList[3] = self._nodeList[1]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[7,ii]
			xyzel1[1,ii] = xyzel[4,ii]
			xyzel1[2,ii] = xyzel[3,ii]
			xyzel1[3,ii] = xyzel[1,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[2]
		nodeList[1] = self._nodeList[3]
		nodeList[2] = self._nodeList[1]
		nodeList[3] = self._nodeList[6]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[2,ii]
			xyzel1[1,ii] = xyzel[3,ii]
			xyzel1[2,ii] = xyzel[1,ii]
			xyzel1[3,ii] = xyzel[6,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[7]
		nodeList[1] = self._nodeList[3]
		nodeList[2] = self._nodeList[6]
		nodeList[3] = self._nodeList[5]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[7,ii]
			xyzel1[1,ii] = xyzel[3,ii]
			xyzel1[2,ii] = xyzel[6,ii]
			xyzel1[3,ii] = xyzel[5,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		nodeList[0] = self._nodeList[5]
		nodeList[1] = self._nodeList[6]
		nodeList[2] = self._nodeList[1]
		nodeList[3] = self._nodeList[3]
		for ii in range(3):
			xyzel1[0,ii] = xyzel[5,ii]
			xyzel1[1,ii] = xyzel[6,ii]
			xyzel1[2,ii] = xyzel[1,ii]
			xyzel1[3,ii] = xyzel[3,ii]
		tetra = LinearTetrahedron(nodeList,1)
		if tetra.isinside(xyz,xyzel1,epsi): return True
		return False

cdef class pOrderHexahedron(Element3D):
	'''
	SEM pOrder Hexahedron: gauss points = nodes. GLL quadrature.
	'''
	cdef int _porder
	cdef double[:] _pnodes
	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check       
	def __init__(pOrderHexahedron self, int[:] nodeList, int ngauss, double[:] xi, double[:,:] posnod, double[:] weigp, double[:,:] shapef, double[:,:,:] gradi):
		# check number of Gauss points
		self._porder = np.sqrt(ngauss) - 1
		if not len(nodeList) == ngauss:
			raiseError('Invalid pOrder Quadrangle! Number of nodes (%d) is different to pOrder (%d)' % (len(nodeList),self._porder))
		# Allocate memory by initializing the parent class
		super(pOrderHexahedron, self).__init__(nodeList, ngauss, SEM=True)
		# Nodes/Gauss points positions and weights
		self._pnodes = xi
		self._posnod = posnod
		self._weigp  = weigp
		# Shape function and derivatives in the Gauss Points
		self._shapef = shapef
		self._gradi  = gradi
		
	@property
	def type(pOrderHexahedron self):
		return 40

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check 
	@cr('pOrderHexahedron.shape_func')
	def shape_func(pOrderHexahedron self, double[:] stz):
		'''
		Shape function and gradient for a set of coordinates.
		'''
		cdef int nnod   = self._nnod
		cdef int npoint = self._pnodes.shape[0]
		cdef int c = 0
		cdef int kk, ii, jj
		cdef double lag_xi0, lag_xi1, lag_xi2
		cdef double dlag_xi0, dlag_xi1, dlag_xi2
		cdef np.ndarray[np.double_t, ndim=2] gradi    = np.zeros((3, nnod), dtype=np.double)
		cdef np.ndarray[np.double_t, ndim=1] shapef   = np.zeros((nnod), dtype=np.double)
		cdef np.ndarray[np.double_t, ndim=1] points   = np.zeros((npoint), dtype=np.double)
		for ii in range(npoint):
			points[ii] = self._pnodes[ii]
		for kk in range(npoint):
			lag_xi2   = lagrange(stz[2], kk, points)
			dlag_xi2  = dlagrange(stz[2], kk, points)
			for ii in range(npoint):
				lag_xi0  = lagrange(stz[0], ii, points)
				dlag_xi0 = dlagrange(stz[0], ii, points)
				for jj in range(npoint):
					lag_xi1    = lagrange(stz[1], jj, points)
					dlag_xi1   = dlagrange(stz[1], jj, points)
					shapef[c]  = lag_xi0*lag_xi1*lag_xi2
					gradi[0,c] = dlag_xi0*lag_xi1*lag_xi2
					gradi[1,c] = lag_xi0*dlag_xi1*lag_xi2
					gradi[2,c] = lag_xi0*lag_xi1*dlag_xi2
					c = c + 1 
		
		return shapef, gradi
	
	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	@cython.cdivision(True)    # turn off zero division check 
	@cr('pOrderHexahedron.isinside')
	def isinside(pOrderHexahedron self, double[:] xyz, double[:,:] xyzel, double epsi=1e-10):
		'''
		Find if a point is inside an element.
		'''
		cdef int max_ite  = 50
		cdef double alpha = 1
		cdef double t1, t2, t3, detJ
		cdef int ite, ii, jj, ipoint
		cdef int nnod = self._nnod
		cdef bint conv    = False
		cdef np.ndarray[np.double_t, ndim=2] J      = np.zeros((3, 3), dtype=np.double)
		cdef np.ndarray[np.double_t, ndim=2] Jinv   = np.zeros((3, 3), dtype=np.double)
		cdef np.ndarray[np.double_t, ndim=2] gradi  = np.zeros((3, nnod), dtype=np.double)
		cdef np.ndarray[np.double_t, ndim=1] shapef = np.zeros((nnod), dtype=np.double)
		cdef np.ndarray[np.double_t, ndim=2] xyzel1 = np.zeros((nnod,3), dtype=np.double)
		cdef np.ndarray[np.double_t, ndim=1] X      = np.zeros((3), dtype=np.double)
		cdef np.ndarray[np.double_t, ndim=1] Nr     = np.zeros((3), dtype=np.double)
		cdef np.ndarray[np.double_t, ndim=1] f      = np.zeros((3), dtype=np.double)
		for ii in range(3):
			for jj in range(nnod):
				xyzel1[jj,ii] = xyzel[jj,ii]
		for ite in range(max_ite):
			for ii in range(3):
				f[ii] = xyz[ii]
				for jj in range(3):
					J[ii,jj] = 0.0
			shapef, gradi = self.shape_func(X)
			for ipoint in range(nnod):
				f[0]   = f[0] - shapef[ipoint]*xyzel1[ipoint,0]
				f[1]   = f[1] - shapef[ipoint]*xyzel1[ipoint,1]
				f[2]   = f[2] - shapef[ipoint]*xyzel1[ipoint,2]
				J[0,0] = J[0,0] - gradi[0,ipoint]*xyzel1[ipoint,0]
				J[0,1] = J[0,1] - gradi[1,ipoint]*xyzel1[ipoint,0]
				J[0,2] = J[0,2] - gradi[2,ipoint]*xyzel1[ipoint,0]
				J[1,0] = J[1,0] - gradi[0,ipoint]*xyzel1[ipoint,1]
				J[1,1] = J[1,1] - gradi[1,ipoint]*xyzel1[ipoint,1]
				J[1,2] = J[1,2] - gradi[2,ipoint]*xyzel1[ipoint,1]
				J[2,0] = J[2,0] - gradi[0,ipoint]*xyzel1[ipoint,2]
				J[2,1] = J[2,1] - gradi[1,ipoint]*xyzel1[ipoint,2]
				J[2,2] = J[2,2] - gradi[2,ipoint]*xyzel1[ipoint,2]
			# Determinant 
			t1   =  J[1,1]*J[2,2] - J[2,1]*J[1,2]
			t2   = -J[1,0]*J[2,2] + J[2,0]*J[1,2]
			t3   =  J[1,0]*J[2,1] - J[2,0]*J[1,1]
			detJ =  J[0,0]*t1 + J[0,1]*t2 + J[0,2]*t3
			# Inverse of Jacobian
			Jinv[0,0] = t1/detJ
			Jinv[1,0] = t2/detJ
			Jinv[2,0] = t3/detJ
			Jinv[1,1] = ( J[0,0]*J[2,2] - J[2,0]*J[0,2])/detJ
			Jinv[2,1] = (-J[0,0]*J[2,1] + J[0,1]*J[2,0])/detJ
			Jinv[2,2] = ( J[0,0]*J[1,1] - J[1,0]*J[0,1])/detJ
			Jinv[0,1] = (-J[0,1]*J[2,2] + J[2,1]*J[0,2])/detJ
			Jinv[0,2] = ( J[0,1]*J[1,2] - J[1,1]*J[0,2])/detJ
			Jinv[1,2] = (-J[0,0]*J[1,2] + J[1,0]*J[0,2])/detJ
			# Newton-Rapson
			#Nr    = np.matmul(Jinv, f)
			Nr[0] = Jinv[0,0]*f[0] + Jinv[0,1]*f[1] + Jinv[0,2]*f[2]
			Nr[1] = Jinv[1,0]*f[0] + Jinv[1,1]*f[1] + Jinv[1,2]*f[2]
			Nr[2] = Jinv[2,0]*f[0] + Jinv[2,1]*f[1] + Jinv[2,2]*f[2]
			X[0] -= alpha*Nr[0]
			X[1] -= alpha*Nr[1]
			X[2] -= alpha*Nr[2]
			if max(abs(f)) > 1000:
				break
			if max(abs(f)) < epsi:
				conv = True
				break
		if conv and X[0] <= 1 + epsi and X[0] >= -1 -epsi and X[1] <= 1 + epsi and X[1] >= -1 - epsi and X[2] <= 1 + epsi and X[2] >= -1 -epsi:
			return True
		else:
			return False


ALYA_ELEMDICT = {
	2  : {'class':Bar,              'nnod':2},  # BAR02
#	4  : {3rd order line element}, # BAR04
	10 : {'class':LinearTriangle,   'nnod':3},  # TRI03
	12 : {'class':LinearQuadrangle, 'nnod':4},  # QUA04
	15 : {'class':pOrderQuadrangle, 'nnod':-1}, # QUAHO
	30 : {'class':LinearTetrahedron,'nnod':4},  # TET04
	32 : {'class':LinearPyramid,    'nnod':5},  # PYR05
	34 : {'class':LinearPrism,      'nnod':6},  # PEN06
	37 : {'class':TrilinearBrick,   'nnod':8},  # HEX08
	39 : {'class':TriQuadraticBrick,'nnod':27}, # HEX27
	40 : {'class':pOrderHexahedron, 'nnod':-1}, # HEXHO
}

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def defineHighOrderElement(int porder, int ndime):
	cdef int npoint = porder + 1
	cdef int npoin2 = npoint*npoint
	cdef int ngauss = <int>pow(npoint, ndime)
	cdef int c = 0, igp, off, off2, idime, ii, multiplier
	cdef np.ndarray[np.double_t, ndim=1] xi     = np.zeros((npoint), dtype=np.double)
	cdef np.ndarray[np.double_t, ndim=1] wi     = np.zeros((npoint), dtype=np.double)
	cdef np.ndarray[np.double_t, ndim=2] posnod = np.zeros((ngauss, ndime), dtype=np.double)
	cdef np.ndarray[np.double_t, ndim=1] weigp  = np.zeros((ngauss,), dtype=np.double)
	cdef np.ndarray[np.double_t, ndim=2] shapef = np.eye(ngauss, dtype=np.double)
	cdef np.ndarray[np.double_t, ndim=3] gradi  = np.zeros((ndime, ngauss, ngauss), dtype=np.double)
	cdef np.ndarray[np.double_t, ndim=1] dlag  = np.zeros(ndime, dtype=np.double)
	cdef np.ndarray[np.int32_t, ndim=1] offset = np.zeros((ndime,), dtype=np.int32)

	# Use quadrature_GaussLobatto to get xi, wi
	xi, wi = quadrature_GaussLobatto(npoint)

	if ndime == 3:
		for k in range(npoint):
			for i in range(npoint):
				for j in range(npoint):
					posnod[c,:] = np.array([xi[i],xi[j],xi[k]], np.double)
					weigp[c]    = np.array(wi[i]*wi[j]*wi[k], np.double)
					c += 1
	if ndime == 2:
		for i in range(npoint):
			for j in range(npoint):
				posnod[c,:] = np.array([xi[i],xi[j]], np.double)
				weigp[c]    = np.array(wi[i]*wi[j], np.double)
				c += 1

	for igp in range(ngauss):
		off  = <int>floor(igp / npoint)
		off2 = <int>floor(igp / npoin2)
		for ii in range(npoint):
			offset[0] = igp - npoint * (off - ii)
			offset[1] = off * npoint + ii
			if ndime == 3:
				offset[0] += off2 * npoin2
				offset[2] = igp - npoin2 * (off2 - ii)
			for idime in range(ndime):
				dlag[idime] = dlagrange(posnod[igp,idime], ii, xi)
				gradi[idime, offset[idime], igp] = dlag[idime]

	return ngauss, xi, posnod, weigp, shapef, gradi

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)                
def createElementByType(int ltype, int [:] nodeList, int ngauss=-1, double[:] xi=None, double[:,:] posnod=None, double[:] weigp=None, double[:,:] shapef=None, double[:,:,:] gradi=None):
	'''
	Use the data in LTYPE to create an element according
	to its type as defined in Alya defmod/def_elmtyp.f90.

	IN:
		> ltype:         type of element
		> nodeList:  array with the number of nodes of the element
		> ngauss:        number of gauss points, optional
	'''
	if not ltype in ALYA_ELEMDICT.keys(): raiseError('Element type %d not implemented!' % ltype)
	cdef int nnod
	nnod = ngauss if ALYA_ELEMDICT[ltype]['nnod'] < 0 else ALYA_ELEMDICT[ltype]['nnod']
	# Return element and node cut according to the dict
	cdef object elclass = ALYA_ELEMDICT[ltype]['class']
	cdef int    nList   = nnod, ilist
	# Alya must be consistent with the nodeList if the mesh
	# contains more than one element, hence it puts -1 on the
	# elements that don't have a node in that position.
	# We will simply filter them
	cdef np.ndarray[np.int32_t,ndim=1] nodeList_aux = -np.ones((nList,),np.int32)
	for ilist in range(nList):
		nodeList_aux[ilist] = nodeList[ilist]
	if ALYA_ELEMDICT[ltype]['nnod'] < 0:
		return elclass(nodeList_aux,ngauss,xi,posnod,weigp,shapef,gradi)
	else:
		return elclass(nodeList_aux) if ngauss < 0 else elclass(nodeList_aux,ngauss)