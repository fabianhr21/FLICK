#!/usr/bin/env python
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

import numpy as np

from ..utils.common import raiseError
from .quadratures   import lagrange, dlagrange, quadrature_GaussLobatto


class Element1D(object):
	'''
	Basic FEM parent class that implements the
	basic operations for a 1D element.
	'''
	def __init__(self, nodeList, ngauss, dtype=np.double):
		'''
		Define an element given a list of nodes and
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:          number of Gauss points
		'''
		self._ngp      = ngauss
		self._nodeList = np.array(nodeList).copy()
		self._nnod     = len(nodeList)
		self._dtype    = dtype

		# Nodes position in local coordinates
		self._posnod = np.zeros((self.nnod, 1), dtype=self.dtype)
		# Gauss points position in local coordinates
		self._posgp  = np.zeros((self.ngauss, 1), dtype=self.dtype)
		# Gauss points weights
		self._weigp  = np.zeros((self.ngauss,), dtype=self.dtype)
		self._shapef = np.zeros((self.nnod, self.ngauss), dtype=self.dtype) # Shape function
		# Gradient function (in local coordinates)
		self._gradi = np.zeros((1, self.nnod, self.ngauss), dtype=self.dtype)

		# Normal of the element (its norm is the surface of the element)
		self._normal = np.zeros((1, 3), dtype=self.dtype)

	def __del__(self):
		del self._ngp, self._nodeList, self._nnod, self.dtype
		# Nodes position in local coordinates
		del self._posnod
		# Gauss points position in local coordinates
		del self._posgp
		# Gauss points weights
		del self._weigp, self._shapef
		# Gradient function (in local coordinates)
		del self._gradi
		# Normal of the element (its norm is the surface of the element)
		del self._normal

	def __str__(self):
		s = 'Element 1D nnod=%d\n' % self._nnod
		return s

	def __len__(self):
		return self._nnod

	def __eq__(self, other):
		'''
		Element1 == Element2
		'''
		if other is None:
			return type(self) == None
		else:
			return self._ngp == other._ngp and self._nnod == other._nnod

	def centroid(self, xyel):
		'''
		Element centroid
		'''
		return np.sum(xyel, axis=0)/self._nnod

	def normal(self, xyel):
		'''
		Element normal, expects 2D coordinates.
		'''
		cen = self.centroid(xyel)
		u = np.array([0., 0., 0.], self.dtype) 
		v = np.array([0., 0., 1.], self.dtype) 
		for inod in range(self._nnod):  # This works because of python
			U[:1] = xyel[inod] - cen
			self._normal -= 0.5*np.cross(u, v)
		return self._normal[:1]

	def transform1D(self, xy, xyel):
		'''
		TRAnsforms the coordinate system of a surface element in a 2D mesh
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
		# Compute RST axis
		r = self.normal(xyel)
		r /= np.sqrt(np.sum(r*r))
		s = xyel[1, :] - xyel[0, :]
		s /= np.sqrt(np.sum(s*s))
		# Coordinate change matrix
		M = np.zeros((2, 2))
		M[0, :] = s
		M[1, :] = r
		# Project
		if len(xy.shape) > 1:
			x = np.zeros((xy.shape[0],), dtype=self.dtype)
			for inod in range(xy.shape[0]):
				x[inod] = np.matmul(M, xy[inod, :])[0:1]
		else:
			x = np.matmul(M, xy)[0:1]
		return x

	def nodes2gp(self,elfield):
		'''
		Transforms an entry field on the nodes
		(or their position) to the Gauss point 
		equivalent.
		'''
		return np.matmul(self._shapef.T,elfield)

	def derivative(self, xyel):
		'''
		Derivative of the element:
		IN:
			> xyel(nnod,1):         position of the nodes in cartesian coordinates
		OUT:
			> deri(1,nnod,ngauss):  derivatives per each gauss point
			> vol(ngauss):          volume per each gauss point
			> mle(nnod):            lumped mass matrix per each node
		'''
		deri = np.zeros((1, self.nnod, self.ngauss), dtype=self.dtype)
		vol  = np.zeros((self.ngauss,), dtype=self.dtype)
		# Ensure dealing with a 1D array
		xel = xyel if xyel.shape[1] == 1 else self.transform1D(xyel, xyel)
		# Derivative computation
		for igp in range(self.ngauss):
			J = np.matmul(self._gradi[:, :, igp], xel[:self.nnod])  # Jacobian
			# Determinant of the Jacobian
			detJ = np.linalg.det(J)
			# Inverse of the Jacobian
			Jinv = np.linalg.inv(J)

			# Derivatives, global shape gradients
			self._deri[:, :, igp] = np.matmul(Jinv, self._gradi[:, :, igp])
			self._vol[igp] = detJ*self._weigp[igp]                # Element volume

		return deri, vol

	def integrative(self, xyel):
		'''
		Integral of the element:
		IN:
			> xyel(nnod,2):       position of the nodes in cartesian coordinates
		OUT:
			> integ(nnod,ngauss): integral per each gauss point
		'''
		# Ensure dealing with a 1D array
		xel = xyel if xyel.shape[1] == 1 else self.transform1D(xyel, xyel)
		# Integral computation
		for igp in range(self.ngauss):
			J = np.matmul(self._gradi[:, :, igp], xel[:self.nnod])  # Jacobian
			# Determinant of the Jacobian
			detJ = np.linalg.det(J)
			self._integ[:, igp] = self._weigp[igp]*self._shapef[:, igp]*detJ
		
		return self._integ

	def consistent(self,xyel):
		'''
		Consistent mass matrix of the element:
		IN:
			> xyel(nnod,2):   position of the nodes in cartesian coordinates
		OUT:
			> mle(nnod,nnod): consistent mass matrix over the Gauss points	
		'''
		mle = np.zeros((self.nnod,self.nnod),self.dtype)
		# Ensure dealing with a 1D array
		xel = xyel if xyel.shape[1] == 1 else self.transform1D(xyel, xyel)
		# Computation
		for igp in range(self.ngauss):
			J = np.matmul(self._gradi[:, :, igp], xel[:self.nnod])  # Jacobian
			# Determinant of the Jacobian
			detJ = np.linalg.det(J)
			# Assemble mass matrix
			mle += self._weigp[igp]*detJ*np.outer(self._shapef[:, igp],self._shapef[:, igp])
		return mle		

	def find_stz(self, xy, xyel, max_iter=20, tol=1e-10):
		'''
		Find a position of the point in xyz coordinates
		in element local coordinates stz:
		IN:
			> xyz(1,3):      position of the point
			> xyzel(nnod,3): position of the element nodes
		OUT:
			> stz(1,):       position of the point in local coordinates
		'''
		# Ensure dealing with a 2D array
		x = xy if xyel.shape[1] == 1 else self.transform1D(xy, xyel)
		xel = xyel if xyel.shape[1] == 1 else self.transform1D(xyel, xyel)
		# Initial guess
		stz = self._posgp[0, :]
		shapef = self._shapef[:, 0]
		gradi = self._gradi[:, :, 0]
		# Compute residual
		f = xy - np.dot(shapef, xyel)
		r = np.sqrt(np.dot(f, f))
		# Newton-Raphson method
		for ii in range(max_iter):
			T = -np.dot(gradi, xyel)
			Tinv = np.linalg.inv(T).T
			delta = -np.dot(Tinv, f)
			# New guess
			stz += delta
			shapef, gradi = self.shape_func(stz)
			# Compute new residual
			f = xy - np.dot(shapef, xyel)
			r = np.sqrt(np.dot(f, f))
			# Exit criteria
			if r < tol: break
		return stz

	def interpolate(self, stz, elfield):
		'''
		Interpolate a variable on a point inside the element:
		IN:
			> stz(1):             position of the point in local coordinates
			> elfield(nnod,ndim): variable to be interpolated
		OUT:
			> out(1,ndim):        interpolated variable at stz
		'''
		# Recover the shape function at the point stz
		shapef, _ = self.shape_func(stz)
		# Allocate output array
		out = np.zeros((1, elfield.shape[1]) if len(elfield.shape) > 1 else (1,), dtype=np.double)
		# Compute output array
		if len(elfield.shape) > 1:
			# Vectorial array
			for ii in range(elfield.shape[1]):
				out[0, ii] = np.dot(shapef, elfield[:, ii])
		else:
			# Scalar array
			out = np.dot(shapef, elfield)
		# Return output
		return out

	@property
	def nnod(self):
		return self._nnod

	@property
	def ngauss(self):
		return self._ngp

	@property
	def ndim(self):
		return 1

	@property
	def dtype(self):
		return self._dtype

	@property
	def nodes(self):
		return self._nodeList

	@property
	def posnod(self):
		return self._posnod

	@property
	def posgp(self):
		return self._posgp

	@property
	def shape(self):
		return self._shapef


class Element2D(object):
	'''
	Basic FEM parent class that implements the
	basic operations for a 2D element.
	'''
	def __init__(self, nodeList, ngauss, SEM=False, dtype=np.double):
		'''
		Define an element given a list of nodes and
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:          number of Gauss points
		'''
		self._ngp      = ngauss
		self._nodeList = np.array(nodeList).copy()
		self._nnod     = len(nodeList)
		self._dtype    = dtype
		self._sem      = SEM

		if not self._sem:
			# Gauss points position in local coordinates
			self._posgp = np.zeros((self.ngauss, 2), dtype=self.dtype) #Not needed
			# Nodes position in local coordinates
			self._posnod = np.zeros((self.nnod, 2), dtype=self.dtype) # Allocated outside the element
			# Gauss points weights
			self._weigp = np.zeros((self.ngauss,), dtype=self.dtype)
			# Shape function
			self._shapef = np.zeros((self.nnod, self.ngauss),dtype=self.dtype)   
			# Gradient function (in local coordinates)
			self._gradi = np.zeros((2, self.nnod, self.ngauss), dtype=self.dtype)
		# Normal of the element (its norm is the surface of the element)
		self._normal = np.zeros((1, 3), dtype=np.double)

	def __del__(self):
		del self._ngp, self._nodeList, self._nnod, self._dtype
		# Nodes position in local coordinates
		del self._posnod
		# Gauss points position in local coordinates
		if not self._sem:
			del self._posgp
		# Gauss points weights
		del self._weigp, self._shapef
		# Gradient function (in local coordinates)
		del self._gradi
		# Normal of the element (its norm is the surface of the element)
		del self._normal

	def __str__(self):
		s = 'Element 2D nnod=%d\n' % self._nnod
		return s

	def __len__(self):
		return self._nnod

	def __eq__(self, other):
		'''
		Element1 == Element2
		'''
		if other is None:
			return type(self) == None
		else:
			return self._ngp == other._ngp and self._nnod == other._nnod

	def centroid(self, xyzel):
		'''
		Element centroid
		'''
		return np.sum(xyzel, axis=0)/self._nnod

	def normal(self, xyzel):
		'''
		Element normal, expects 3D coordinates.
		'''
		gradi = self._gradi
		gradi_xi = np.sum(gradi[0][:, :, np.newaxis] * xyzel[:, np.newaxis, :], axis=0)
		gradi_et = np.sum(gradi[1][:, :, np.newaxis] * xyzel[:, np.newaxis, :], axis=0)
		J = np.cross(gradi_et, gradi_xi)
		normal = J / np.linalg.norm(J, axis=1)[:, None]
		return normal

	def transform2D(self, xyz, xyzel):
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
		u = xyzel[1, :] - xyzel[0, :]
		# Compute RST axis
		r = self.normal(xyzel)
		r /= np.sqrt(np.sum(r*r))
		s = u/np.sqrt(np.sum(u*u))
		t = np.cross(r, s)
		t /= np.sqrt(np.sum(t*t))
		# Coordinate change matrix
		M = np.zeros((3, 3))  # [s, t, r]
		M[0, :] = s
		M[1, :] = t
		M[2, :] = r
		# Project
		if len(xyz.shape) > 1:
			xy = np.zeros((xyz.shape[0], 2), dtype=self.dtype)
			for inod in range(xyz.shape[0]):
				xy[inod, :] = np.matmul(M, xyz[inod, :])[0:2]
		else:
			xy = np.matmul(M, xyz)[0:2]
		return xy

	def nodes2gp(self,elfield):
		'''
		Transforms an entry field on the nodes
		(or their position) to the Gauss point 
		equivalent.
		'''
		return np.matmul(self._shapef.T,elfield)

	def derivative(self, xyzel):
		'''
		Derivative of the element: #TODO: PARTICULARIZE FOR X,Y,Z -> XI, ETA
		IN:
			> xyzel(nnod,2):        position of the nodes in cartesian coordinates
		OUT:
			> deri(2,nnod,ngauss):  derivatives per each gauss point
			> vol(ngauss):          volume per each gauss point
		'''
		deri = np.zeros((2, self.nnod, self.ngauss), dtype=self.dtype)
		vol  = np.zeros((self.ngauss,), dtype=self.dtype)
		# Ensure dealing with a 2D array
		#xyel = xyzel if xyzel.shape[1] == 2 else self.transform2D(xyzel, xyzel)
		# Derivative computation
		J = np.zeros((2,2),dtype=np.double)
		for igp in range(self.ngauss):
			J = np.matmul(self._gradi[:, :, igp], xyzel)  # Jacobian
			# Determinant of the Jacobian
			detJ = np.linalg.det(J)
			# Inverse of the Jacobian
			Jinv = np.linalg.inv(J)
			# Derivatives, global shape gradients
			deri[:, :, igp] = np.matmul(Jinv, self._gradi[:, :, igp])
			vol[igp]        = detJ*self._weigp[igp] # Element volume

		return deri, vol

	def integrative(self, xyzel):
		'''
		Integral of the element:
		IN:
			> xyzel(nnod,):       position of the nodes in cartesian coordinates
		OUT:
			> integ(nnod,ngauss): integral per each gauss point
		'''
		# Integral computation
		integ = np.zeros((self.nnod, self.ngauss),dtype=np.double)
		gradi_xi = np.zeros((1,3))
		gradi_et = np.zeros((1,3))
		for igp in range(self.ngauss):
			# Compute Jacobian
			Ji = np.matmul(self._gradi[:,:,igp],xyzel)
			gradi_xi = Ji[0,:]
			gradi_et = Ji[1,:]
			J = np.cross(gradi_xi,gradi_et)
			detJ = np.linalg.norm(J)
			integ[:, igp] = self._weigp[igp]*self._shapef[:, igp]*detJ
		return integ


	def consistent(self,xyzel):
		'''
		Consistent mass matrix of the element:
		IN:
			> xyel(nnod,2):   position of the nodes in cartesian coordinates
		OUT:
			> mle(nnod,nnod): consistent mass matrix over the Gauss points	
		'''
		mle = np.zeros((self.nnod,self.nnod),self.dtype)
		# Ensure dealing with a 2D array
		xyel = xyzel if xyzel.shape[1] == 2 else self.transform2D(xyzel, xyzel)
		# Computation
		for igp in range(self.ngauss):
			J = np.matmul(self._gradi[:, :, igp], xyel[:self.nnod])  # Jacobian
			# Determinant of the Jacobian
			detJ = np.linalg.det(J)
			# Assemble mass matrix
			mle += self._weigp[igp]*detJ*np.outer(self._shapef[:, igp],self._shapef[:, igp])
		return mle

	def find_stz(self, xyz, xyzel, max_iter=20, tol=1e-10):
		'''
		Find a position of the point in xyz coordinates
		in element local coordinates stz:
		IN:
			> xyz(1,3):      position of the point
			> xyzel(nnod,3): position of the element nodes
		OUT:
			> stz(2,):       position of the point in local coordinates
		'''
		# Ensure dealing with a 2D array
		xy   = xyz if xyzel.shape[1] == 2 else self.transform2D(xyz, xyzel)
		xyel = xyzel if xyzel.shape[1] == 2 else self.transform2D(xyzel, xyzel)
		# Initial guess
		stz = self._posgp[0, :]
		shapef = self._shapef[:, 0]
		gradi = self._gradi[:, :, 0]
		# Compute residual
		f = xy - np.dot(shapef, xyel)
		r = np.sqrt(np.dot(f, f))
		# Newton-Raphson method
		for ii in range(max_iter):
			T = -np.dot(gradi, xyel)
			Tinv = np.linalg.inv(T).T
			delta = -np.dot(Tinv, f)
			# New guess
			stz += delta
			shapef, gradi = self.shape_func(stz)
			# Compute new residual
			f = xy - np.dot(shapef, xyel)
			r = np.sqrt(np.dot(f, f))
			# Exit criteria
			if r < tol: break
		return stz

	def interpolate(self, stz, elfield):
		'''
		Interpolate a variable on a point inside the element:
		IN:
			> stz(2):             position of the point in local coordinates
			> elfield(nnod,ndim): variable to be interpolated
		OUT:
			> out(1,ndim):        interpolated variable at stz
		'''
		# Recover the shape function at the point stz
		shapef, _ = self.shape_func(stz)
		# Allocate output array
		out = np.zeros((1, elfield.shape[1]) if len(elfield.shape) > 1 else (1,), dtype=self.dtype)
		# Compute output array
		if len(elfield.shape) > 1:
			# Vectorial array
			for ii in range(elfield.shape[1]):
				out[0, ii] = np.dot(shapef, elfield[:, ii])
		else:
			# Scalar array
			out = np.dot(shapef, elfield)
		# Return output
		return out

	@property
	def nnod(self):
		return self._nnod

	@property
	def ngauss(self):
		return self._ngp

	@property
	def ndim(self):
		return 2

	@property
	def dtype(self):
		return self._dtype

	@property
	def nodes(self):
		return self._nodeList

	@property
	def posnod(self):
		return self._posnod

	@property
	def posgp(self):
		return self._posgp

	@property
	def shape(self):
		return self._shapef


class Element3D(object):
	'''
	Basic FEM parent class that implements the
	basic operations for a 3D element.
	'''
	def __init__(self, nodeList, ngauss, SEM=False, dtype=np.double):
		'''
		Define an element given a list of nodes and
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:          number of Gauss points
		'''
		self._ngp      = ngauss
		self._nodeList = np.array(nodeList).copy()
		self._nnod     = len(nodeList)
		self._dtype    = dtype 
		self._sem      = SEM
		if not self._sem:
			# Nodes position in local coordinates
			self._posnod = np.zeros((self.nnod, 3), dtype=self.dtype)
		# Gauss points position in local coordinates
		self._posgp = np.zeros((self.ngauss, 3), dtype=self.dtype)
		# Gauss points weights
		self._weigp = np.zeros((self.ngauss,), dtype=self.dtype)
		self._shapef = np.zeros((self.nnod, self.ngauss), dtype=self.dtype)   # Shape function
		# Gradient function (in local coordinates)
		self._gradi = np.zeros((3, self.nnod, self.ngauss), dtype=self.dtype)

	def __del__(self):
		del self._ngp, self._nodeList, self._nnod, self._dtype
		# Gauss points position in local coordinates
		del self._posgp
		# Nodes points position in local coordinates
		if not self._sem:
			del self._posnod
		# Gauss points weights
		del self._weigp, self._shapef
		# Gradient function (in local coordinates)
		del self._gradi

	def __str__(self):
		s = 'Element 3D nnod=%d\n' % self._nnod
		return s

	def __len__(self):
		return self._nnod

	def __eq__(self, other):
		'''
		Element1 == Element2
		'''
		if other is None:
			return type(self) == None
		else:
			return self._ngp == other._ngp and self._nnod == other._nnod

	def centroid(self, xyzel):
		'''
		Element centroid
		'''
		return np.sum(xyzel, axis=0)/self._nnod

	def nodes2gp(self,elfield):
		'''
		Transforms an entry field on the nodes
		(or their position) to the Gauss point 
		equivalent.
		'''
		return np.matmul(self._shapef.T,elfield)

	def derivative(self, xyzel):
		'''
		Derivative of the element:
		IN:
			> xyzel(nnod,3):        position of the nodes in cartesian coordinates
		OUT:
			> deri(3,nnod,ngauss):  derivatives per each gauss point
			> vol(ngauss):          volume per each gauss point
			> mle(nnod):            lumped mass matrix per each node
		'''
		# Allocate output arrays
		deri = np.zeros((self.ndim, self.nnod, self.ngauss), dtype=self.dtype)
		vol  = np.zeros((self.ngauss,), dtype=self.dtype)
		for igp in range(self.ngauss):
			J = np.matmul(self._gradi[:, :, igp], xyzel[:self.nnod, :])  # Jacobian
			# Determinant of the Jacobian
			detJ = np.linalg.det(J)
			# Inverse of the Jacobian
			Jinv = np.linalg.inv(J)
			# Derivatives, global shape gradients
			deri[:, :, igp] = np.matmul(Jinv, self._gradi[:, :, igp])
			vol[igp] = detJ*self._weigp[igp]  # Element volume
		return deri, vol  

	def integrative(self, xyzel):
		'''
		Integral of the element:
		IN:
			> xyzel(nnod,3):      position of the nodes in cartesian coordinates
		OUT:
			> integ(nnod,ngauss): integral per each gauss point
		'''
		integ = np.zeros((self.nnod, self.ngauss),dtype=np.double) # Integral coefficients
		for igp in range(self.ngauss):
			J = np.matmul(self._gradi[:, :, igp].astype(np.float64), xyzel.astype(np.float64), dtype=np.float64)  # Jacobian
			# Determinant of the Jacobian
			detJ = np.linalg.det(J)
			integ[:, igp] = self._weigp[igp]*self._shapef[:, igp]*detJ
		return integ

	def consistent(self,xyzel):
		'''
		Consistent mass matrix of the element:
		IN:
			> xyel(nnod,3):   position of the nodes in cartesian coordinates
		OUT:
			> mle(nnod,nnod): consistent mass matrix over the Gauss points	
		'''
		mle = np.zeros((self.nnod,self.nnod),self.dtype)
		# Integral computation
		for igp in range(self.ngauss):
			J = np.matmul(self._gradi[:, :, igp], xyzel[:self.nnod])  # Jacobian
			# Determinant of the Jacobian
			detJ = np.linalg.det(J)

			mle += self._weigp[igp]*detJ*np.outer(self._shapef[:, igp],self._shapef[:, igp])
		return mle

	def find_stz(self, xyz, xyzel, max_iter=20, tol=1e-10):
		'''
		Find a position of the point in xyz coordinates
		in element local coordinates stz:
		IN:
			> xyz(1,3):      position of the point
			> xyzel(nnod,3): position of the element nodes
		OUT:
			> stz(3,):       position of the point in local coordinates
		'''
		# Initial guess
		stz    = self._posgp[0, :]
		shapef = self._shapef[:, 0]
		gradi  = self._gradi[:, :, 0]
		# Compute residual
		f = xyz - np.dot(shapef, xyzel)
		r = np.sqrt(np.dot(f, f))
		# Newton-Raphson method
		for ii in range(max_iter):
			T = -np.dot(gradi, xyzel)
			Tinv = np.linalg.inv(T).T
			delta = -np.dot(Tinv, f)
			# New guess
			stz += delta
			shapef, gradi = self.shape_func(stz)
			# Compute new residual
			f = xyz - np.dot(shapef, xyzel)
			r = np.sqrt(np.dot(f, f))
			# Exit criteria
			if r < tol: break
		return stz

	def interpolate(self, stz, elfield):
		'''
		Interpolate a variable on a point inside the element:
		IN:
			> stz(3):             position of the point in local coordinates
			> elfield(nnod,ndim): variable to be interpolated
		OUT:
			> out(1,ndim):        interpolated variable at stz
		'''
		# Recover the shape function at the point stz
		shapef, _ = self.shape_func(stz)
		# Allocate output array
		out = np.zeros((1, elfield.shape[1]) if len(
			elfield.shape) > 1 else (1,), dtype=self.dtype)
		# Compute output array
		if len(elfield.shape) > 1:
			# Vectorial array
			for ii in range(elfield.shape[1]):
				out[0, ii] = np.dot(shapef, elfield[:, ii])
		else:
			# Scalar array
			out = np.dot(shapef, elfield)
		# Return output
		return out

	@property
	def nnod(self):
		return self._nnod

	@property
	def ngauss(self):
		return self._ngp

	@property
	def ndim(self):
		return 3

	@property
	def dtype(self):
		return self._dtype

	@property
	def nodes(self):
		return self._nodeList

	@property
	def posnod(self):
		return self._posnod

	@property
	def posgp(self):
		return self._posgp

	@property
	def shape(self):
		return self._shapef


class Bar(Element1D):
	'''
	Bar element
	'''
	def __init__(self, nodeList, ngauss=2):
		'''
		Define a bar element given a list of nodes and
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:          number of Gauss points
		'''
		# check number of Gauss points
		if not ngauss in [1, 2, 3, 4]:
			raiseError('Invalid number of Gauss points (%d)!' % ngauss)
		if not len(nodeList) == 2:
			raiseError('Invalid Bar (%d)!' % len(nodeList))
		# Allocate memory by initializing the parent class
		super(Bar, self).__init__(nodeList, ngauss)
		# Nodes positions
		self._posnod[0, 0] = -1.
		self._posnod[1, 0] = 1.
		# Gauss points positions and weights
		if ngauss == 1:
			self._posgp[0, 0] = 0.
			self._weigp[0] = 2.
		if ngauss == 2:
			self._posgp[0, 0] = -0.577350269189625764509148780502
			self._posgp[1, 0] = 0.577350269189625764509148780502
			self._weigp[0] = 1.
			self._weigp[1] = 1.
		if ngauss == 3:
			self._posgp[0, 0] = -0.774596669241483377035853079956
			self._posgp[1, 0] = 0.
			self._posgp[2, 0] = 0.774596669241483377035853079956
			self._weigp[0] = 5./9.
			self._weigp[1] = 8./9.
			self._weigp[2] = 5./9.
		if ngauss == 4:
			self._posgp[0, 0] = -0.861136311594052575223946488893
			self._posgp[1, 0] = -0.339981043584856264802665759103
			self._posgp[2, 0] = 0.339981043584856264802665759103
			self._posgp[3, 0] = 0.861136311594052575223946488893
			self._weigp[0] = 0.347854845137453857373063949222
			self._weigp[1] = 0.652145154862546142626936050778
			self._weigp[2] = 0.652145154862546142626936050778
			self._weigp[3] = 0.347854845137453857373063949222
		# Compute shape function and derivatives
		for igp in range(self.ngauss):
			shapef, gradi = self.shape_func(self._posgp[igp, :])
			self._shapef[:, igp] = shapef
			self._gradi[:, :, igp] = gradi

	def __str__(self):
		s = 'Bar nnod=%d' % self._nnod
		return s

	@property
	def type(self):
		return 2

	def new(self,ngauss=2,dtype=None):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return Bar(self.nodes,ngauss,dtype=self.dtype if dtype is None else dtype)

	def shape_func(self, stz):
		'''
		Shape function and gradient for a set of
		coordinates.
		'''
		shapef = np.zeros((self._nnod,))
		gradi = np.zeros((1, self._nnod))
		# Define the shape function in local coordinates
		shapef[0] = 0.5*(1.-stz[0])
		shapef[1] = 0.5*(1.+stz[0])
		# Define the gradient in local coordinates
		gradi[0, 0] = -0.5
		gradi[0, 1] = 0.5
		return shapef, gradi

	def isinside(self, xy, xyel, epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		# Ensure dealing with a 1D array
		x = xy if xyel.shape[1] == 1 else self.transform1D(xy, xyel)
		xel = xyel if xyel.shape[1] == 1 else self.transform1D(xyel, xyel)
		# Compute Jacobian and its inverse
		J = np.matmul(self._gradi[:, :, 0], xel[:])  # Jacobian
		Jinv = np.linalg.inv(J)                     # Inverse of the Jacobian
		# Find the point in natural coordinates
		rh_side = xy - xyel[0, :]
		point_loc = np.matmul(Jinv.T, rh_side)
		# The point has to be between -1 and 1
		min_loc = -1. - epsi
		max_loc = 1. + epsi
		if point_loc[0] >= min_loc and point_loc[0] <= max_loc:
			return True
		return False


class LinearTriangle(Element2D):
	'''
	Linear Triangle:
		|  3
		|
		|
		|
		|  1       2
	'''
	def __init__(self, nodeList, ngauss=3):
		'''
		Define a linear triangle given a list of nodes and
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:          number of Gauss points
		'''
		# check number of Gauss points
		if not ngauss in [1, 3]:
			raiseError('Invalid number of Gauss points (%d)!' % ngauss)
		if not len(nodeList) == 3:
			raiseError('Invalid Linear Tiangle (%d)!' % len(nodeList))
		# Allocate memory by initializing the parent class
		super(LinearTriangle, self).__init__(nodeList, ngauss)
		# Nodes positions
		self._posnod[0, :] = np.array([0., 0.], np.double)
		self._posnod[1, :] = np.array([1., 0.], np.double)
		self._posnod[2, :] = np.array([0., 1.], np.double)
		# Gauss points positions and weights
		if ngauss == 1:
			self._posgp[0, 0] = 1./3.
			self._posgp[0, 1] = 1./3.
			self._weigp[0] = 1./2.
		if ngauss == 3:
			self._posgp[0, 0] = 1./6.
			self._posgp[0, 1] = 1./6.
			self._posgp[1, 0] = 2./3.
			self._posgp[1, 1] = 1./6.
			self._posgp[2, 0] = 1./6.
			self._posgp[2, 1] = 2./3.
			self._weigp[0] = 1./6.
			self._weigp[1] = 1./6.
			self._weigp[2] = 1./6.
		# Compute shape function and derivatives
		for igp in range(self.ngauss):
			shapef, gradi = self.shape_func(self._posgp[igp, :])
			self._shapef[:, igp] = shapef
			self._gradi[:, :, igp] = gradi

	def __str__(self):
		s = 'Linear triangle nnod=%d' % self._nnod
		return s

	@property
	def type(self):
		return 10
	
	def new(self,ngauss=3,dtype=None):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return LinearTriangle(self.nodes,ngauss,dtype=self.dtype if dtype is None else dtype)

	def shape_func(self, stz):
		'''
		Shape function and gradient for a set of
		coordinates.
		'''
		shapef = np.zeros((self._nnod,))
		gradi = np.zeros((2, self._nnod))
		# Define the shape function in local coordinates
		shapef[0] = 1. - stz[0] - stz[1]
		shapef[1] = stz[0]
		shapef[2] = stz[1]
		# Define the gradient in local coordinates
		gradi[0, 0] = -1.
		gradi[1, 0] = -1.
		gradi[0, 1] = 1.
		gradi[1, 1] = 0.
		gradi[0, 2] = 0.
		gradi[1, 2] = 1.

		return shapef, gradi

	def isinside(self, xyz, xyzel, epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		# Ensure dealing with a 2D array
		xy   = xyz   if xyzel.shape[1] == 2 else self.transform2D(xyz, xyzel)
		xyel = xyzel if xyzel.shape[1] == 2 else self.transform2D(xyzel, xyzel)
		# Compute Jacobian and its inverse
		J    = np.matmul(self._gradi[:, :, 0], xyel[:, :])  # Jacobian
		Jinv = np.linalg.inv(J)                        # Inverse of the Jacobian
		# Find the point in natural coordinates
		rh_side   = xy - xyel[0, :]
		point_loc = np.matmul(Jinv.T, rh_side)
		# The point has to be between 0 and 1
		min_loc = -epsi
		max_loc = 1 + epsi
		if point_loc[0] >= min_loc and point_loc[0] <= max_loc:
			if point_loc[1] >= min_loc and point_loc[1] <= max_loc:
				# The sum of the points in natural coordinates also
				# has to be between 0 and 1
				ezzzt = 1 - np.sum(point_loc)
				if ezzzt >= min_loc and ezzzt <= max_loc:
					return True
		return False


class LinearQuadrangle(Element2D):
	'''
	Linear Quadrangle:
		|  3       4
		|
		|
		|
		|  1       2
	'''
	def __init__(self, nodeList, ngauss=4):
		'''
		Define a linear triangle given a list of nodes and
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:          number of Gauss points
		'''
		# check number of Gauss points
		if not ngauss in [1, 4]:
			raiseError('Invalid number of Gauss points (%d)!' % ngauss)
		if not len(nodeList) == 4:
			raiseError('Invalid Linear Quadrangle (%d)!' % len(nodeList))
		# Allocate memory by initializing the parent class
		super(LinearQuadrangle, self).__init__(nodeList, ngauss)
		# Nodes positions
		self._posnod[0, :] = np.array([-1., -1.], np.double)
		self._posnod[1, :] = np.array([ 1., -1.], np.double)
		self._posnod[2, :] = np.array([ 1.,  1.], np.double)
		self._posnod[3, :] = np.array([-1.,  1.], np.double)
		# Gauss points positions and weights
		if ngauss == 1:
			self._posgp[0, 0] = 0.
			self._posgp[0, 1] = 0.
			self._weigp[0] = 4.
		if ngauss == 4:
			q = 1.0/np.sqrt(3.0)
			self._posgp[0, 0] = -q
			self._posgp[0, 1] = -q
			self._posgp[1, 0] =  q
			self._posgp[1, 1] = -q
			self._posgp[2, 0] =  q
			self._posgp[2, 1] =  q
			self._posgp[3, 0] = -q
			self._posgp[3, 1] =  q
			self._weigp[0] = 1.
			self._weigp[1] = 1.
			self._weigp[2] = 1.
			self._weigp[3] = 1.
		# Compute shape function and derivatives
		for igp in range(self.ngauss):
			shapef, gradi = self.shape_func(self._posgp[igp, :])
			self._shapef[:, igp] = shapef
			self._gradi[:, :, igp] = gradi

	def __str__(self):
		s = 'Linear quadrangle nnod=%d' % self._nnod
		return s

	@property
	def type(self):
		return 12

	def new(self,ngauss=4,dtype=None):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return LinearQuadrangle(self.nodes,ngauss,dtype=self.dtype if dtype is None else dtype)

	def shape_func(self, stz):
		'''
		Shape function and gradient for a set of
		coordinates.
		'''
		shapef = np.zeros((self._nnod,))
		gradi = np.zeros((2, self._nnod))
		# Define the shape function in local coordinates
		shapef[0] = 0.25*(1.0-stz[0])*(1.0-stz[1])
		shapef[1] = 0.25*(1.0+stz[0])*(1.0-stz[1])
		shapef[2] = 0.25*(1.0+stz[0])*(1.0+stz[1])
		shapef[3] = 0.25*(1.0-stz[0])*(1.0+stz[1])
		# Define the gradient in local coordinates
		gradi[0, 0] = 0.25*(-1.0+stz[1])
		gradi[1, 0] = 0.25*(-1.0+stz[0])
		gradi[0, 1] = 0.25*( 1.0-stz[1])
		gradi[1, 1] = 0.25*(-1.0-stz[0])
		gradi[0, 2] = 0.25*( 1.0+stz[1])
		gradi[1, 2] = 0.25*( 1.0+stz[0])
		gradi[0, 3] = 0.25*(-1.0-stz[1])
		gradi[1, 3] = 0.25*( 1.0-stz[0])

		return shapef, gradi

	def isinside(self, xyz, xyzel, epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		# Split the element into triangles
		triangles = [[0,1,3], [3,2,1]]
		# Run over the triangles and see if the point is inside
		for triangle in triangles:
			xyzel1 = xyzel[triangle]
			t = LinearTriangle(self._nodeList[triangle], 1)
			if t.isinside(xyz, xyzel1, epsi=epsi): return True
		return False
	
class pOrderQuadrangle(Element2D):
	'''
	SEM pOrder Quadrangle: gauss points = nodes. GLL quadrature.
	'''
	def __init__(self, nodeList, ngauss, xi, posnod, weigp, shapef, gradi):
		# check number of Gauss points
		self._porder = np.int32(np.sqrt(ngauss) - 1)
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

	def __str__(self):
		s = 'High-order spectral quadrangle porder=%d' % self._porder
		return s

	@property
	def type(self):
		return 15
	
	def shape_func(self, stz):
		'''
		Shape function and gradient for a set of
		coordinates.
		'''
		shapef = np.array(np.zeros((self._nnod,)), dtype = np.double)
		gradi  = np.array(np.zeros((2, self._nnod)), dtype = np.double)
		c = 0
		for ii in range(self._pnodes.shape[0]):
			lag_xi  = lagrange(stz[0], ii, self._pnodes)
			dlag_xi = dlagrange(stz[0], ii, self._pnodes)
			for jj in range(self._pnodes.shape[0]):
				lag_eta      = lagrange(stz[1], jj, self._pnodes)
				dlag_eta     = dlagrange(stz[1], jj, self._pnodes)
				shapef[c]    = lag_xi*lag_eta
				gradi[0][c]  = dlag_xi*lag_eta
				gradi[1][c]  = lag_xi*dlag_eta
				c += 1
		return shapef, gradi


	def isinside(self, xyz, xyzel, epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		max_ite = 50
		X_0     = np.array([0, 0])
		alpha   = 1
		i    	= 0  # Iteration counter
		X    	= X_0
		conv    = False
		for ite in range(max_ite):
			f = xyz
			J = np.zeros((2,2))
			shapef, gradi = self.shape_func(X_0)
			for ipoint in range(xyzel.shape[0]):
				f = f - shapef[ipoint]*xyzel[ipoint,:]
				J[0,0] = J[0,0] - gradi[0,ipoint]*xyzel[ipoint,0]
				J[0,1] = J[0,1] - gradi[1,ipoint]*xyzel[ipoint,0]
				J[0,2] = J[0,2] - gradi[2,ipoint]*xyzel[ipoint,0]
				J[1,0] = J[1,0] - gradi[0,ipoint]*xyzel[ipoint,1]
				J[1,1] = J[1,1] - gradi[1,ipoint]*xyzel[ipoint,1]
				J[1,2] = J[1,2] - gradi[2,ipoint]*xyzel[ipoint,1]
				J[2,0] = J[2,0] - gradi[0,ipoint]*xyzel[ipoint,2]
				J[2,1] = J[2,1] - gradi[1,ipoint]*xyzel[ipoint,2]
				J[2,2] = J[2,2] - gradi[2,ipoint]*xyzel[ipoint,2]
			K  = np.linalg.inv(J)
			Xn = X - alpha*np.matmul(K, f)  # Newton-Raphson equation
			X  = Xn
			i  = i + 1
			if np.max(np.abs(f)) > 1000:
				break
			if np.max(np.abs(f))*np.max(np.abs(f)) < epsi:
				conv = True
		if conv and X[0] <= 1 + epsi and X[0] >= -1 -epsi and X[1] <= 1 + epsi and X[1] >= -1 -epsi:
			return True
		else:
			return False



class LinearTetrahedron(Element3D):
	'''
	Linear tetrahedron: s=[0:1], t=[0:1], z=[0:1]
	'''
	def __init__(self, nodeList, ngauss=4, dtype=np.double):
		'''
		Define a linear tetrahedron given a list of nodes and
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:          number of Gauss points
		'''
		# check number of Gauss points
		if not ngauss in [1, 4]:
			raiseError('Invalid number of Gauss points (%d)!' % ngauss)
		if not len(nodeList) == 4:
			raiseError('Invalid Linear Tetrahedron (%d)!' % len(nodeList))
		# Allocate memory by initializing the parent class
		super(LinearTetrahedron, self).__init__(nodeList, ngauss, dtype=dtype)
		# Nodes positions
		self._posnod[0, :] = np.array([0., 0., 0.],self.dtype)
		self._posnod[1, :] = np.array([1., 0., 0.],self.dtype)
		self._posnod[2, :] = np.array([0., 1., 0.],self.dtype)
		self._posnod[3, :] = np.array([0., 0., 1.],self.dtype)
		# Gauss points positions and weights
		if ngauss == 1:
			self._posgp[0, 0] = 1./4.
			self._posgp[0, 1] = 1./4.
			self._posgp[0, 2] = 1./4.
			self._weigp[0]    = 1./6.
		if ngauss == 4:
			a = 0.5854101966249685
			b = 0.1381966011250105
			self._posgp[0, 0] = b
			self._posgp[0, 1] = b
			self._posgp[0, 2] = b
			self._posgp[1, 0] = a
			self._posgp[1, 1] = b
			self._posgp[1, 2] = b
			self._posgp[2, 0] = b
			self._posgp[2, 1] = a
			self._posgp[2, 2] = b
			self._posgp[3, 0] = b
			self._posgp[3, 1] = b
			self._posgp[3, 2] = a
			self._weigp[0]    = 1./24.
			self._weigp[1]    = 1./24.
			self._weigp[2]    = 1./24.
			self._weigp[3]    = 1./24.
		# Compute shape function and derivatives
		for igp in range(self.ngauss):
			shapef, gradi = self.shape_func(self._posgp[igp, :])
			self._shapef[:, igp]   = shapef
			self._gradi[:, :, igp] = gradi
			del shapef, gradi

	def __str__(self):
		s = 'Linear tetrahedron nnod=%d' % self._nnod
		return s

	@property
	def type(self):
		return 30

	def new(self,ngauss=4,dtype=None):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return LinearTetrahedron(self.nodes,ngauss,dtype=self.dtype if dtype is None else dtype)

	def shape_func(self, stz):
		'''
		Shape function and gradient for a set of
		coordinates.
		'''
		shapef = np.zeros((self._nnod,),self.dtype)
		gradi = np.zeros((3, self._nnod),self.dtype)
		# Define the shape function in local coordinates
		shapef[0] = 1. - stz[0] - stz[1] - stz[2]
		shapef[1] = stz[0]
		shapef[2] = stz[1]
		shapef[3] = stz[2]
		# Define the gradient in local coordinates
		gradi[0, 0] = -1.
		gradi[1, 0] = -1.
		gradi[2, 0] = -1.
		gradi[0, 1] = 1.
		gradi[1, 2] = 1.
		gradi[2, 3] = 1.

		return shapef, gradi

	def isinside(self, xyz, xyzel, epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		# Compute Jacobian and its inverse
		J = np.matmul(self._gradi[:, :, 0], xyzel[:, :])  # Jacobian
		Jinv = np.linalg.inv(J)                         # Inverse of the Jacobian
		# Find the point in natural coordinates
		rh_side = xyz - xyzel[0, :]
		point_loc = np.matmul(Jinv.T, rh_side)
		# The point has to be between 0 and 1
		min_loc = -epsi
		max_loc = 1 + epsi
		if point_loc[0] >= min_loc and point_loc[0] <= max_loc:
			if point_loc[1] >= min_loc and point_loc[1] <= max_loc:
				if point_loc[2] >= min_loc and point_loc[2] <= max_loc:
					# The sum of the points in natural coordinates also
					# has to be between 0 and 1
					ezzzt = 1 - np.sum(point_loc)
					if ezzzt >= min_loc and ezzzt <= max_loc:
						return True
		return False


class LinearPyramid(Element3D):
	'''
	Linear Pyramid: s=[-1:1], t=[-1:1], z=[-1:1]
	'''
	def __init__(self, nodeList, ngauss=5, dtype=np.double):
		'''
		Define a linear pyramid given a list of nodes and
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:          number of Gauss points
		'''
		# check number of Gauss points
		if not ngauss in [1, 5]:
			raiseError('Invalid number of Gauss points (%d)!' % ngauss)
		if not len(nodeList) == 5:
			raiseError('Invalid Linear Pyramid (%d)!' % len(nodeList))
		# Allocate memory by initializing the parent class
		super(LinearPyramid, self).__init__(nodeList, ngauss, dtype=dtype)
		# Nodes positions
		self._posnod[0, :] = np.array([-1., -1., -1.], self.dtype)
		self._posnod[1, :] = np.array([1., -1., -1.],  self.dtype)
		self._posnod[2, :] = np.array([1., 1., -1.],   self.dtype)
		self._posnod[3, :] = np.array([-1., 1., -1.],  self.dtype)
		self._posnod[4, :] = np.array([0., 0., 1.],    self.dtype)
		# Gauss points positions and weights
		jk = np.zeros((4, 2), dtype=self.dtype)
		jk[0, 0] = -1.
		jk[0, 1] = -1.
		jk[1, 0] = 1.
		jk[1, 1] = -1.
		jk[2, 0] = 1.
		jk[2, 1] = 1.
		jk[3, 0] = -1.
		jk[3, 1] = 1.
		if ngauss == 1:
			self._posgp[0, 0] = 0.
			self._posgp[0, 1] = 0.
			self._posgp[0, 2] = 0.5
			self._weigp[0] = 128./27.
		if ngauss == 5:
			g1 = 8.*np.sqrt(2./15.)/5.
			for ii in range(4):
				j = jk[ii, 0]
				k = jk[ii, 1]
				self._posgp[ii, 0] = j*g1
				self._posgp[ii, 1] = k*g1
				self._posgp[ii, 2] = -2./3.
				self._weigp[ii] = 81./100.
			self._posgp[4, 0] = 0.
			self._posgp[4, 1] = 0.
			self._posgp[4, 2] = 2./5.
			self._weigp[4] = 125./27.
		# Compute shape function and derivatives
		for igp in range(self.ngauss):
			shapef, gradi = self.shape_func(self._posgp[igp, :])
			self._shapef[:, igp]   = shapef
			self._gradi[:, :, igp] = gradi
			del shapef, gradi

	def __str__(self):
		s = 'Linear pyramid nnod=%d' % self._nnod
		return s

	@property
	def type(self):
		return 32

	def new(self,ngauss=5,dtype=None):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return LinearPyramid(self.nodes,ngauss,dtype=self.dtype if dtype is None else dtype)

	def shape_func(self, stz):
		'''
		Shape function and gradient for a set of
		coordinates.
		'''
		shapef = np.zeros((self._nnod,),  self.dtype)
		gradi = np.zeros((3, self._nnod), self.dtype)
		# Define the shape function in local coordinates
		one8 = 0.125
		shapef[0] = one8*(1. - stz[0])*(1. - stz[1])*(1. - stz[2])
		shapef[1] = one8*(1. + stz[0])*(1. - stz[1])*(1. - stz[2])
		shapef[2] = one8*(1. + stz[0])*(1. + stz[1])*(1. - stz[2])
		shapef[3] = one8*(1. - stz[0])*(1. + stz[1])*(1. - stz[2])
		shapef[4] = 0.5*(1. + stz[2])
		# Define the gradient in local coordinates
		gradi[0, 0] = -one8*(1. - stz[1])*(1. - stz[2])
		gradi[1, 0] = -one8*(1. - stz[0])*(1. - stz[2])
		gradi[2, 0] = -one8*(1. - stz[0])*(1. - stz[1])
		gradi[0, 1] = one8*(1. - stz[1])*(1. - stz[2])
		gradi[1, 1] = -one8*(1. + stz[0])*(1. - stz[2])
		gradi[2, 1] = -one8*(1. + stz[0])*(1. - stz[1])
		gradi[0, 2] = one8*(1. + stz[1])*(1. - stz[2])
		gradi[1, 2] = one8*(1. + stz[0])*(1. - stz[2])
		gradi[2, 2] = -one8*(1. + stz[0])*(1. + stz[1])
		gradi[0, 3] = -one8*(1. + stz[1])*(1. - stz[2])
		gradi[1, 3] = one8*(1. - stz[0])*(1. - stz[2])
		gradi[2, 3] = -one8*(1. - stz[0])*(1. + stz[1])
		gradi[0, 4] = 0.
		gradi[1, 4] = 0.
		gradi[2, 4] = 0.5
		return shapef, gradi

	def isinside(self, xyz, xyzel, epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		# Split the element into tetras
		tetras = [[0, 1, 2, 4], [0, 2, 3, 4]]
		# Run over the tetras and see if the point is inside
		for tetra in tetras:
			xyzel1 = xyzel[tetra]
			t = LinearTetrahedron(self._nodeList[tetra], 1)
			if t.isinside(xyz, xyzel1, epsi=epsi): return True
		return False


class LinearPrism(Element3D):
	'''
	Linear Prism: s=[0:1], t=[0:1], z=[0:1]
	'''
	def __init__(self, nodeList, ngauss=6, dtype=np.double):
		'''
		Define a linear prism given a list of nodes and
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:          number of Gauss points
		'''
		# check number of Gauss points
		if not ngauss in [1, 6]:
			raiseError('Invalid number of Gauss points (%d)!' % ngauss)
		if not len(nodeList) == 6:
			raiseError('Invalid Linear Prism (%d)!' % len(nodeList))
		# Allocate memory by initializing the parent class
		super(LinearPrism, self).__init__(nodeList, ngauss, dtype=dtype)
		# Nodes positions
		self._posnod[0, :] = np.array([0., 0., 0.],self.dtype)
		self._posnod[1, :] = np.array([1., 0., 0.],self.dtype)
		self._posnod[2, :] = np.array([0., 1., 0.],self.dtype)
		self._posnod[3, :] = np.array([0., 0., 1.],self.dtype)
		self._posnod[4, :] = np.array([1., 0., 1.],self.dtype)
		self._posnod[5, :] = np.array([0., 1., 1.],self.dtype)
		# Gauss points positions and weights
		if ngauss == 1:
			self._posgp[0, 0] = 1./3.
			self._posgp[0, 1] = 1./3.
			self._posgp[0, 2] = 1./2.
			self._weigp[0] = 1./2.
		if ngauss == 6:
			self._posgp[0, 0] = 2./3.
			self._posgp[0, 1] = 1./6.
			self._posgp[0, 2] = 0.21132486540518711774542560974902
			self._posgp[1, 0] = 1./6.
			self._posgp[1, 1] = 2./3.
			self._posgp[1, 2] = 0.21132486540518711774542560974902
			self._posgp[2, 0] = 1./6.
			self._posgp[2, 1] = 1./6.
			self._posgp[2, 2] = 0.21132486540518711774542560974902
			self._posgp[3, 0] = 2./3.
			self._posgp[3, 1] = 1./6.
			self._posgp[3, 2] = 0.78867513459481288225457439025098
			self._posgp[4, 0] = 1./6.
			self._posgp[4, 1] = 2./3.
			self._posgp[4, 2] = 0.78867513459481288225457439025098
			self._posgp[5, 0] = 1./6.
			self._posgp[5, 1] = 1./6.
			self._posgp[5, 2] = 0.78867513459481288225457439025098
			self._weigp[0] = 1./12.
			self._weigp[1] = 1./12.
			self._weigp[2] = 1./12.
			self._weigp[3] = 1./12.
			self._weigp[4] = 1./12.
			self._weigp[5] = 1./12.
		# Compute shape function and derivatives
		for igp in range(self.ngauss):
			shapef, gradi = self.shape_func(self._posgp[igp, :])
			self._shapef[:, igp]   = shapef
			self._gradi[:, :, igp] = gradi
			del shapef, gradi

	def __str__(self):
		s = 'Linear prism nnod=%d' % self._nnod
		return s

	@property
	def type(self):
		return 34

	def new(self,ngauss=6,dtype=None):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return LinearPrism(self.nodes,ngauss,dtype=self.dtype if dtype is None else dtype)

	def shape_func(self, stz):
		'''
		Shape function and gradient for a set of
		coordinates.
		'''
		shapef = np.zeros((self._nnod,),self.dtype)
		gradi  = np.zeros((3, self._nnod),self.dtype)
		# Define the shape function in local coordinates
		shapef[0] = (1. - stz[0] - stz[1])*(1. - stz[2])
		shapef[1] = stz[0]*(1. - stz[2])
		shapef[2] = stz[1]*(1. - stz[2])
		shapef[3] = (1. - stz[0] - stz[1])*stz[2]
		shapef[4] = stz[0]*stz[2]
		shapef[5] = stz[1]*stz[2]
		# Define the gradient in local coordinates
		gradi[0, 0] = stz[2] - 1.
		gradi[1, 0] = stz[2] - 1.
		gradi[2, 0] = stz[0] + stz[1] - 1.
		gradi[0, 1] = 1. - stz[2]
		gradi[2, 1] = -stz[0]
		gradi[1, 2] = 1. - stz[2]
		gradi[2, 2] = -stz[1]
		gradi[0, 3] = -stz[2]
		gradi[1, 3] = -stz[2]
		gradi[2, 3] = 1. - stz[0] - stz[1]
		gradi[0, 4] = stz[2]
		gradi[2, 4] = stz[0]
		gradi[1, 5] = stz[2]
		gradi[2, 5] = stz[1]
		return shapef, gradi

	def isinside(self, xyz, xyzel, epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		# Split the element into tetras
		tetras = [[0, 1, 2, 3], [4, 1, 3, 5], [5, 3, 2, 1]]
		# Run over the tetras and see if the point is inside
		for tetra in tetras:
			xyzel1 = xyzel[tetra]
			t = LinearTetrahedron(self._nodeList[tetra], 1)
			if t.isinside(xyz, xyzel1, epsi=epsi): return True
		return False


class TrilinearBrick(Element3D):
	'''
	Trilinear brick
	'''
	def __init__(self, nodeList, ngauss=8):
		'''
		Define a trilinear brick given a list of nodes and
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:          number of Gauss points
		'''
		if not len(nodeList) == 8:
			raiseError('Invalid Trilinear Brick (%d)!' % len(nodeList))
		# Allocate memory by initializing the parent class
		super(TrilinearBrick, self).__init__(nodeList, ngauss)
		posgl = np.zeros((4,), dtype=np.double)
		weigl = np.zeros((4,), dtype=np.double)
		nlocs = int(ngauss**(1./3.))
		# Nodes positions
		self._posnod[0, :] = np.array([-1., -1., -1.],self.dtype)
		self._posnod[1, :] = np.array([1., -1., -1.], self.dtype)
		self._posnod[2, :] = np.array([1., 1., -1.],  self.dtype)
		self._posnod[3, :] = np.array([-1., 1., -1.], self.dtype)
		self._posnod[4, :] = np.array([-1., -1., 1.], self.dtype)
		self._posnod[5, :] = np.array([1., -1., 1.],  self.dtype)
		self._posnod[6, :] = np.array([1., 1., 1.],   self.dtype)
		self._posnod[7, :] = np.array([-1., 1., 1.],  self.dtype)
		# Gauss points positions and weights
		if nlocs == 1:
			posgl[0] = 0.
			weigl[0] = 2.
		elif nlocs == 2:
			posgl[0] = -0.577350269189626
			posgl[1] = 0.577350269189626
			weigl[0] = 1.
			weigl[1] = 1.
		else:
			raiseError('Invalid number of Gauss points (%d)!' % ngauss)

		igauss = 0
		for ilocs in range(nlocs):
			for jlocs in range(nlocs):
				for klocs in range(nlocs):
					self._weigp[igauss] = weigl[ilocs]*weigl[jlocs]*weigl[klocs]
					self._posgp[igauss, 0] = posgl[ilocs]
					self._posgp[igauss, 1] = posgl[jlocs]
					self._posgp[igauss, 2] = posgl[klocs]
					igauss += 1

		# Compute shape function and derivatives
		for igp in range(self.ngauss):
			shapef, gradi = self.shape_func(self._posgp[igp, :])
			self._shapef[:, igp]   = shapef
			self._gradi[:, :, igp] = gradi
			del shapef, gradi

	def __str__(self):
		s = 'Trilinear brick nnod=%d' % self._nnod
		return s

	@property
	def type(self):
		return 37

	def new(self,ngauss=8,dtype=None):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return TrilinearBrick(self.nodes,ngauss,dtype=self.dtype if dtype is None else dtype)

	def shape_func(self, stz):
		'''
		Shape function and gradient for a set of
		coordinates.
		'''
		shapef = np.zeros((self._nnod,),  self.dtype)
		gradi = np.zeros((3, self._nnod), self.dtype)
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
		gradi[0, 0] = -0.5*tm*zm
		gradi[1, 0] = -0.5*sm*zm
		gradi[2, 0] = -0.5*sm*tm
		gradi[0, 1] = 0.5*tm*zm
		gradi[1, 1] = -0.5*sq*zm
		gradi[2, 1] = -0.5*sq*tm
		gradi[0, 2] = 0.5*tp*zm
		gradi[1, 2] = 0.5*sq*zm
		gradi[2, 2] = -0.5*sq*tp
		gradi[0, 3] = -0.5*tp*zm
		gradi[1, 3] = 0.5*sm*zm
		gradi[2, 3] = -0.5*sm*tp
		gradi[0, 4] = -0.5*tm*zp
		gradi[1, 4] = -0.5*sm*zp
		gradi[2, 4] = 0.5*sm*tm
		gradi[0, 5] = 0.5*tm*zp
		gradi[1, 5] = -0.5*sq*zp
		gradi[2, 5] = 0.5*sq*tm
		gradi[0, 6] = 0.5*tp*zp
		gradi[1, 6] = 0.5*sq*zp
		gradi[2, 6] = 0.5*sq*tp
		gradi[0, 7] = -0.5*tp*zp
		gradi[1, 7] = 0.5*sm*zp
		gradi[2, 7] = 0.5*sm*tp
		return shapef, gradi

	def isinside(self, xyz, xyzel, epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		# Split the element into tetras
		tetras = [[0,1,3,4], [5,1,4,7], [7,4,3,1],
				  [2,3,1,6], [7,3,6,5], [5,6,1,3]]
		# Run over the tetras and see if the point is inside
		for tetra in tetras:
			xyzel1 = xyzel[tetra]
			t = LinearTetrahedron(self._nodeList[tetra], 1)
			if t.isinside(xyz, xyzel1, epsi=epsi): return True
		return False


class TriQuadraticBrick(Element3D):
	'''
	TriQuadratic brick
	'''
	def __init__(self, nodeList, ngauss=27, dtype=np.double):
		'''
		Define a triquadratic brick given a list of nodes and
		the number of Gauss points.
		IN:
			> nodeList(nnod):  list of the nodes on the element
			> ngauss:          number of Gauss points
		'''
		if not len(nodeList) == 27:
			raiseError('Invalid Trilinear Brick (%d)!' % len(nodeList))
		# Allocate memory by initializing the parent class
		super(TriQuadraticBrick, self).__init__(nodeList, ngauss, dtype=dtype)
		posgl = np.zeros((4,), dtype=self.dtype)
		weigl = np.zeros((4,), dtype=self.dtype)
		nlocs = int(ngauss**(1./3.))
		# Nodes positions
		self._posnod[0, :] = np.array([-1., -1., -1.], self.dtype)
		self._posnod[1, :] = np.array([1., -1., -1.],  self.dtype)
		self._posnod[2, :] = np.array([1., 1., -1.],   self.dtype)
		self._posnod[3, :] = np.array([-1., 1., -1.],  self.dtype)
		self._posnod[4, :] = np.array([-1., -1., 1.],  self.dtype)
		self._posnod[5, :] = np.array([1., -1., 1.],   self.dtype)
		self._posnod[6, :] = np.array([1., 1., 1.],    self.dtype)
		self._posnod[7, :] = np.array([-1., 1., 1.],   self.dtype)
		self._posnod[8, :] = np.array([0., -1., -1.],  self.dtype)
		self._posnod[9, :] = np.array([1., 0., -1.],   self.dtype)
		self._posnod[10, :] = np.array([0., 1., -1.],  self.dtype)
		self._posnod[11, :] = np.array([-1., 0., -1.], self.dtype)
		self._posnod[12, :] = np.array([-1., -1., 0.], self.dtype)
		self._posnod[13, :] = np.array([1., -1., 0.],  self.dtype)
		self._posnod[14, :] = np.array([1., 1., 0.],   self.dtype)
		self._posnod[15, :] = np.array([-1., 1., 0.],  self.dtype)
		self._posnod[16, :] = np.array([0., -1., 1.],  self.dtype)
		self._posnod[17, :] = np.array([1., 0., 1.],   self.dtype)
		self._posnod[18, :] = np.array([0., 1., 1.],   self.dtype)
		self._posnod[19, :] = np.array([-1., 0., 1.],  self.dtype)
		self._posnod[20, :] = np.array([0., 0., -1.],  self.dtype)
		self._posnod[21, :] = np.array([0., -1., 0.],  self.dtype)
		self._posnod[22, :] = np.array([1., 0., 0.],   self.dtype)
		self._posnod[23, :] = np.array([0., 1., 0.],   self.dtype)
		self._posnod[24, :] = np.array([-1., 0., 0.],  self.dtype)
		self._posnod[25, :] = np.array([0., 0., 1.],   self.dtype)
		self._posnod[26, :] = np.array([0., 0., 0.],   self.dtype)
		# Gauss points positions and weights
		if nlocs == 1:
			posgl[0] = 0.
			weigl[0] = 2.
		elif nlocs == 2:
			posgl[0] = -0.577350269189626
			posgl[1] = 0.577350269189626
			weigl[0] = 1.
			weigl[1] = 1.
		elif nlocs == 3:
			posgl[0] = -0.774596669241483377035853079956
			posgl[1] = 0.0
			posgl[2] = 0.774596669241483377035853079956
			weigl[0] = 5./9.
			weigl[1] = 8./9.
			weigl[2] = 5./9.
		else:
			raiseError('Invalid number of Gauss points (%d)!' % ngauss)

		igauss = 0
		for ilocs in range(nlocs):
			for jlocs in range(nlocs):
				for klocs in range(nlocs):
					self._weigp[igauss] = weigl[ilocs]*weigl[jlocs]*weigl[klocs]
					self._posgp[igauss, 0] = posgl[ilocs]
					self._posgp[igauss, 1] = posgl[jlocs]
					self._posgp[igauss, 2] = posgl[klocs]
					igauss += 1

		# Compute shape function and derivatives
		for igp in range(self.ngauss):
			shapef, gradi = self.shape_func(self._posgp[igp, :])
			self._shapef[:, igp]   = shapef
			self._gradi[:, :, igp] = gradi
			del shapef, gradi

	def __str__(self):
		s = 'TriQuadratic brick nnod=%d' % self._nnod
		return s

	@property
	def type(self):
		return 39

	def new(self,ngauss=8,dtype=None):
		'''
		Return a new instance of the class with a different
		number of gauss points
		'''
		return TriQuadraticBrick(self.nodes,ngauss,dtype=self.dtype if dtype is None else dtype)

	def shape_func(self, stz):
		'''
		Shape function and gradient for a set of
		coordinates.
		'''
		shapef = np.zeros((self._nnod,),  self.dtype)
		gradi = np.zeros((3, self._nnod), self.dtype)

		sl = stz[0]*(stz[0]-1.)
		tl = stz[1]*(stz[1]-1.)
		zl = stz[2]*(stz[2]-1.)
		sq = stz[0]*(stz[0]+1.)
		tp = stz[1]*(stz[1]+1.)
		zp = stz[2]*(stz[2]+1.)
		s1 = 2.*stz[0]-1.
		t1 = 2.*stz[1]-1.
		z1 = 2.*stz[2]-1.
		s2 = 1.-stz[0]*stz[0]
		t2 = 1.-stz[1]*stz[1]
		z2 = 1.-stz[2]*stz[2]
		s3 = 1.+2.*stz[0]
		t3 = 1.+2.*stz[1]
		z3 = 1.+2.*stz[2]
		s4 = -2.*stz[0]
		t4 = -2.*stz[1]
		z4 = -2.*stz[2]

		shapef[0] = 0.125*sl*tl*zl
		shapef[1] = 0.125*sq*tl*zl
		shapef[2] = 0.125*sq*tp*zl
		shapef[3] = 0.125*sl*tp*zl
		shapef[4] = 0.125*sl*tl*zp
		shapef[5] = 0.125*sq*tl*zp
		shapef[6] = 0.125*sq*tp*zp
		shapef[7] = 0.125*sl*tp*zp
		shapef[8] = 0.25*s2*tl*zl
		shapef[9] = 0.25*sq*t2*zl
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
		gradi[0, 10] = 0.25*s4*tp*zl
		gradi[1, 10] = 0.25*s2*t3*zl
		gradi[2, 10] = 0.25*s2*tp*z1
		gradi[0, 11] = 0.25*s1*t2*zl
		gradi[1, 11] = 0.25*sl*t4*zl
		gradi[2, 11] = 0.25*sl*t2*z1
		gradi[0, 12] = 0.25*s1*tl*z2
		gradi[1, 12] = 0.25*sl*t1*z2
		gradi[2, 12] = 0.25*sl*tl*z4
		gradi[0, 13] = 0.25*s3*tl*z2
		gradi[1, 13] = 0.25*sq*t1*z2
		gradi[2, 13] = 0.25*sq*tl*z4
		gradi[0, 14] = 0.25*s3*tp*z2
		gradi[1, 14] = 0.25*sq*t3*z2
		gradi[2, 14] = 0.25*sq*tp*z4
		gradi[0, 15] = 0.25*s1*tp*z2
		gradi[1, 15] = 0.25*sl*t3*z2
		gradi[2, 15] = 0.25*sl*tp*z4
		gradi[0, 16] = 0.25*s4*tl*zp
		gradi[1, 16] = 0.25*s2*t1*zp
		gradi[2, 16] = 0.25*s2*tl*z3
		gradi[0, 17] = 0.25*s3*t2*zp
		gradi[1, 17] = 0.25*sq*t4*zp
		gradi[2, 17] = 0.25*sq*t2*z3
		gradi[0, 18] = 0.25*s4*tp*zp
		gradi[1, 18] = 0.25*s2*t3*zp
		gradi[2, 18] = 0.25*s2*tp*z3
		gradi[0, 19] = 0.25*s1*t2*zp
		gradi[1, 19] = 0.25*sl*t4*zp
		gradi[2, 19] = 0.25*sl*t2*z3
		gradi[0, 20] = 0.5*s4*t2*zl
		gradi[1, 20] = 0.5*s2*t4*zl
		gradi[2, 20] = 0.5*s2*t2*z1
		gradi[0, 21] = 0.5*s4*tl*z2
		gradi[1, 21] = 0.5*s2*t1*z2
		gradi[2, 21] = 0.5*s2*tl*z4
		gradi[0, 22] = 0.5*s3*t2*z2
		gradi[1, 22] = 0.5*sq*t4*z2
		gradi[2, 22] = 0.5*sq*t2*z4
		gradi[0, 23] = 0.5*s4*tp*z2
		gradi[1, 23] = 0.5*s2*t3*z2
		gradi[2, 23] = 0.5*s2*tp*z4
		gradi[0, 24] = 0.5*s1*t2*z2
		gradi[1, 24] = 0.5*sl*t4*z2
		gradi[2, 24] = 0.5*sl*t2*z4
		gradi[0, 25] = 0.5*s4*t2*zp
		gradi[1, 25] = 0.5*s2*t4*zp
		gradi[2, 25] = 0.5*s2*t2*z3
		gradi[0, 26] = s4*t2*z2
		gradi[1, 26] = s2*t4*z2
		gradi[2, 26] = s2*t2*z4

		return shapef, gradi

	def isinside(self, xyz, xyzel, epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		# Split the element into tetras
		tetras = [[0,1,3,4], [5,1,4,7], [7,4,3,1],
				  [2,3,1,6], [7,3,6,5], [5,6,1,3]]
		# Run over the tetras and see if the point is inside
		for tetra in tetras:
			xyzel1 = xyzel[tetra]
			t = LinearTetrahedron(self._nodeList[tetra], 1)
			if t.isinside(xyz, xyzel1, epsi=epsi): return True
		return False

class pOrderHexahedron(Element3D):
	'''
	SEM pOrder Hexahedron: gauss points = nodes. GLL quadrature.
	'''
	def __init__(self, nodeList, ngauss, xi, posnod, weigp, shapef, gradi):
		# check number of Gauss points
		self._porder = np.int32(np.cbrt(ngauss) - 1)
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

	def __str__(self):
		s = 'High-order spectral hexahedron porder=%d' % self._porder
		return s

	@property
	def type(self):
		return 40
	
	def shape_func(self, stz):
		'''
		Shape function and gradient for a set of coordinates.
		'''
		shapef = np.array(np.zeros((self._nnod,)), dtype = np.double)
		gradi  = np.array(np.zeros((3, self._nnod)), dtype = np.double)
		c = 0
		for kk in range(self._pnodes.shape[0]):
			lag_xi2   = lagrange(stz[2], kk, self._pnodes)
			dlag_xi2  = dlagrange(stz[2], kk, self._pnodes)
			for ii in range(self._pnodes.shape[0]):
				lag_xi0  = lagrange(stz[0], ii, self._pnodes)
				dlag_xi0 = dlagrange(stz[0], ii, self._pnodes)
				for jj in range(self._pnodes.shape[0]):
					lag_xi1     = lagrange(stz[1], jj, self._pnodes)
					dlag_xi1    = dlagrange(stz[1], jj, self._pnodes)
					shapef[c]   = lag_xi0*lag_xi1*lag_xi2
					gradi[0][c] = dlag_xi0*lag_xi1*lag_xi2
					gradi[1][c] = lag_xi0*dlag_xi1*lag_xi2
					gradi[2][c] = lag_xi0*lag_xi1*dlag_xi2
					c = c + 1 
		
		return shapef, gradi
	
	def isinside(self, xyz, xyzel, epsi=np.finfo(np.double).eps):
		'''
		Find if a point is inside an element.
		'''
		max_ite = 50
		X_0     = np.array([0, 0, 0])
		alpha   = 1
		i    	= 0  # Iteration counter
		X    	= X_0
		conv    = False
		for ite in range(max_ite):
			f = xyz
			J = np.zeros((3,3))
			shapef, gradi = self.shape_func(X)
			for ipoint in range(xyzel.shape[0]):
				f = f - shapef[ipoint]*xyzel[ipoint,:]
				J[0,0] = J[0,0] - gradi[0,ipoint]*xyzel[ipoint,0]
				J[0,1] = J[0,1] - gradi[1,ipoint]*xyzel[ipoint,0]
				J[0,2] = J[0,2] - gradi[2,ipoint]*xyzel[ipoint,0]
				J[1,0] = J[1,0] - gradi[0,ipoint]*xyzel[ipoint,1]
				J[1,1] = J[1,1] - gradi[1,ipoint]*xyzel[ipoint,1]
				J[1,2] = J[1,2] - gradi[2,ipoint]*xyzel[ipoint,1]
				J[2,0] = J[2,0] - gradi[0,ipoint]*xyzel[ipoint,2]
				J[2,1] = J[2,1] - gradi[1,ipoint]*xyzel[ipoint,2]
				J[2,2] = J[2,2] - gradi[2,ipoint]*xyzel[ipoint,2]
			K  = np.linalg.inv(J)
			Xn = X - alpha*np.matmul(K, f)  # Newton-Raphson equation
			X  = Xn
			i  = i + 1
			if np.max(np.abs(f)) > 1000:
				break
			if np.max(np.abs(f))*np.max(np.abs(f)) < epsi:
				conv = True
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

def defineHighOrderElement(porder, ndime):
	'''
	Computes the Gauss points positions and weights for the GLL quadrature
	Computes the shape functions and derivatives on the Gauss points with the Lagrangian polynomial
	ndime = 2 elements are quads and ndime=3 elements are hexes
	'''
	npoint = porder+1
	npoin2 = npoint**2
	ngauss = npoint**ndime
	posnod = np.zeros((ngauss, ndime), dtype=np.double)
	weigp  = np.zeros((ngauss,), dtype=np.double)
	shapef = np.eye(ngauss, dtype=np.double) # Shape function properties: 1 in its Gauss point and 0 on the rest
	gradi  = np.zeros((ndime, ngauss, ngauss), dtype=np.double)
	dlag   = np.zeros(ndime, dtype=np.double)
	offset = np.zeros((ndime,), dtype=np.int32)
	#Compute gauss points positions and weights with the GLL quadrature
	xi, wi = quadrature_GaussLobatto(npoint)
	c=0
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

	# Compute the derivatives of the shape functions in the Gauss points with the Lagrangian polynomials
	for igp in range(ngauss):
		off  = np.floor((igp/npoint))
		off2 = np.floor((igp/npoin2))
		for ii in range(npoint):
			offset[0] = igp-npoint*(off-ii)
			offset[1] = off*npoint+ii
			if ndime == 3: #Particularize for HEXES
				offset[0] += off2*npoin2
				offset[2]  = igp-npoin2*(off2-ii)
			for idime in range(ndime):
				dlag[idime] = dlagrange(posnod[igp,idime], ii, xi)
				gradi[idime, offset[idime], igp] = dlag[idime]

	
	return ngauss, xi, posnod, weigp, shapef, gradi

def createElementByType(ltype, nodeList, ngauss=-1, xi=None, posnod=None, weigp=None, shapef=None, gradi=None):
	'''
	Use the data in LTYPE to create an element according
	to its type as defined in Alya defmod/def_elmtyp.f90.

	IN:
		> ltype:     type of element
		> nodeList:  array with the number of nodes of the element
		> ngauss:    number of gauss points, optional
	'''
	if not ltype in ALYA_ELEMDICT.keys(): raiseError('Element type %d not implemented!' % ltype)
	# Return element and node cut according to the dict
	elem = ALYA_ELEMDICT[ltype]
	# Alya must be consistent with the nodeList if the mesh
	# contains more than one element, hence it puts -1 on the
	# elements that don't have a node in that position.
	# We will simply filter them
	if elem['nnod'] < 0: #Case for High Order elements with number of nodes depending on the order
		nnod = ngauss
		return elem['class'](nodeList[:nnod].copy(),ngauss, xi, posnod, weigp, shapef, gradi)
	else: # Case for linear elements with a fix number of nodes
		nnod = elem['nnod']
		return elem['class'](nodeList[:nnod].copy()) if ngauss < 0 else elem['class'](nodeList[:nnod].copy(),ngauss)
