#!/usr/bin/env cython
# cython: boundscheck = False
# cython: wraparound = False
# cython: nonecheck = False
#
# pyQvarsi, GEOM entities.
#
# Small geometric module to cut a field or a mesh.
#
# Basic entities and operators.
#
# Last rev: 25/02/2021
from __future__ import print_function, division

import numpy as np
cimport numpy as np

from libcpp cimport bool

from ..cr           import cr
from ..mem          import mem
from ..utils.common import raiseError

# Declare the class with cdef
cdef extern from "geometry.h" namespace "Geom":
	# Point class
	cdef cppclass CPoint "Geom::Point":
		CPoint() except +
		CPoint(const double, const double, const double) except +
		void    set(const int i, const double val)
		void    set(const double *v)
		double  get(const int i)
		double  x()
		double  y()
		double  z()
		double  dist(const CPoint &pp) const
		double  dist2(const CPoint &pp) const
		double isLeft(const CPoint &P0, const CPoint &P1) const
	# Vector class
	cdef cppclass CVector "Geom::Vector":
		CVector() except +
		CVector(const double x, const double y, const double z) except +
		CVector(const double *p) except +
		void    set(const int i, const double val)
		void    set(const double *p)
		double  get(const int i)
		double  x()
		double  y()
		double  z()
		double  dot(const CVector &vv) const
		CVector cross(const CVector &vv) const
		double  norm() const
		double  norm2() const
	# Ball class
	cdef cppclass CBall "Geom::Ball":
		CBall() except +
		CBall(const double r, const CPoint &p) except +
		CBall(const CPoint &p, const double r) except +
		CBall(const CPolygon &p) except +
		CPoint get_center() const
		double get_radius() const
		void   set_center(const CPoint &p)
		void   set_radius(const double r)
		bool isempty() const
		bool isinside(const CPoint &p) const
		bool isdisjoint(const CBall &b) const
	# Polygon class
	cdef cppclass CPolygon "Geom::Polygon":
		CPolygon() except +
		CPolygon(const int nn) except +
		CPolygon(const int nn, const CPoint &v) except +
		CPolygon(const int nn, const CPoint *v) except +
		void    set_npoints(const int nn)
		void    set_point(const int i, const CPoint &v)
		void    set_points(const CPoint v)
		void    set_points(const CPoint *v)
		void    set_centroid(const CPoint v)
		void    set_bbox(CBall &b)
		void    clear()
		void    set(const int nn, const CPoint &v)
		void    set(const int nn, const CPoint *v)
		CPoint *get_points() const
		CPoint  get_point(const int i) const
		CPoint  get_centroid() const
		int     get_npoints() const
		CBall   get_bbox() const
		bool    isempty() const
		bool    isinside(const CPoint &v) const
		bool    isinside_cn(const CPoint &v) const
		bool    isinside_wn(const CPoint &v) const
		CPoint  compute_centroid()
		void    rotate(const double theta[3], const CPoint o);
		void    rotate_rodrigues(const double theta, const double k[3], const CPoint o);


# Class wrapping a point
cdef class Point:
	'''
	A simple 3D point.
	'''
	cdef CPoint _point
#	@mem('Geom.Point')
	def __init__(Point self,double x, double y, double z):
		self._point = CPoint(x,y,z)

	def __str__(Point self):
		return '[ %f %f %f ]' % (self.x,self.y,self.z)

	# Operators
	def __getitem__(Point self,int i):
		'''
		Point[i]
		'''
		return self._point.get(i)

	def __setitem__(Point self,int i,double value):
		'''
		Point[i] = value
		'''
		self._point.set(i,value)

	def __add__(Point self,Vector other):
		'''
		Point = Point + Vector
		'''
		return Point(self.x+other.x,self.y+other.y,self.z+other.z)

	def __sub__(Point self,object other):
		'''
		Point  = Point - Vector
		Vector = Point - Point
		'''
		if isinstance(other,Vector):
			return Point(self.x-other.x,self.y-other.y,self.z-other.z)
		if isinstance(other,Point):
			return Vector(self.x-other.x,self.y-other.y,self.z-other.z)
		raiseError('Unknown instance in Point subtraction!')

	def __eq__(Point self,Point other):
		'''
		Point == Point
		'''
		return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)

	def __ne__(Point self,Point other):
		'''
		Point != Point
		'''
		return not self == other

	# Functions
	def dist(Point self,Point p):
		'''
		Distance between two points
		'''
		return self._point.dist(p._point)

	def dist2(Point self,Point p):
		'''
		Distance between two points squared
		'''
		return self._point.dist2(p._point)

	def isLeft(Point self,Point p1,Point p2):
		'''
		ISLEFT

		Tests if a point is Left|On|Right of an infinite line.

		Input:  two points P1, P2; defining a line
		Return: >0 for P2 left of the line through P0 and P1
				=0 for P2  on the line
				<0 for P2  right of the line
		
		from: http://geomalgorithms.com/a03-_inclusion.html
		
		Copyright 2001 softSurfer, 2012 Dan Sunday
		This code may be freely used and modified for any purpose
		providing that this copyright notice is included with it.
		SoftSurfer makes no warranty for this code, and cannot be held
		liable for any real or imagined damage resulting from its use.
		Users of this code must verify correctness for their application.
		'''
		cdef double out = self._point.isLeft(p1._point,p2._point)
		return out

	@staticmethod
	def areLeft(double[:,:] xyz,Point p1,Point p2):
		'''
		ARELEFT

		Tests if a set of points are Left|On|Right of an infinite line.

		Input:  two points P1, P2; defining a line
		Return: >0 for P2 left of the line through P0 and P1
				=0 for P2  on the line
				<0 for P2  right of the line
		
		from: http://geomalgorithms.com/a03-_inclusion.html
		
		Copyright 2001 softSurfer, 2012 Dan Sunday
		This code may be freely used and modified for any purpose
		providing that this copyright notice is included with it.
		SoftSurfer makes no warranty for this code, and cannot be held
		liable for any real or imagined damage resulting from its use.
		Users of this code must verify correctness for their application.
		'''
		cdef int ii, npoints = xyz.shape[0]
		cdef double x1 = p1.x, x2 = p2.x, y1 = p1.y, y2 = p2.y
		cdef np.ndarray[np.double_t,ndim=1] out = np.ndarray((npoints,),dtype=np.double)
		for ii in range(npoints):
			out[ii] = ( (x2 - x1)*(xyz[ii,1] - y1) - (xyz[ii,0] - x1)*(y2 - y1) )
		return out

	@classmethod
	def from_array(Point cls,double[:] xyz):
		'''
		Build a point from an xyz array of shape (3,)
		'''
		return cls(xyz[0],xyz[1],xyz[2])

	@property
	def x(Point self):
		return self._point.x()
	@property
	def y(Point self):
		return self._point.y()
	@property
	def z(Point self):
		return self._point.z()
	@property
	def xyz(Point self):
		cdef np.ndarray[np.double_t,ndim=1] _xyz = np.ndarray((3,),dtype=np.double)
		_xyz[0] = self._point.x()
		_xyz[1] = self._point.y()
		_xyz[2] = self._point.z()
		return _xyz
	@xyz.setter
	def xyz(Point self,double[:] value):
		self._point.set(&value[0])


# Class wrapping a vector
cdef class Vector:
	'''
	A simple 3D vector.
	'''
	cdef CVector _vector
#	@mem('Geom.Vector')
	def __init__(Vector self, double x, double y, double z):
		self._vector = CVector(x,y,z)

	def __str__(Vector self):
		return '( %f %f %f )' % (self.x,self.y,self.z)

	# Operators
	def __getitem__(Vector self,int i):
		'''
		Point[i]
		'''
		return self._vector.get(i)

	def __setitem__(Vector self,int i,double value):
		'''
		Point[i] = value
		'''
		self._vector.set(i,value)

	def __add__(Vector self,Vector other):
		'''
		Vector = Vector + Vector
		'''
		return Vector(self.x+other.x,self.y+other.y,self.z+other.z)

	def __sub__(Vector self,Vector other):
		'''
		Vector = Vector - Vector
		'''
		return Vector(self.x-other.x,self.y-other.y,self.z-other.z)

	def __mul__(Vector self,object other):
		'''
		Vector = Vector*val
		val    = Vector*Vector
		'''
		if isinstance(other,Vector):
			return self.dot(other)
		else:
			return Vector(other*self.x,other*self.y,other*self.z)

	def __rmul__(Vector self,object other):
		'''
		Vector = val*Vector
		val    = Vector*Vector
		'''
		return self.__mul__(other)

	def __truediv__(Vector self,double other):
		'''
		Vector = Vector/val
		'''
		return Vector(self.x/other,self.y/other,self.z/other)

	def __eq__(Vector self,Vector other):
		'''
		Vector == Vector
		'''
		return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)

	def __ne__(Vector self,Vector other):
		'''
		Vector != Vector
		'''
		return not self == other

	# Functions
	def dot(Vector self,Vector v):
		'''
		Dot product
		'''
		cdef double out = self._vector.dot(v._vector)
		return out
	
	def cross(Vector self,Vector v):
		'''
		Cross product
		'''
		cdef Vector out = Vector(0.,0.,0.)
		out._vector = self._vector.cross(v._vector)
		return out

	def norm(Vector self):
		'''
		Vector norm
		'''
		cdef double out = self._vector.norm()
		return out

	def norm2(Vector self):
		'''
		Vector norm squared
		'''
		cdef double out = self._vector.norm2()
		return out

	@property
	def x(Vector self):
		return self._vector.x()
	@property
	def y(Vector self):
		return self._vector.y()
	@property
	def z(Vector self):
		return self._vector.z()
	@property
	def xyz(Vector self):
		cdef np.ndarray[np.double_t,ndim=1] _xyz = np.ndarray((3,),dtype=np.double)
		_xyz[0] = self._vector.x()
		_xyz[1] = self._vector.y()
		_xyz[2] = self._vector.z()
		return _xyz
	@xyz.setter
	def xyz(Vector self,double[:] value):
		self._vector.set(&value[0])


# Class wrapping a ball
cdef class Ball:
	'''
	A 2D circle or a 3D sphere wrapped in a single class
	'''
	cdef CBall _ball
#	@mem('Geom.Ball')
	def __init__(Ball self, Point center = Point(0.,0.,0.), double radius = 0.):
		self._ball = CBall(center._point,radius)

	def __str__(Ball self):
		return 'center = ' + self.center.__str__() + ' radius = %f' % (self.radius)

	# Operators
	def __eq__(Ball self,Ball other):
		'''
		Ball == Ball
		'''
		return self.center == other.center and self.radius == other.radius

	def __gt__(Ball self,object other):
		'''
		self.isinside(other)
		'''
		if isinstance(other,Point):
			return self.isinside(other)
		else:
			return self.areinside(other)

	def __lt__(Ball self,object other):
		'''
		not self.isinside(other)
		'''
		if isinstance(other,Point):
			return not self.isinside(other)
		else:
			return np.logical_not(self.areinside(other))

	# Functions
	def isempty(Ball self):
		cdef bool out = self._ball.isempty()
		return out
	
	def isinside(Ball self,Point point):
		cdef bool out = self._ball.isinside(point._point)
		return out

	def areinside(Ball self,double[:,:] xyz):
		cdef int ii, npoints = xyz.shape[0]
		cdef CPoint p
		cdef np.ndarray[np.npy_bool,ndim=1,cast=True] out = np.ndarray((npoints,),dtype=np.bool_)
		for ii in range(npoints):
			p       = CPoint(xyz[ii,0],xyz[ii,1],xyz[ii,2])
			out[ii] = self._ball.isinside(p)
		return out

	def isdisjoint(Ball self,Ball ball):
		cdef bool out = self._ball.isdisjoint(ball._ball)
		return out

	@classmethod
	def fastBall(Ball cls,Polygon poly):
		'''
		FASTBALL

		Get a fast approximation for the 2D bounding ball 
		(based on the algorithm given by [Jack Ritter, 1990]).

		Input:  A polygon
		Output: Nothing, sets the ball class

		from: http://geomalgorithms.com/a08-_containers.html

		Copyright 2001 softSurfer, 2012 Dan Sunday
		This code may be freely used and modified for any purpose
		providing that this copyright notice is included with it.
		SoftSurfer makes no warranty for this code, and cannot be held
		liable for any real or imagined damage resulting from its use.
		Users of this code must verify correctness for their application.
		'''
		cdef Ball out = cls(Point(0.,0.,0.),0.)
		out._ball = CBall(poly._poly)
		return out

	@property
	def center(Ball self):
		cdef Point out = Point(0.,0.,0.)
		out._point = self._ball.get_center()
		return out
	@center.setter
	def center(Ball self, Point value):
		self._ball.set_center(value._point)
	@property
	def radius(Ball self):
		cdef double out = self._ball.get_radius()
		return out
	@radius.setter
	def radius(Ball self, double value):
		self._ball.set_radius(value)

# Wrapper class for polygon
cdef class Polygon:
		'''
		A polygon set as an array of points. Can be either 2D or 3D.
		'''
		cdef CPolygon _poly
		cdef Point[:] _points
		cdef Point    _centroid
#		@mem('Geom.Polygon')
		def __init__(Polygon self,Point[:] points):
			cdef int ip, npoints = points.shape[0]
			self._poly.set_npoints(npoints) # Already allocates npoints + 1!
			self._points = points
			for ip in range(npoints):
				self._poly.set_point(ip,points[ip]._point)
			self._poly.set_point(npoints,points[0]._point)
			# Set boundig box
			cdef CBall bbox = CBall(self._poly)
			self._poly.set_bbox(bbox)
			# Compute and set centroid
			self._centroid = self.compute_centroid()
			self._poly.set_centroid(self._centroid._point)

		def __dealloc__(Polygon self):
			self._poly.clear()

		def __str__(Polygon self):
			cdef int ip = 0
			cdef object retstr = 'Point %d %s' % (ip,self.points[ip].__str__())
			for ip in range(1,self.npoints):
				retstr += '\nPoint %d %s' % (ip,self.points[ip].__str__())
			return retstr

		# Operators
		def __getitem__(Polygon self,int i):
			'''
			Polygon[i]
			'''
			cdef Point out = Point(0.,0.,0.)
			out._point = self._poly.get_point(i)
			return out

		def __setitem__(Polygon self,int i,Point value):
			'''
			Polygon[i] = value
			'''
			self._poly.set_point(i,value._point)

		def __eq__(Polygon self,Polygon other):
			'''
			Polygon == Polygon
			'''
			cdef int ip, np1= self.npoints, np2 = other.npoints
			# Check if polygons have the same number of points
			if not np1 == np2:
				return False
			# Check if the points are equal
			for ip in range(np1):
				if not self[ip] == other[ip]:
					return False
			return True

		def __ne__(Polygon self,Polygon other):
			'''
			Polygon != Polygon
			'''
			return not self.__eq__(other)

		def __gt__(Polygon self,object other):
			'''
			self.isinside(other)
			'''
			if isinstance(other,Point):
				return self.isinside(other) # Return true if Point inside Polygon
			else: # Assume numpy array
				return self.areinside(other)

		def __lt__(Polygon self,object other):
			'''
			not self.isinside(other)
			'''
			if isinstance(other,Point):
				return not self.isinside(other)
			else:
				return np.logical_not(self.areinside(other))

		# Functions
		def isempty(Polygon self):
			cdef bool out = self._poly.isempty()
			return out

		@cr('Geom.Poly.isinside')
		def isinside(Polygon self,Point point):
			'''
			Returns True if the point is inside the polygon, else False.
			'''
			cdef bool out = self._poly.isinside(point._point)
#			cdef bool out =  self._poly.isinside_cn(point._point)
#			cdef bool out =  self._poly.isinside_wn(point._point)
			return out

		@cr('Geom.Poly.areinside')
		def areinside(Polygon self,double[:,:] xyz):
			'''
			Returns True if the points are inside the polygon, else False.
			'''
			cdef int ii, npoints = xyz.shape[0]
			cdef CPoint p
			cdef np.ndarray[np.npy_bool,ndim=1,cast=True] out = np.ndarray((npoints,),dtype=np.bool_)
			for ii in range(npoints):
				p       = CPoint(xyz[ii,0],xyz[ii,1],xyz[ii,2])
				out[ii] = self._poly.isinside(p)
			return out

		@cr('Geom.Poly.centroid')
		def compute_centroid(Polygon self):
			'''
			Returns the centroid (Point) of a (2D) polygon.
			3D version to be implemented.

			https://wwwf.imperial.ac.uk/~rn/centroid.pdf
			https://en.wikipedia.org/wiki/Centroid
			'''
			cdef Point out = Point(0.,0.,0.)
			out._point = self._poly.compute_centroid()
			return out

		@cr('Geom.Poly,rotate')
		def rotate(Polygon self, double[:] theta, double[:] o=np.array([])):
			'''
			Rotate a polygon by a theta radians 3D angle array
			wrt to an origin Point (o).
			'''
			# Input must be a 3D angle
			if theta.shape[0] != 3:
				raiseError('Rotation does not contain a 3D angle')
			# Declare variables			
			cdef int ip
			cdef CPoint p = self._centroid._point if o.shape[0] == 0 else CPoint(o[0],o[1],o[2])
			# Compute the rotation
			self._poly.rotate(&theta[0],p)
			# Update the points
			for ip in range(self._poly.get_npoints()):
				self._points[ip]._point = self._poly.get_point(ip)
			return self

		@cr('Geom.Poly.rot_rodrig')
		def rotate_rodrigues(Polygon self, double theta, double[:] k, double[:] o=np.array([])):
			'''
			Rotate a polygon by a theta radians around a 3-D unit vector axis (k).
			This implements the general matrix rotation formula by Rodrigues:
			https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
			'''
			# Input must be a scalar angle and 3-D non-zero axis
			if k.shape[0] != 3:
				raiseError('Rotation axis is not a 3-D vector')
			if (k[0] == 0.) and (k[1] == 0.) and (k[2] == 0.):
				raiseError('Rotation axis vector components are all 0')
			# Define variables
			cdef int ip
			cdef CPoint p = self._centroid._point if o.shape[0] == 0 else CPoint(o[0],o[1],o[2])
			# Compute the rotation
			self._poly.rotate_rodrigues(theta,&k[0],p)
			# Update the points
			for ip in range(self._poly.get_npoints()):
				self._points[ip]._point = self._poly.get_point(ip)
			return self

		@classmethod
		def from_array(Polygon cls,double[:,:] xyz):
			'''
			Build a polygon from an array of points
			of shape (npoints,3).
			'''
			cdef int ii, npoints = xyz.shape[0]
			pointList = np.array([Point.from_array(xyz[ii,:]) for ii in range(npoints)],dtype=Point)
			return cls(pointList)

		@property
		def npoints(Polygon self):
			return self._poly.get_npoints() # Returns correctly
		@property
		def points(Polygon self):
			return self._points
		@points.setter
		def points(Polygon self,Point[:] value):
			cdef int ip
			cdef npoints = value.shape[0]
			self._poly.set_npoints(npoints) # Already allocates npoints + 1!
			self._points = value
			for ip in range(npoints):
				self._poly.set_point(ip,value[ip]._point)
			self._poly.set_point(npoints,value[0]._point)
		@property
		def bbox(Polygon self):
			cdef Ball out = Ball()
			out._ball = self._poly.get_bbox()
			return out
		@bbox.setter
		def bbox(Polygon self,Ball value):
			self._poly.set_bbox(value._ball)
		@property
		def centroid(Polygon self):
			return self._centroid
		@centroid.setter
		def centroid(Polygon self,Point value):
			self._centroid = value
			self._poly.set_centroid(self._centroid._point)
		@property
		def xyz(Polygon self):
			cdef int ii, npoints = self.npoints+1
			cdef np.ndarray[np.double_t,ndim=2] out = np.ndarray((npoints,3),dtype=np.double)
			for ii in range(npoints):
				out[ii,0] = self._poly.get_point(ii).x()
				out[ii,1] = self._poly.get_point(ii).y()
				out[ii,2] = self._poly.get_point(ii).z()
			return out			
		@property
		def x(Polygon self):
			cdef int ii, npoints = self.npoints+1
			cdef np.ndarray[np.double_t,ndim=1] out = np.ndarray((npoints,),dtype=np.double)
			for ii in range(npoints):
				out[ii] = self._poly.get_point(ii).x()
			return out
		@property
		def y(Polygon self):
			cdef int ii, npoints = self.npoints+1
			cdef np.ndarray[np.double_t,ndim=1] out = np.ndarray((npoints,),dtype=np.double)
			for ii in range(npoints):
				out[ii] = self._poly.get_point(ii).y()
			return out
		@property
		def z(Polygon self):
			cdef int ii, npoints = self.npoints+1
			cdef np.ndarray[np.double_t,ndim=1] out = np.ndarray((npoints,),dtype=np.double)
			for ii in range(npoints):
				out[ii] = self._poly.get_point(ii).z()
			return out