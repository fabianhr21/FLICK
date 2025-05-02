#!/usr/bin/env python
#
# pyQvarsi, GEOM entities.
#
# Small geometric module to cut a field or a mesh.
#
# Basic entities and operators.
#
# Last rev: 11/01/2021
from __future__ import print_function, division

import numpy as np

from ..cr           import cr
from ..mem          import mem
from ..utils.common import raiseError


class Point(object):
	'''
	A simple 3D point.
	'''
#	@mem('Geom.Point')
	def __init__(self, x, y , z):
		self._xyz = np.array([x,y,z])

	def __str__(self):
		return '[ %f %f %f ]' % (self.x,self.y,self.z)

	# Operators
	def __getitem__(self,i):
		'''
		Point[i]
		'''
		return self._xyz[i]

	def __setitem__(self,i,value):
		'''
		Point[i] = value
		'''
		self._xyz[i] = value

	def __add__(self,other):
		'''
		Point = Point + Point
		Point = Point + Vector
		'''
		if isinstance(other,Point) or isinstance(other,Vector):
			return Point(self.x+other.x,self.y+other.y,self.z+other.z)
		raiseError('Only Point + Point or Point + Vector is allowed!')

	def __sub__(self,other):
		'''
		Point  = Point - Vector
		Vector = Point - Point
		'''
		if isinstance(other,Vector):
			return Point(self.x-other.x,self.y-other.y,self.z-other.z)
		if isinstance(other,Point):
			return Vector(self.x-other.x,self.y-other.y,self.z-other.z)
		raiseError('Unknown instance in Point subtraction!')

	def __eq__(self,other):
		'''
		Point == Point
		'''
		if not isinstance(other,Point):
			raiseError('Only Point == Point is allowed!')
		return ( (self.x == other.x) and (self.y == other.y) and (self.z == other.z) )

	def __ne__(self,other):
		'''
		Point != Point
		'''
		return not self == other

	# Functions
	def dist(self,p):
		'''
		Distance between two points
		'''
		v = self - p # Vector as the difference between two points
		return v.norm()

	def dist2(self,p):
		'''
		Distance between two points squared
		'''
		v = self - p # Vector as the difference between two points
		return v.norm2()

	def isLeft(self,p1,p2):
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
		return ( (p2.x - p1.x)*(self.y - p1.y) - (self.x - p1.x)*(p2.y - p1.y) )

	@staticmethod
	def areLeft(xyz,p1,p2):
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
		npoints = xyz.shape[0]
		p1x     = np.tile(p1.x,(npoints,))	
		p2x     = np.tile(p2.x,(npoints,))	
		p1y     = np.tile(p1.y,(npoints,))	
		p2y     = np.tile(p2.y,(npoints,))	
		return ( (p2x - p1x)*(xyz[:,1] - p1y) - (xyz[:,0] - p1x)*(p2y - p1y) )


	@classmethod
	def from_array(cls,xyz):
		'''
		Build a point from an xyz array of shape (3,)
		'''
		return cls(xyz[0],xyz[1],xyz[2])

	@property
	def x(self):
		return self._xyz[0]
	@property
	def y(self):
		return self._xyz[1]
	@property
	def z(self):
		return self._xyz[2]
	@property
	def xyz(self):
		return self._xyz
	@xyz.setter
	def xyz(self,value):
		self._xyz = value


class Vector(object):
	'''
	A simple 3D vector.
	'''
#	@mem('Geom.Vector')
	def __init__(self, x, y, z):
		self._xyz = np.array([x,y,z])

	def __str__(self):
		return '( %f %f %f )' % (self.x,self.y,self.z)

	# Operators
	def __getitem__(self,i):
		'''
		Point[i]
		'''
		return self._xyz[i]

	def __setitem__(self,i,value):
		'''
		Point[i] = value
		'''
		self._xyz[i] = value

	def __add__(self,other):
		'''
		Vector = Vector + Vector
		'''
		if not isinstance(other,Vector):
			raiseError('Only Vector + Vector is allowed!')
		return Vector(self.x+other.x,self.y+other.y,self.z+other.z)

	def __sub__(self,other):
		'''
		Vector = Vector - Vector
		'''
		if not isinstance(other,Vector):
			raiseError('Only Vector - Vector is allowed!')
		return Vector(self.x-other.x,self.y-other.y,self.z-other.z)

	def __mul__(self,other):
		'''
		Vector = Vector*val
		val    = Vector*Vector
		'''
		if isinstance(other,Vector):
			return self.dot(other)
		else:
			return Vector(other*self.x,other*self.y,other*self.z)

	def __rmul__(self,other):
		'''
		Vector = val*Vector
		val    = Vector*Vector
		'''
		return self.__mul__(other)

	def __truediv__(self,other):
		'''
		Vector = Vector/val
		'''
		return Vector(self.x/other,self.y/other,self.z/other)

	def __eq__(self,other):
		'''
		Vector == Vector
		'''
		if not isinstance(other,Vector):
			raiseError('Only Vector == Vector is allowed!')
		return ( (self.x == other.x) and (self.y == other.y) and (self.z == other.z) )

	def __ne__(self,other):
		'''
		Vector != Vector
		'''
		return not self == other

	# Functions
	def dot(self,v):
		'''
		Dot product
		'''
		return (self.x*v.x + self.y*v.y + self.z*v.z)
	
	def cross(self,v):
		'''
		Cross product
		'''
		return Vector(self.y*v.z-self.z*v.y,-self.x*v.z+self.z*v.x,self.x*v.y-self.y*v.x)

	def norm(self):
		'''
		Vector norm
		'''
		return np.sqrt(self.norm2())

	def norm2(self):
		'''
		Vector norm squared
		'''
		return self.dot(self)

	@property
	def x(self):
		return self._xyz[0]
	@property
	def y(self):
		return self._xyz[1]
	@property
	def z(self):
		return self._xyz[2]
	@property
	def xyz(self):
		return self._xyz
	@xyz.setter
	def xyz(self,value):
		self._xyz = value


class Ball(object):
	'''
	A 2D circle or a 3D sphere wrapped in a single class
	'''
#	@mem('Geom.Ball')
	def __init__(self, center = Point(0.,0.,0.), radius = 0):
		self._center = center
		self._radius = radius

	def __str__(self):
		return 'center = ' + self.center.__str__() + ' radius = %f' % (self.radius)

	# Operators
	def __eq__(self,other):
		'''
		Ball == Ball
		'''
		if not isinstance(other,Ball):
			raiseError('Only Ball == Ball is allowed!')
		return self.center == other.center and self.radius == other.radius

	def __gt__(self, other):
		'''
		self.isinside(other)
		'''
		if isinstance(other,Point):
			return self.isinside(other)
		else:
			return self.areinside(other)

	def __lt__(self,other):
		'''
		not self.isinside(other)
		'''
		if isinstance(other,Point):
			return not self.isinside(other)
		else:
			return np.logical_not(self.areinside(other))

	# Functions
	def isempty(self):
		return self._radius == 0
	
	def isinside(self,point):
		return True if not self.isempty() and point.dist(self.center) < self.radius else False

	def areinside(self,xyz):
		vec  = xyz - np.tile(self.center.xyz,(xyz.shape[0],1))
		dist = np.sqrt(np.sum(vec*vec,axis=1))
		return dist < self.radius if not self.isempty() else np.zeros((xyz.shape[0],),dtype=bool)

	def isdisjoint(self,ball):
		return True if not self.isempty() and ball.center.dist(self.center) < self.radius + ball.radius else False

	@classmethod
	def fastBall(cls,poly):
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
		# Find a large diameter to start with
		# first get the bounding box and the extreme points
		xpoints = poly.x
		ypoints = poly.y
		zpoints = poly.z

		idx_min, idx_max = np.argmin(xpoints), np.argmax(xpoints)
		idy_min, idy_max = np.argmin(ypoints), np.argmax(ypoints)
		idz_min, idz_max = np.argmin(zpoints), np.argmax(zpoints)

		# Select the largest extent as an initial diameter for the  ball
		center  = Point(0.,0.,0.)
		dPx     = poly[idx_max] - poly[idx_min]
		dPy     = poly[idy_max] - poly[idy_min]
		dPz     = poly[idz_max] - poly[idz_min]
		radius2 = 0.

		if   dPx.norm2() >= dPy.norm2() and dPx.norm2() >= dPz.norm2(): # x direction is largest extent
			center  = poly[idx_min] + dPx/2.
			radius2 = poly[idx_max].dist2(center)
		elif dPy.norm2() >= dPx.norm2() and dPy.norm2() >= dPz.norm2(): # y direction is largest extent
			center  = poly[idy_min] + dPy/2.
			radius2 = poly[idy_max].dist2(center)
		else: 															# z direction is largest extent
			center  = poly[idz_min] + dPz/2.
			radius2 = poly[idz_max].dist2(center)

		radius = np.sqrt(radius2)

		# Now check that all points p[i] are in the ball
		# and if not, expand the ball just enough to include them
		for p in poly.points:
			dP    = p - center
			dist2 = dP.norm2()
			if dist2 <= radius2: continue # The point is inside the ball already
			# p not in ball, so expand ball to include it
			dist    = np.sqrt(dist2)
			radius  = 0.5*(radius + dist)              # enlarge radius just enough
			radius2 = radius*radius
			center  = center + ((dist-radius)/dist)*dP # shift center towards p

		# Return the ball
		return cls(center,radius)

	@property
	def center(self):
		return self._center
	@center.setter
	def center(self,value):
		self._center = value
	@property
	def radius(self):
		return self._radius
	@radius.setter
	def radius(self,value):
		self._radius = value


class Polygon(object):
		'''
		A polygon set as an array of points. Can be either 2D or 3D.
		'''
#		@mem('Geom.Polygon')
		def __init__(self, points):
			self._points   = np.hstack((points,points[0]))
			self._bbox     = Ball.fastBall(self) # Create a ball bounding box using fastBall
			self._centroid = self.compute_centroid()

		def __str__(self):
			retstr = 'Point %d %s' % (0,self.points[0].__str__())
			for ip in range(1,self.npoints):
				retstr += '\nPoint %d %s' % (ip,self.points[ip].__str__())
			return retstr

		# Operators
		def __getitem__(self,i):
			'''
			Polygon[i]
			'''
			return self._points[i]

		def __setitem__(self,i,value):
			'''
			Polygon[i] = value
			'''
			self._points[i] = value

		def __eq__(self,other):
			'''
			Polygon == Polygon
			'''
			# Check if polygons have the same number of points
			if not self.npoints == other.npoints:
				return False
			# Check if the points are equal
			for ip in range(self.npoints):
				if not self[ip] == other[ip]:
					return False
			return True

		def __ne__(self,other):
			'''
			Polygon != Polygon
			'''
			return not self.__eq__(other)

		def __gt__(self, other):
			'''
			self.isinside(other)
			'''
			if isinstance(other,Point):
				return self.isinside(other) # Return true if Point inside Polygon
			else: # Assume numpy array
				return self.areinside(other)

		def __lt__(self,other):
			'''
			not self.isinside(other)
			'''
			if isinstance(other,Point):
				return not self.isinside(other)
			else:
				return np.logical_not(self.areinside(other))

		# Functions
		def isempty(self):
			return self.npoints == 0

		@cr('Geom.Poly.isinside')
		def isinside(self,point,algorithm='wn'):
			'''
			Returns True if the point is inside the polygon, else False.
			'''
			if self.bbox > point: # Point is inside the bounding box
				# Select the algorithm to use
				if algorithm == 'wn':
					return True if wn_PinPoly(point,self) > 0  else False
				else:
					return True if cn_PinPoly(point,self) == 1 else False
			else:
				return False

		@cr('Geom.Poly.areinside')
		def areinside(self,xyz,algorithm='wn'):
			'''
			Returns True if the points are inside the polygon, else False.
			'''
			out = np.zeros((xyz.shape[0],),dtype=bool)
			idx = self.bbox > xyz   # Point are inside the bounding box

			out[idx] = wn_PinPoly_vec(xyz[idx],self) != 0 if algorithm == 'wn' else cn_PinPoly_vec(xyz[idx],self) == 1

			return out

		@cr('Geom.Poly.centroid')
		def compute_centroid(self):
			'''
			Returns the centroid (Point) of a (2D) polygon.	
			3D version to be implemented.

			https://wwwf.imperial.ac.uk/~rn/centroid.pdf
			https://en.wikipedia.org/wiki/Centroid
			'''
			Cx, Cy, A = 0, 0, 0
			for ip in range(self.npoints):
				Cx += (self[ip  ][0] + self[ip+1][0]) * \
				      (self[ip  ][0] * self[ip+1][1] -
				       self[ip+1][0] * self[ip  ][1])
				Cy += (self[ip  ][1] + self[ip+1][1]) * \
				      (self[ip  ][0] * self[ip+1][1] -
				       self[ip+1][0] * self[ip  ][1])
				A +=   self[ip  ][0] * self[ip+1][1] - \
				       self[ip+1][0] * self[ip  ][1]
			return Point(Cx/(3*A),Cy/(3*A),0.)

		@cr('Geom.Poly.rotate')
		def rotate(self, theta, o=np.array([])):
			'''
			Rotate a polygon by a theta radians 3D angle array 
			wrt to an origin Point (o).
			'''
			# Input must be a 3D angle
			if len(theta) != 3:
				raiseError('Rotation does not contain a 3D angle')
			o = self.centroid if o.size == 0 else Point.from_array(o)
			# Compute sin and cos
			cx, sx = np.cos(theta[0]), np.sin(theta[0])
			cy, sy = np.cos(theta[1]), np.sin(theta[1])
			cz, sz = np.cos(theta[2]), np.sin(theta[2])
			# Rotation matrices
			Rx = np.array([[   1,   0,   0],
			               [   0,  cx, -sx],
			               [   0,  sx,  cx]])
			Ry = np.array([[  cy,   0,  sy],
			               [   0,   1,   0],
			               [ -sy,   0,  cy]])
			Rz = np.array([[  cz, -sz,   0],
			               [  sz,  cz,   0],
			               [   0,   0,   1]])
			# Compute rotation matrix R
			R = np.matmul(Rx,np.matmul(Ry,Rz))
			# Project the points
			for ip in range(self.npoints): # Make sure to ge the last one too
				self._points[ip].xyz = np.matmul(R,self._points[ip].xyz - o.xyz) + o.xyz
			return self

		@cr('Geom.Poly.rot_rodrig')
		def rotate_rodrigues(self, theta, k, o=np.array([])):
			'''
			Rotate a polygon by a theta radians around a 3-D unit vector axis (k).
			This implements the general matrix rotation formula by Rodrigues:
			https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
			'''
			# Input must be a scalar angle and 3-D non-zero axis
			if len(k) != 3:
				raiseError('Rotation axis is not a 3-D vector')
			elif np.all(k==0):
				raiseError('Rotation axis vector components are all 0')
			elif not np.isscalar(theta):
				raiseError('Rotation angle is not scalar')
			o = self.centroid if o.size == 0 else Point.from_array(o)

			# Compute rotation matrix R
			K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
			I = np.eye(3,3)
			R = I + np.sin(theta)*K + (1-np.cos(theta))*(np.matmul(K,K))

			# Project the points
			for ip in range(self.npoints): # Make sure to ge the last one too
				self._points[ip].xyz = np.matmul(R,self._points[ip].xyz - o.xyz) + o.xyz
			return self

		@classmethod
		def from_array(cls,xyz):
			'''
			Build a polygon from an array of points
			of shape (npoints,3).
			'''
			pointList = [Point.from_array(xyz[ii,:]) for ii in range(xyz.shape[0])] 
			return cls(pointList)

		@property
		def npoints(self):
			return len(self._points) - 1
		@property
		def points(self):
			return self._points
		@points.setter
		def points(self,value):
			self._points = value
		@property
		def bbox(self):
			return self._bbox
		@bbox.setter
		def bbox(self,value):
			self._bbox = value
		@property
		def centroid(self):
			return self._centroid
		@centroid.setter
		def centroid(self,value):
			self._centroid = value
		@property
		def xyz(self):
			return np.array([[p.x,p.y,p.z] for p in self._points])
		@property
		def x(self):
			return np.array([p.x for p in self._points])
		@property
		def y(self):
			return np.array([p.y for p in self._points])
		@property
		def z(self):
			return np.array([p.z for p in self._points])


def cn_PinPoly(point, poly):
	'''
	CN_PINPOLY

	2D algorithm.
	Crossing number test for a point in a polygon.

	Input:   P = a point,
	Return:  0 = outside, 1 = inside

	This code is patterned after [Franklin, 2000]
	from: http://geomalgorithms.com/a03-_inclusion.html

	Copyright 2001 softSurfer, 2012 Dan Sunday
	This code may be freely used and modified for any purpose
	providing that this copyright notice is included with it.
	SoftSurfer makes no warranty for this code, and cannot be held
	liable for any real or imagined damage resulting from its use.
	Users of this code must verify correctness for their application.
	'''
	cn = 0 # The crossing number counter
	# Loop through all edges of the Polygon
	for ip in range(poly.npoints): 
		# an upward crossing or a downward crossing
		if ( (poly[ip][1] <= point[1]) and (poly[ip+1][1] >  point[1]) ) or \
		   ( (poly[ip][1] >  point[1]) and (poly[ip+1][1] <= point[1]) ):
			# Compute  the actual edge-ray intersect x-coordinate
			vt = (point[1] - poly[ip][1])/(poly[ip+1][1] - poly[ip][1])
					
			if point[0] <  poly[ip][0] + vt * (poly[ip+1][0] - poly[ip][0]): # P.x < intersect
				cn += 1 # A valid crossing of y=P.y right of P.x
	return not cn%2 == 0 # 0 if even (out), and 1 if  odd (in)

def cn_PinPoly_vec(xyz, poly):
	'''
	CN_PINPOLY

	2D algorithm.
	Crossing number test for a point in a polygon.

	Input:   xyz = an array of points,
	Return:  0 = outside, 1 = inside

	This code is patterned after [Franklin, 2000]
	from: http://geomalgorithms.com/a03-_inclusion.html

	Copyright 2001 softSurfer, 2012 Dan Sunday
	This code may be freely used and modified for any purpose
	providing that this copyright notice is included with it.
	SoftSurfer makes no warranty for this code, and cannot be held
	liable for any real or imagined damage resulting from its use.
	Users of this code must verify correctness for their application.
	'''
	npoints = xyz.shape[0]
	cn = np.zeros((npoints,)) # The crossing number counter
	# Loop through all edges of the Polygon
	for ip in range(poly.npoints): 
		vt   = np.zeros((npoints,))
		idx2 = np.zeros((npoints,),dtype=bool)
		# an upward crossing or a downward crossing
		ip_poly_tile0  = np.tile(poly[ip][0],(npoints,))
		ip_poly_tile1  = np.tile(poly[ip][1],(npoints,))
		ip1_poly_tile0 = np.tile(poly[ip+1][0],(npoints,))
		ip1_poly_tile1 = np.tile(poly[ip+1][1],(npoints,))
		
		idx1 = np.logical_or( np.logical_and(ip_poly_tile1 <= xyz[:,1],ip1_poly_tile1 >  xyz[:,1]),
			np.logical_and(ip_poly_tile1 >  xyz[:,1],ip1_poly_tile1 <= xyz[:,1]) )
		
		# Compute  the actual edge-ray intersect x-coordinate
		vt[idx1] = (xyz[idx1,1] - ip_poly_tile1[idx1])/(ip1_poly_tile1[idx1] - ip_poly_tile1[idx1])

		# HERE
		idx2[idx1] = xyz[idx1,0] <  ip_poly_tile0[idx1] + vt[idx1] * (ip1_poly_tile0[idx1] - ip_poly_tile0[idx1]) # P.x < intersect
		cn[idx2]  += 1 # A valid crossing of y=P.y right of P.x	

	return np.logical_not(cn%2 == 0) # 0 if even (out), and 1 if  odd (in)

def wn_PinPoly(point, poly):
	'''
	WN_PINPOLY

	2D algorithm.
	Winding number test for a point in a polygon.

	Input:   P = a point,
	Return:  wn = the winding number (=0 only when P is outside)

	from: http://geomalgorithms.com/a03-_inclusion.html

	Copyright 2001 softSurfer, 2012 Dan Sunday
	This code may be freely used and modified for any purpose
	providing that this copyright notice is included with it.
	SoftSurfer makes no warranty for this code, and cannot be held
	liable for any real or imagined damage resulting from its use.
	Users of this code must verify correctness for their application.
	'''
	wn = 0 # The  winding number counter

	# Loop through all the edges of the polygon
	for ip in range(poly.npoints): 		 				  # edge from V[i] to  V[i+1]
		if poly[ip][1] <= point[1]: 	 				  # start y <= P.y
			if poly[ip+1][1] > point[1]: 				  # an upward crossing
				if point.isLeft(poly[ip],poly[ip+1]) > 0: # P left of  edge
					wn += 1 							  # have  a valid up intersect
		else:											  # start y > P.y (no test needed)
			if poly[ip+1][1] <= point[1]: 				  # a downward crossing
				if point.isLeft(poly[ip],poly[ip+1]) < 0: # P left of  edge
					wn -= 1
	return wn

def wn_PinPoly_vec(xyz, poly):
	'''
	WN_PINPOLY

	2D algorithm.
	Winding number test for a point in a polygon.

	Input:   P = a point,
	Return:  wn = the winding number (=0 only when P is outside)

	from: http://geomalgorithms.com/a03-_inclusion.html

	Copyright 2001 softSurfer, 2012 Dan Sunday
	This code may be freely used and modified for any purpose
	providing that this copyright notice is included with it.
	SoftSurfer makes no warranty for this code, and cannot be held
	liable for any real or imagined damage resulting from its use.
	Users of this code must verify correctness for their application.
	'''
	npoints = xyz.shape[0]
	wn = np.zeros((npoints,)) # The  winding number counter

	# Loop through all the edges of the polygon
	for ip in range(poly.npoints): 		 				  # edge from V[i] to  V[i+1]
		ip_poly_tile0  = np.tile(poly[ip][0],(npoints,))
		ip_poly_tile1  = np.tile(poly[ip][1],(npoints,))
		ip1_poly_tile0 = np.tile(poly[ip+1][0],(npoints,))
		ip1_poly_tile1 = np.tile(poly[ip+1][1],(npoints,))
		c1 = np.zeros((npoints,),dtype=bool)
		c2 = np.zeros((npoints,),dtype=bool)
		c3 = np.zeros((npoints,),dtype=bool)
		
		c1      = ip_poly_tile1 <= xyz[:,1]                       # start y <= P.y
		c2[c1]  = ip1_poly_tile1[c1] > xyz[c1,1]                  # an upward crossing
		c3[c2]  = Point.areLeft(xyz[c2],poly[ip],poly[ip+1]) > 0  # P left of  edge
		wn[c3] += 1                                               # have a valid up intersect

		c2[:]   = False
		c3[:]   = False

		c1      = np.logical_not(c1)                              # start y > P.y (no test needed)
		c2[c1]  = ip1_poly_tile1[c1] <= xyz[c1,1]                 # a downward crossing
		c3[c2]  = Point.areLeft(xyz[c2],poly[ip],poly[ip+1]) < 0  # P left of  edge
		wn[c3] -= 1

	return wn