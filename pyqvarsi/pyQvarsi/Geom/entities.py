#!/usr/bin/env python
#
# pyQvarsi, GEOM entities.
#
# Small geometric module to cut a field or a mesh.
#
# Library of geometric entities.
#
# Last rev: 12/01/2021
from __future__ import print_function, division

import numpy as np, copy

from ..cr           import cr
from ..mem          import mem
from .basic         import Point, Vector, Ball, Polygon
from ..utils.common import raiseError


class Line(object):
	'''
	A 2D line defined by 2 points and discretized by a number
	of points inside that line.
	'''
#	@mem('Geom.Line')
	def __init__(self, p1, p2, npoints=100):
		d = p2 - p1 # vector pointing in the direction of the line
		f = np.linspace(0.,1.,npoints)
		# Preallocate
		self._points = [Point(p1.x,p1.y,p1.z) for p in range(npoints)]
		self._bbox   = [Ball() for p in range(npoints)]
		for ip in range(1,npoints):
			# Build the point list
			self._points[ip].xyz = p1.xyz + f[ip]*d.xyz
			# For each point compute a ball centered on the point with
			# a radius of half the distance to the last point
			vec = self._points[ip] - self._points[ip-1]
			self._bbox[ip] = Ball(self._points[ip],vec.norm())
		# Add the ball for the first point
		vec = self._points[1] - self._points[0]
		self._bbox[0] = Ball(self._points[0],vec.norm())
		if not self._points[-1] == p2: raiseError('Last point does not match!!')
		self._dist = np.array([])

	def isempty(self):
		return self.npoints == 0

	def isinside(self,point,algorithm=None):
		'''
		Returns True if the point is inside (close to) the line, else False.
		'''
		for b in self._bbox:
			if b.isinside(point): return True # Point is inside the bounding box
		return False

	def areinside(self,xyz,algorithm=None):
		'''
		Returns True if the points are inside (close to) the polygon, else False.
		'''
		out = np.zeros((xyz.shape[0],),dtype=bool)
		# Loop on the point boxes and compute the points that are inside the box
		for b in self._bbox:
			idx      = b.areinside(xyz) # Points are inside the bounding box
			out[idx] = True
		return out

	@cr('Geom.Line.interp')
	def interpolate(self,xyz,var):
		'''
		Interpolates a variable value to the points of the line.
		Assume xyz and var as masked points.
		'''
		if len(self._dist) == 0:
			self._dist = np.zeros((self.npoints,xyz.shape[0]),dtype=np.double)
			# Loop on the point boxes and compute the points that are inside the box
			for ip,b in enumerate(self._bbox):
				idx = b.areinside(xyz) # Points are inside the bounding box
				if len(idx) > 0:
					vec = xyz[idx] - np.tile(b.center.xyz,(xyz[idx].shape[0],1))
					self._dist[ip,idx] = np.sqrt(np.sum(vec*vec,axis=1))
					# If there is a point matching exactly our point then
					# the distance will be 0, so when we invert the distances
					# the weight should be infinite
					id0 = self._dist[ip,idx] == 0.
					self._dist[ip,idx]      = 1./self._dist[ip,idx]
					self._dist[ip,idx][id0] = 1.e20
		# Compute interpolated variable	
		out = np.zeros((len(self._bbox),var.shape[1]) if len(var.shape) > 1 else (len(self._bbox),) ,dtype=var.dtype)
		sum_dist = np.sum(self._dist,axis=1) # length of npoints on the line
		# Compute the averaged for the field
		if len(var.shape) > 1: 
			# Vectorial array
			for idim in range(var.shape[1]):
				out[:,idim] = np.matmul(self._dist,var[:,idim])/sum_dist
		else:
			# Scalar array
			out[:] = np.matmul(self._dist,var)/sum_dist
		return out

	@classmethod
	def from_array(cls,xyz,npoints=100):
		'''
		Build a line from an array of points
		of shape (npoints,3).
		'''
		np = xyz.shape[0]
		if not np == 2: raiseError('Invalid number of points for Line %d' % np)
		return cls(Point(xyz[0,0],xyz[0,1],xyz[0,2]),Point(xyz[1,0],xyz[1,1],xyz[1,2]),npoints)

	@classmethod
	def from_pnd(cls,p1,n,d,npoints=100):
		'''
		Build a line by giving a point (p), the normal (n) at the point
		and the distance (d) to be reached.
		'''
		p2 = p1 + d*n # Compute the second point
		return cls(p1,p2,npoints)

	@property
	def npoints(self):
		return len(self._points)
	@property
	def points(self):
		return self._points
	@points.setter
	def points(self,value):
		self._points = value
	@property
	def xyz(self):
		return np.array([p.xyz for p in self._points])
	@property
	def x(self):
		return np.array([p.x for p in self._points])
	@property
	def y(self):
		return np.array([p.y for p in self._points])
	@property
	def z(self):
		return np.array([p.z for p in self._points])


class SimpleRectangle(Polygon):
	'''
	2D rectangle. Assumes z = 0 and the points aligned with the axis.
	For any other shape please use Rectangle or Polygon.

	4-------3
	|		|
	|		|
	1-------2
	'''
#	@mem('Geom.SRectangle')
	def __init__(self,xmin,xmax,ymin,ymax):
		pointList = np.array([
			Point(xmin,ymin,0.), # 1
			Point(xmax,ymin,0.), # 2
			Point(xmax,ymax,0.), # 3
			Point(xmin,ymax,0.), # 4
		])
		super(SimpleRectangle, self).__init__(pointList)

	def isinside(self,point,algorithm=None):
		'''
		A fast algorithm for simple rectangles.
		'''
		x_inside = point[0] >= self.points[0][0] and point[0] <= self.points[1][0]
		y_inside = point[1] >= self.points[0][1] and point[0] <= self.points[3][1]
		return x_inside and y_inside

	def areinside(self,xyz,algorithm=None):
		'''
		A fast algorithm for simple rectangles.
		'''
		x_inside = np.logical_and(xyz[:,0] >= self.points[0][0],xyz[:,0] <= self.points[1][0])
		y_inside = np.logical_and(xyz[:,1] >= self.points[0][1],xyz[:,1] <= self.points[3][1])
		return np.logical_and(x_inside,y_inside)

	@classmethod
	def from_array(cls,xyz):
		'''
		Build a square from an array of points
		of shape (npoints,3).
		'''
		npoints   = xyz.shape[0]
		if not npoints == 5: raiseError('Invalid number of points for Rectangle %d' % npoints)
		return super(SimpleRectangle, cls).from_array(xyz)


class Rectangle(Polygon):
	'''
	2D rectangle. Assumes z = 0.

	4-------3
	|		|
	|		|
	1-------2
	'''
#	@mem('Geom.Rectangle')
	def __init__(self,points):
		if not len(points) == 4: raiseError('Invalid Rectangle!')
		super(Rectangle, self).__init__(points)
		self.centroid = Point.from_array(0.25*(points[0].xyz+points[1].xyz+points[2].xyz+points[3].xyz))

	def normal(self):
		'''
		Returns the unitary normal that defines the plane
		of the Rectangle.
		'''
		# Code_Saturne algorithm
		u = self.points[1] - self.centroid
		v = self.points[0] - self.centroid
		n = u.cross(v)
		return n/n.norm()

	def project(self,point):
		'''
		Given a point outside the plane defined by the Rectangle, 
		it projects the point into the Rectangle plane.
		'''
		n = self.normal() # Normal to the plane
		if isinstance(point,Point): 
			# We are dealing with a single point
			vp   = point - self.points[0]
			dist = vp.dot(n)
		else:
			# We are dealing with a list of points
			npoints = point.shape[0]
			n       = np.tile(n.xyz,(npoints,)).reshape(npoints,3)
			vp      = point - np.tile(self.points[0].xyz,(npoints,)).reshape(npoints,3)
			dist    = np.tile(np.sum(vp*n,axis=1),(3,1)).T
		# Projected point in the Rectangle plane
		return point - n*dist, dist

	def inclusion3D(self,point):
		'''
		3D inclusion is easily determined by projecting the point and polygon into 2D. 
		To do this, one simply ignores one of the 3D coordinates and uses the other two.
		To optimally select the coordinate to ignore, compute a normal vector to the plane, 
		and select the coordinate with the largest absolute value [Snyder & Barr, 1987]. 
		This gives the projection of the polygon with maximum area, and results in robust computations.

		This function is for internal use inside the Cube method.
		'''
		n   = self.normal()       # Normal to the plane
		p,_ = self.project(point) # Projected point
		# Which is the biggest dimension?
		idmax = np.argmax(np.abs(n.xyz))
		# Convert to xy the smallest dimensions for Rectangle
		points = self.points
		for ip in range(self.npoints):
			points[ip].xyz = np.append(np.delete(self.points[ip].xyz,idmax),np.zeros((1,)))
		self.points = points
		# Redo the bounding box
		self.bbox = Ball.fastBall(self)
		# Do the same for the points
		if isinstance(point,Point):
			p.xyz =  np.append(np.delete(p.xyz,idmax),np.zeros((1,)))
		else:
			npoints = p.shape[0]
			p =  np.append(np.delete(p,idmax,axis=1),np.zeros((npoints,1)),axis=1)
		return p

	@classmethod
	def from_array(cls,xyz):
		'''
		Build a square from an array of points
		of shape (npoints,3).
		'''
		npoints = xyz.shape[0]
		if not npoints == 4: raiseError('Invalid number of points for Rectangle %d' % npoints)
		return super(Rectangle, cls).from_array(xyz)


class Polygon2D(Polygon):
	'''
	2D polygon that assumes z = 0.
	'''
#	@mem('Geom.Poly2D')
	def __init__(self,points):
		super(Polygon2D, self).__init__(points)
		self.centroid = Point.from_array(np.mean([p.xyz for p in points],axis=0))

	def normal(self):
		'''
		Returns the unitary normal that defines the plane
		of the Polygon2D.
		'''
		# Code_Saturne algorithm
		u = self.points[1] - self.centroid
		v = self.points[0] - self.centroid
		n = u.cross(v)
		return n/n.norm()

	def project(self,point):
		'''
		Given a point outside the plane defined by the Polygon2D, 
		it projects the point into the Polygon2D plane.
		'''
		n = self.normal() # Normal to the plane
		if isinstance(point,Point): 
			# We are dealing with a single point
			vp   = point - self.points[0]
			dist = vp.dot(n)
		else:
			# We are dealing with a list of points
			npoints = point.shape[0]
			n       = np.tile(n.xyz,(npoints,)).reshape(npoints,3)
			vp      = point - np.tile(self.points[0].xyz,(npoints,)).reshape(npoints,3)
			dist    = np.tile(np.sum(vp*n,axis=1),(3,1)).T
		# Projected point in the Polygon2D plane
		return point - n*dist, dist

	def inclusion3D(self,point):
		'''
		3D inclusion is easily determined by projecting the point and polygon into 2D. 
		To do this, one simply ignores one of the 3D coordinates and uses the other two.
		To optimally select the coordinate to ignore, compute a normal vector to the plane, 
		and select the coordinate with the largest absolute value [Snyder & Barr, 1987]. 
		This gives the projection of the polygon with maximum area, and results in robust computations.

		This function is for internal use inside the Polygon3D method.
		'''
		n   = self.normal()       # Normal to the plane
		p,_ = self.project(point) # Projected point
		# Which is the biggest dimension?
		idmax = np.argmax(np.abs(n.xyz))
		# Convert to xy the smallest dimensions for Polygon2D
		points = self.points
		for ip in range(self.npoints):
			points[ip].xyz = np.append(np.delete(self.points[ip].xyz,idmax),np.zeros((1,)))
		self.points = points
		# Redo the bounding box
		self.bbox = Ball.fastBall(self)
		# Do the same for the points
		if isinstance(point,Point):
			p.xyz =  np.append(np.delete(p.xyz,idmax),np.zeros((1,)))
		else:
			npoints = p.shape[0]
			p =  np.append(np.delete(p,idmax,axis=1),np.zeros((npoints,1)),axis=1)
		return p

	@classmethod
	def from_array(cls,xyz):
		'''
		Build a square from an array of points
		of shape (npoints,3).
		'''
		npoints = xyz.shape[0]
		return super(Polygon2D, cls).from_array(xyz)


class Plane(Rectangle):
	'''
	3D plane in rectangular form, useful for slices.

	4-------3
	|		|
	|		|
	1-------2
	'''
#	@mem('Geom.Plane')
	def __init__(self,points,mindist=0.1):
		self._mindist = mindist
		if not len(points) == 4: raiseError('Invalid Plane!')
		super(Plane, self).__init__(points)

	def isinside(self,point,algorithm=None):
		'''
		Project the point to the plane defined by the 3D rectangle
		and obtain the inclusion.
		'''
		# Create an auxiliary rectangle
		points = np.array([self.points[0].xyz,
						   self.points[1].xyz,
						   self.points[2].xyz,
						   self.points[3].xyz
						  ]).copy()
		aux = Rectangle.from_array(points)
		# Obtain the distance of the point to the plane
		_,dist = aux.project(point)
		# Check if the projected point is inside the face
		inside = aux.isinside(aux.inclusion3D(point))
		# Appart from being inside the point has to fulfill the minimum distance
		return inside and dist <= self._mindist

	def areinside(self,xyz,algorithm=None):
		'''
		Project the points to the plane defined by the 3D rectangle
		and obtain the inclusion.
		'''
		# Create an auxiliary rectangle
		points = np.array([self.points[0].xyz,
						   self.points[1].xyz,
						   self.points[2].xyz,
						   self.points[3].xyz
						  ]).copy()
		aux = Rectangle.from_array(points)
		# Obtain the distance of the point to the plane
		_,dist = aux.project(xyz)
		# Check if the projected point is inside the face
		inside = aux.areinside(aux.inclusion3D(xyz))
		# Appart from being inside the point has to fulfill the minimum distance
		return np.logical_and(inside,np.abs(dist[:,0]) <= self._mindist)


class SimpleCube(Polygon):
	'''
	3D cube. Assumes the points to be aligned with the axis.
	For any other shape please use Cube or Polygon.

	  8-------7
	 /|      /|
	4-------3 |
	| 5-----|-6
	|/      |/
	1-------2
	'''
#	@mem('Geom.SCube')
	def __init__(self,xmin,xmax,ymin,ymax,zmin,zmax):
		pointList = np.array([
			Point(xmin,ymin,zmin), # 1
			Point(xmax,ymin,zmin), # 2
			Point(xmax,ymax,zmin), # 3
			Point(xmin,ymax,zmin), # 4
			Point(xmin,ymin,zmax), # 5
			Point(xmax,ymin,zmax), # 6
			Point(xmax,ymax,zmax), # 7
			Point(xmin,ymax,zmax), # 8
		])
		super(SimpleCube, self).__init__(pointList)
		self.centroid = Point.from_array(np.mean([p.xyz for p in self.points],axis=0))

	def isinside(self,point,algorithm=None):
		'''
		A fast algorithm for cubes.
		'''
		x_inside = point[0] >= self.points[0][0] and point[0] <= self.points[1][0]
		y_inside = point[1] >= self.points[0][1] and point[1] <= self.points[3][1]
		z_inside = point[2] >= self.points[0][2] and point[2] <= self.points[7][2]
		return x_inside and y_inside and z_inside

	def areinside(self,xyz,algorithm=None):
		'''
		A fast algorithm for cubes.
		'''
		x_inside = np.logical_and(xyz[:,0] >= self.points[0][0], xyz[:,0] <= self.points[1][0])
		y_inside = np.logical_and(xyz[:,1] >= self.points[0][1], xyz[:,1] <= self.points[3][1])
		z_inside = np.logical_and(xyz[:,2] >= self.points[0][2], xyz[:,2] <= self.points[7][2])
		return np.logical_and(np.logical_and(x_inside,y_inside),z_inside)

	@classmethod
	def from_array(cls,xyz):
		'''
		Build a cube from an array of points
		of shape (npoints,3).
		'''
		npoints   = xyz.shape[0]
		if not npoints == 8: raiseError('Invalid number of points for Cube %d' % npoints)
		return super(SimpleCube, cls).from_array(xyz)


class Cube(Polygon):
	'''
	3D cube.

	  8-------7
	 /|      /|
	4-------3 |
	| 5-----|-6
	|/      |/
	1-------2
	'''
#	@mem('Geom.Cube')
	def __init__(self,points):
		if not len(points) == 8: raiseError('Invalid Cube!')
		super(Cube, self).__init__(points)
		self.centroid = Point.from_array(np.mean([p.xyz for p in points],axis=0))
		# Generate the indices for each face
		self._face_ids = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,6,7,3),(0,3,7,4),(1,2,6,5)]

	def isinside(self,point,algorithm=None):
		'''
		Project the point to each of the faces of the cube and check
		if the point is inside or outside of the 2D geometry.
		Each face is a rectangle
		'''
		# Loop the faces
		for face_id in self._face_ids:
			face_points = np.array([self.points[face_id[0]].xyz,
							        self.points[face_id[1]].xyz,
							        self.points[face_id[2]].xyz,
							        self.points[face_id[3]].xyz
							      ]).copy()
			# Obtain each face as a Rectangle
			face = Rectangle.from_array(face_points)
			# Check if the projected point is inside the face
			inside = face.isinside(face.inclusion3D(point))
			# If the point is outside the face we can already stop
			if not inside: return False
		# If we reached here it means the point is inside all the faces
		return True

	def areinside(self,xyz,algorithm=None):
		'''
		Project the point to each of the faces of the cube and check
		if the point is inside or outside of the 2D geometry.
		Each face is a rectangle
		'''
		npoints = xyz.shape[0]
		out     = np.ones((npoints,),dtype=bool)
		# Loop the faces
		for face_id in self._face_ids:
			face_points = np.array([self.points[face_id[0]].xyz,
							        self.points[face_id[1]].xyz,
							        self.points[face_id[2]].xyz,
							        self.points[face_id[3]].xyz
							      ]).copy()
			# Obtain each face as a Rectangle
			face = Rectangle.from_array(face_points)
			# Check if the projected points are inside the face
			inside = face.areinside(face.inclusion3D(xyz))
			# Filter out the points that are outside (False)
			out = np.logical_and(out,inside)
		return out

	@classmethod
	def from_array(cls,xyz):
		'''
		Build a cube from an array of points
		of shape (npoints,3).
		'''
		npoints   = xyz.shape[0]
		if not npoints == 8: raiseError('Invalid number of points for Cube %d' % npoints)
		return super(Cube, cls).from_array(xyz)


class Polygon3D(Polygon):
	'''
	3D Polygon. One needs to provide the list of points and
	list of faces that compose the polygon. Faces must be ordered so
	that the normal is pointing outwards the face.
	'''
#	@mem('Geom.Poly3D')
	def __init__(self,points,faces):
		super(Polygon3D, self).__init__(points)
		self.centroid = Point.from_array(np.mean([p.xyz for p in points],axis=0))
		# Generate the indices for each face
		self._face_ids = faces
		# Store a Delaunay representation of the polygon
		from scipy.spatial import Delaunay
		self._delaunay = Delaunay([p.xyz for p in points])

	def isinside(self,point,algorithm=None):
		'''
		Project the point to each of the faces of the cube and check
		if the point is inside or outside of the 2D geometry.
		Each face is a rectangle
		'''
		return self._delaunay.find_simplex(point.xyz) >= 0

	def areinside(self,xyz):
		'''
		Project the point to each of the faces of the cube and check
		if the point is inside or outside of the 2D geometry.
		Each face is a rectangle
		'''
		return self._delaunay.find_simplex(xyz) >= 0


class Collection(object):
	'''
	A region composed by a collection of geometrical entities
	'''
#	@mem('Geom.Collection')
	def __init__(self,*args):
		self._list   = args

	def __str__(self):
		retstr = 'Collection of %d entities:\n' % (len(self.objects))
		for obj in self.objects:
			retstr += obj.__str__()
		return retstr

	def __len__(self):
		return len(self.objects)

	# Operators
	def __getitem__(self,i):
		'''
		Polygon[i]
		'''
		return self._list[i]

	def __setitem__(self,i,value):
		'''
		Polygon[i] = value
		'''
		self._list[i] = value

	def __iter__(self):
		return self._list.__iter__()

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
		return len(self.objects) == 0

	def isinside(self,point):
		'''
		Returns True if the point is inside the polygon, else False.
		'''
		for obj in self.objects:
			if obj.isinside(point): return True
		return False

	def areinside(self,xyz):
		'''
		Returns True if the points are inside the polygon, else False.
		'''
		out = np.zeros((xyz.shape[0],len(self.objects)),dtype=bool)
		for ii,obj in enumerate(self.objects):
			out[:,ii] = obj.areinside(xyz)
		return np.logical_or.reduce(out,axis=1)

	def compute_centroid(self):
		'''
		Returns the centroid.
		'''
		out = np.array([0.,0.,0.])
		for obj in self.objects:
			out += obj.centroid.xyz
		out /= len(self.obj)
		return Point.from_array(out)

	@property
	def objects(self):
		return self._list

	@property
	def centroid(self):
		return self.compute_centroid()

	@property
	def box(self):
		x,y,z = np.array([],np.double),np.array([],np.double),np.array([],np.double)
		for obj in self.objects:
			x = np.concatenate((x,obj.x))
			y = np.concatenate((y,obj.y))
			z = np.concatenate((z,obj.z))
		return [np.max(x),np.min(x),np.max(y),np.min(y),np.max(z),np.min(z)]


class DiscreteBox(object):
	'''
	A box defining a region that has been discretized 
	in multiple boxes according to a point cloud
	'''
#	@mem('Geom.DiscreteBox')
	def __init__(self,box,points,minsize):
		self._box  = box
		self._m    = None
		self._np_x = 1
		self._np_y = 1
		self._np_z = 1
		self._m    = np.zeros((self._np_x,self._np_y,self._np_z),dtype=bool)
		self.discretize(points,minsize)

	def __len__(self):
		return len(self.objects)

	# Operators
	def __getitem__(self,i):
		'''
		Polygon[i]
		'''
		return self._list[i]

	def __setitem__(self,i,value):
		'''
		Polygon[i] = value
		'''
		self._list[i] = value

	def __iter__(self):
		return self._list.__iter__()

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
		return len(self.objects) == 0

	@cr('Geom.Dbox.discr')
	def discretize(self,points,minsize):
		ext = self.extent
		# Find box sizes
		Lx = ext[0] - ext[1]
		Ly = ext[2] - ext[3]
		Lz = ext[4] - ext[5]
		# Number of npartitions per axis
		self._np_x = max(int(Lx/minsize),1)
		self._np_y = max(int(Ly/minsize),1)
		self._np_z = max(int(Lz/minsize),1)
		# Box matrix
		self._m = np.zeros((self._np_x,self._np_y,self._np_z),dtype=bool)
		# Find normalized index
		ip  = np.round((points[:,0]-ext[1])/Lx,8) if not Lx == 0 else np.zeros_like(points[:,0])
		jp  = np.round((points[:,1]-ext[3])/Ly,8) if not Ly == 0 else np.zeros_like(points[:,1])
		kp  = np.round((points[:,2]-ext[5])/Lz,8) if not Lz == 0 else np.zeros_like(points[:,2])
		# Filter normalized coordinates
		idx = np.logical_and(ip>=0,ip<=1)
		idy = np.logical_and(jp>=0,jp<=1)
		idz = np.logical_and(kp>=0,kp<=1)
		ids = np.logical_and(np.logical_and(idx,idy),idz)
		# Set which points are inside the box
		i = (ip[ids]*(self._np_x-1)).astype(np.int32)
		j = (jp[ids]*(self._np_y-1)).astype(np.int32)
		k = (kp[ids]*(self._np_z-1)).astype(np.int32)
		self._m[i,j,k] = True

	@cr('Geom.Dbox.isinside')
	def isinside(self,point):
		'''
		Returns True if the point is inside the polygon, else False.
		'''
		ext = self.extent
		# Find box sizes
		Lx = ext[0] - ext[1]
		Ly = ext[2] - ext[3]
		Lz = ext[4] - ext[5]
		# Find normalized index
		ip  = np.round((point[0]-ext[1])/Lx,8)
		if ip > 1 or ip < 0: return False
		jp  = np.round((point[1]-ext[3])/Ly,8)
		if jp > 1 or jp < 0: return False
		kp  = np.round((point[2]-ext[5])/Lz,8)
		if kp > 1 or kp < 0: return False
		# Point is inside the box?
		i = (ip*(self._np_x-1)).astype(np.int32)
		j = (jp*(self._np_y-1)).astype(np.int32)
		k = (kp*(self._np_z-1)).astype(np.int32)
		return self._m[i,j,k]

	@cr('Geom.Dbox.areinside')
	def areinside(self,xyz):
		'''
		Returns True if the points are inside the polygon, else False.
		'''
		out = np.zeros((xyz.shape[0],),dtype=bool)
		ext = self.extent
		# Find box sizes
		Lx = ext[0] - ext[1]
		Ly = ext[2] - ext[3]
		Lz = ext[4] - ext[5]
		# Find normalized index
		ip  = np.round((xyz[:,0]-ext[1])/Lx,8) if not Lx == 0 else np.zeros_like(xyz[:,0])
		jp  = np.round((xyz[:,1]-ext[3])/Ly,8) if not Ly == 0 else np.zeros_like(xyz[:,1])
		kp  = np.round((xyz[:,2]-ext[5])/Lz,8) if not Lz == 0 else np.zeros_like(xyz[:,2])
		# Filter normalized coordinates
		idx = np.logical_and(ip>=0,ip<=1)
		idy = np.logical_and(jp>=0,jp<=1)
		idz = np.logical_and(kp>=0,kp<=1)
		ids = np.logical_and(np.logical_and(idx,idy),idz)
		# Do we have more than 1 box?
		if self._np_x == 1 and self._np_y == 1 and self._np_z == 1: 
			out[ids] = True
		else:
			# Point is inside the box?
			i = (ip[ids]*(self._np_x-1)).astype(np.int32)
			j = (jp[ids]*(self._np_y-1)).astype(np.int32)
			k = (kp[ids]*(self._np_z-1)).astype(np.int32)
			out[ids] = self._m[i,j,k]
		return out

	def compute_centroid(self):
		'''
		Returns the centroid.
		'''
		return self._box.centroid

	@property
	def centroid(self):
		return self.compute_centroid()

	@property
	def extent(self):
		x,y,z = self._box.x, self._box.y, self._box.z
		return [np.max(x),np.min(x),np.max(y),np.min(y),np.max(z),np.min(z)]