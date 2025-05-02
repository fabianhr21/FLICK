#!/usr/bin/env python
#
# Example of the GEOM 3D functionality.
#
# Last revision: 18/01/2021
from __future__ import print_function, division

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt

import pyQvarsi

xyz  = np.array([[1.,0.,0.],[1.,1.,0.],[1.,1.,1.],[1.,0.,1.],
				 [0.,0.,0.],[0.,1.,0.],[0.,1.,1.],[0.,0.,1.]])
verts = [
	xyz[0:4,:],
	xyz[4:8,:],
	xyz[[0,1,5,4],:],
	xyz[[2,6,7,3],:],
	xyz[[0,3,7,4],:],
	xyz[[0,3,7,4],:],
	xyz[[1,2,6,5],:],
]

cube = pyQvarsi.Geom.Cube.from_array(xyz)

p1 = pyQvarsi.Geom.Point(0.5,0.5,0.5)
p2 = pyQvarsi.Geom.Point(1.1,0.6,0.4)

print('p1 = ',p1,'inside' if cube.isinside(p1) else 'outside')
print('p2 = ',p2,'inside' if cube.isinside(p2) else 'outside')

# Plot
fig = plt.figure(1,(8,6),dpi=100)
ax  = fig.add_subplot(111, projection='3d')

ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='k', alpha=.1))
ax.set_xlim([0.,1.2])
ax.set_ylim([0.,1.2])
ax.set_zlim([0.,1.2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.scatter(p1[0],p1[1],p1[2],marker='x',c='b' if cube.isinside(p1) else 'r')
ax.scatter(p2[0],p2[1],p2[2],marker='x',c='b' if cube.isinside(p2) else 'r')

# Check if an array of points are inside
xyzp   = np.array([[.5,.5,0.1],[1.,.35,0.6],[0.2,0.5,1.2],[.35,.15,0.5]])
inside = cube > xyzp

fig = plt.figure(2,(8,6),dpi=100)
ax  = fig.add_subplot(111, projection='3d')

ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='k', alpha=.1))
ax.set_xlim([0.,1.2])
ax.set_ylim([0.,1.2])
ax.set_zlim([0.,1.2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

for ip in range(xyzp.shape[0]):
	print('p%d = '%ip,xyzp[ip,:],'inside' if inside[ip] else 'outside')
	ax.scatter(xyzp[ip,0],xyzp[ip,1],xyzp[ip,2],marker='x',c='b' if inside[ip] else 'r') # Blue if inside, red if outside


## Test a Polygon3D
p0     = pyQvarsi.Geom.Point(1.0,1.0,1.0)
p1     = pyQvarsi.Geom.Point(1.0,0.0,1.0)
p2     = pyQvarsi.Geom.Point(0.5,0.0,1.0)
p3     = pyQvarsi.Geom.Point(0.0,0.5,1.0)
p4     = pyQvarsi.Geom.Point(0.0,1.0,1.0)
p5     = pyQvarsi.Geom.Point(1.0,1.0,0.0)
p6     = pyQvarsi.Geom.Point(1.0,0.0,0.0)
p7     = pyQvarsi.Geom.Point(0.5,0.0,0.0)
p8     = pyQvarsi.Geom.Point(0.0,0.5,0.0)
p9     = pyQvarsi.Geom.Point(0.0,1.0,0.0)
points = np.array([p0,p1,p2,p3,p4,p5,p6,p7,p8,p9])
faces  = [(4,3,2,1,0),(9,5,6,7,8),(0,1,6,5),(0,5,9,4),(4,9,8,3),(3,8,7,2),(2,7,6,1)]

obj = pyQvarsi.Geom.Polygon3D(points,faces)

# Check if an array of points are inside
xyzp   = np.array([[1.2,0,0.1],[0.99,.35,0.6],[0.2,0.5,1.2],[.35,.45,0.5],[1.2,0.8,0.1],[1.2,0.2,0.1],[0.8,-0.2,0.1],[0,0.2,0.1],[-0.2,0.8,0.1],[0.5,1.2,0.1]])
inside = obj > xyzp

# Build point array and vertices array to plot
xyz = np.array([p.xyz for p in points])

verts = [
	xyz[faces[0],:],
	xyz[faces[1],:],
	xyz[faces[2],:],
	xyz[faces[3],:],
	xyz[faces[4],:],
	xyz[faces[5],:],
	xyz[faces[6],:],
]

# Plot
fig = plt.figure(1,(8,6),dpi=100)
ax  = fig.add_subplot(111, projection='3d')

ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='k', alpha=.1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.scatter(obj.centroid[0],obj.centroid[1],obj.centroid[2],marker='o',c='k')
for ip in range(xyzp.shape[0]):
	print('p%d = '%ip,xyzp[ip,:],'inside' if inside[ip] else 'outside')
	ax.scatter(xyzp[ip,0],xyzp[ip,1],xyzp[ip,2],marker='x',c='b' if inside[ip] else 'r') # Blue if inside, red if outside

pyQvarsi.cr_info()
plt.show()