#!/bin/env python
#
# 3-D rotation test using Rodrigues general matrix rotation formula.
# Two 3-D vectors (`a` and `b`) are aligned by rotating `a` and angle `theta` around axis `k`.
#
import numpy as np
import pyQvarsi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from itertools import product, combinations
from matplotlib.patches import FancyArrowPatch # draw a vector
from mpl_toolkits.mplot3d import proj3d

# Plotting helpers
class Arrow3D(FancyArrowPatch):
	def __init__(self, xs, ys, zs, *args, **kwargs):
		super().__init__((0,0), (0,0), *args, **kwargs)
		self._verts3d = xs, ys, zs
	def do_3d_projection(self, renderer=None):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
		self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
		return np.min(zs)

def plot_cube(points,edge_length,ax):
	for e, s in combinations(product(points), 2):
		e, s = np.asarray(e).flatten(), np.asarray(s).flatten()
		# print(e,s,np.sum(np.abs(s - e)), np.linalg.norm(edge_length))
		if np.isclose(np.linalg.norm(np.abs(s - e)),edge_length):
			ax.plot3D(*zip(s, e), color="b")

def plot_vector(origin,end,ax,**kwargs):
	ax.scatter(origin[0],origin[1],origin[2], color="g", s=100)
	a = Arrow3D([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]],
				mutation_scale=20, arrowstyle='-|>', **kwargs)
	ax.add_artist(a)

#########################
# Main

a = np.array([0,1,0]) # Cube normal unit vector
b = np.array([0.5,0.45,0.15]) # Target vector

# Normalize target and compute rotation axis
b_hat = b/np.linalg.norm(b)
k = np.cross(a,b)
print('k = {}'.format(k))

# Create unit cube centered on 0,0,0
p1 = [-1.,-1., 1.]
p2 = [ 1.,-1., 1.]
p3 = [ 1., 1., 1.]
p4 = [-1., 1., 1.]
p5 = [-1.,-1.,-1.]
p6 = [ 1.,-1.,-1.]
p7 = [ 1., 1.,-1.]
p8 = [-1., 1.,-1.]
cube_original = pyQvarsi.Geom.Cube.from_array(np.array([p1,p2,p3,p4,p5,p6,p7,p8]))
cube = pyQvarsi.Geom.Cube.from_array(np.array([p1,p2,p3,p4,p5,p6,p7,p8]))
edge_length = np.linalg.norm(np.array(p2) - np.array(p1))

# Compute rotation angle and rotate cube
k_hat = k / np.linalg.norm(k)
cos_theta = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
print('theta = {}'.format(np.arccos(cos_theta)))
cube.rotate_rodrigues(theta=np.arccos(cos_theta),k=k_hat)

original_points = np.array([p.xyz for p in cube_original.points])
rotated_points = np.array([p.xyz for p in cube.points])

# Plot cubes
fig = plt.figure(figsize=plt.figaspect(0.5))
# original cube
ax = fig.add_subplot(1, 2, 1, projection='3d')
plot_cube(original_points, edge_length=edge_length, ax=ax)
plot_vector([0,0,0],a,ax,color='b')
plot_vector([0,0,0],b_hat,ax,color='red')
ax.view_init(azim=30, elev=30, vertical_axis='y')
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
ax.set_xlim(-1.1,1.1)
ax.set_ylim(-1.1,1.1)
ax.set_zlim(-1.1,1.1)
# rotated cube
ax = fig.add_subplot(1, 2, 2, projection='3d')
plot_cube(rotated_points, edge_length=edge_length, ax=ax)
plot_vector([0,0,0],b_hat,ax,color='red')
ax.view_init(azim=30, elev=30, vertical_axis='y')
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
ax.set_xlim(-1.1,1.1)
ax.set_ylim(-1.1,1.1)
ax.set_zlim(-1.1,1.1)

plt.show()

#########################
# Numerical tests without using pyQvarsi:
# def rotate_rodrigues(theta, k):
# 	K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
# 	I = np.eye(3,3)
# 	return I + np.sin(theta)*K + (1-np.cos(theta))*(np.matmul(K,K))
#
# a = np.array([0,1,0])
# b = np.array([0.5,0.5,0])
# b_hat = b/np.linalg.norm(b)
# k = np.cross(a,b)
#
# print('k = {}'.format(k))
#
# if np.all(k == 0) and b[1] < 0:
# 	k_hat = np.array([1,0,0])
# 	theta = np.pi
# 	R = rotate_rodrigues(theta,k_hat)
# 	print('Target: {} \n'
# 		  'Result: {}'.format(b_hat,R@a))
# elif np.any(k != 0)and np.any(b != 0):
# 	print('hi')
# 	k_hat = k/np.linalg.norm(k)
# 	ctheta = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
# 	theta = np.arccos(ctheta)
# 	print(theta)
# 	R = rotate_rodrigues(theta,k_hat)
# 	print('Target: {} \n'
# 		  'Result: {}'.format(b_hat,R@a))
# elif np.all(b == 0):
# 	raise ValueError('Normal vector is 0.')
# else:
# 	print('Vectors already aligned.')
