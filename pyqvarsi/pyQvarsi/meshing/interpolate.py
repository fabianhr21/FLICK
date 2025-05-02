#!/usr/bin/env python
#
# pyQvarsi, utils.
#
# Interpolation utility routines.
#
# Last rev: 10/06/2021
from __future__ import print_function, division

import numpy as np

from ..                import Geom
from ..field           import Field
from ..partition_table import PartitionTable
from ..cr              import cr
from ..utils.common    import raiseWarning, raiseError
from ..utils.parallel  import MPI_RANK,mpi_create_op,mpi_reduce,mpi_bcast


def _interpNN_reduce(f1,f2,dtype):
	'''
	Assume f1 and f2 to be fields at the same point
	and the variable 'dist' must exist in all two fields.

	Return a new field where the minimum distance is found
	between all fields.
	'''
	# Create a matrix to store both distances
	minmat = np.zeros((len(f1),2),np.float32)
	minmat[:,0] = f1['dist']
	minmat[:,1] = f2['dist']
	imin = np.argmin(minmat,axis=1)
	# Now we know whether to choose from f1 or f2
	outf = f1.__class__.field_like(f1)
	for v in outf.varnames:
		outf[v][imin==1] = f2[v][imin==1]
	return outf

interpNN_reduce = mpi_create_op(_interpNN_reduce, commute=True)

@cr('meshing.interpNN')
def interpolateNearestNeighbour(mesh,xyz,f,fact=2.0,f_incr=1.0,r_incr=1.0,
	global_max_iter=1,ball_max_iter=5,root=-1,target_mask=None):
	'''
	Interpolates a field f on this mesh to the given xyz points using a nearest neighbour approach.
	The mass matrix on the mesh needs to be computed in order to obtain the element size.

	Inputs:
	- mesh:            source mesh (partitioned)
	- xyz:             target points
	- f:               source field (on source mesh)
	
	Optionals:
	- fact:            factor to multiply the source mesh max size of element (default: 2.0)
	- f_incr:          the box max size doubles at each iteration (default: 1.0)
	- r_incr:          the ball radius doubles at each iteration (default: 1.0)
	- global_max_iter: maximum iterations on the global algorithm (default: 1)
	- ball_max_iter:   maximum iterations on finding elements inside the ball (default: 5)
	- root:			   where to reduce the data to (default: -1)

	Returns:
	- field of xyz points with interpolated values
	'''
	# 1. Create output field
	tf = f.__class__(xyz=xyz,ptable=PartitionTable.new(1,npoints=1,nelems=1,has_master=mesh.partition_table.has_master)) # ptable not rellevant here
	for v in f.varnames:
		ndim  = 0 if len(f[v].shape) == 1 else f[v].shape[1]
		tf[v] = np.nan*np.zeros((xyz.shape[0],) if ndim==0 else (xyz.shape[0],ndim),dtype=f[v].dtype)
	tf['dist'] = 1e99*np.ones((xyz.shape[0],),dtype=np.double)

	# 2. Initialize ball search parameters
	radius = np.cbrt(np.nanmin(mesh.volume))
	if radius < 0: raiseError('Problems with the mass matrix... Negative radius found interpolation!')

	# 3. Discretized bounding box of the source mesh
	maxSize    = fact*np.cbrt(np.nanmax(mesh.volume))
	points     = np.vstack([mesh.xyz,mesh.xyz_center])
	if maxSize < 0: raiseError('Problems with the mass matrix... Negative maximum size found interpolation!')
	if mesh.xyz.shape[0] == 0: # Deal with an empty partition
		boundedIds = []
	else:
		box        = Geom.DiscreteBox(mesh.boundingBox,points,maxSize)
		mask       = box.areinside(xyz)
		boundedIds = np.where(mask == True)[0]

	## Initialize global iterations
	for itg in range(global_max_iter):

		# 4. Computing bounding box efficiency parameters
		bounded_points_L   = len(boundedIds)
		bounded_points_MAX = mpi_reduce(bounded_points_L,op='max',all=True)
		if MPI_RANK == 0: bounded_points_L = 1e99
		bounded_points_MIN = mpi_reduce(bounded_points_L,op='min',all=True)

		real_points_L      = len(mesh.xyz)
		real_points_MAX    = mpi_reduce(real_points_L,op='max',all=True)
		if MPI_RANK == 0: real_points_L = 1e99
		real_points_MIN    = mpi_reduce(real_points_L,op='min',all=True) 

		# 5. Interpolation process
		# Loop over all the target nodes contained in the source bounding box
		for tnodeId in boundedIds:
			
			if target_mask is not None and target_mask[tnodeId]==0: continue
			
			# Parameters
			mindist = 1e99
			snodeId = -1

			tpoint = xyz[tnodeId,:]
			ball   = Geom.Ball(Geom.Point(tpoint[0],tpoint[1],tpoint[2]),radius)

			# Loop over a growing sphere to determine the target point's closest source nodes
			for it in range(ball_max_iter):
				ball.radius *= (1.+r_incr*it)
				if ball.radius > maxSize: break
				# Subset of source nodes within the sphere centered in the target point
				mask    = ball.areinside(mesh.xyz)
				ssubset = np.where(mask==True)[0]

				# We need at least one close node in the subset to evaluate the distance
				if ssubset.shape[0] < 1: continue

				# Loop to determine the source node with minimum distance to the target node. 
				# The search is performed only within the sphere subset
				for nodeId in ssubset:
					node = mesh.xyz[nodeId,:]
					dist = np.linalg.norm(tpoint-node)
					if dist < mindist:
						mindist = dist
						snodeId = nodeId
				break

			# At least all subdomains will have a value for the distance of this node
			for v in f.varnames:
				tf[v][tnodeId]  = f[v][snodeId]
			tf['dist'][tnodeId] = mindist

		# 6. Now we have a field (tfield) for all subdomains that contains information
		# on the interpolated values and the minimum distance to the target point
		#
		# We can now call a reduction algorithm that will populate all the points
		# of tfield
		tfg = tf.reduce(root=root,op=interpNN_reduce)

		# 7. Find the points that are not found
		not_found = np.array([])
		if MPI_RANK == root or root < 0:
			not_found = np.where(tfg['dist'] >= 1e99)[0]
		
		not_found = mpi_bcast(not_found,root=max(root,0))
		# All points have been found so stop the algorithm
		if not_found.shape[0] == 0: break

		# 8. Otherwise we are still missing some points, increase the box size
		# and try to look for them
		if np.any(np.isnan(mesh.xyz)): # Deal with an empty partition
			boundedIds = []
		else:
			maxSize   *= (1.+f_incr*(itg+1))
			box        = Geom.DiscreteBox(mesh.boundingBox,points,maxSize)
			mask       = box.areinside(xyz[not_found,:])
			boundedIds = not_found[mask]
		tf = tfg
	# 9. Produce a warning if some points are still not found
	if MPI_RANK == root or root < 0:
		not_found = np.where(tfg['dist'] >= 1e99)[0]
		if not_found.shape[0] > 0: 
			raiseWarning('Some points (%d):\n'%(not_found.shape[0]) + str(xyz[not_found,:]) + '\ncould not be found!')
		tfg.xyz[not_found] *= np.nan

	# Return
	tfg.delete('dist') # delete dist variable
	return tfg


def _interpFEM_reduce(f1,f2,dtype):
	'''
	Assume f1 and f2 to be fields at the same point
	and the variable 'dist' must exist in all two fields.

	Return a new field where the minimum distance is found
	between all fields.
	'''
	# Create a matrix to store both distances
	boolmat = np.zeros((len(f1),2),bool)
	boolmat[:,0] = f1['found']
	boolmat[:,1] = f2['found']
	ibool = np.argmax(boolmat,axis=1)
	# Now we know whether to choose from f1 or f2
	outf = f1.__class__.field_like(f1)
	for v in outf.varnames:
		outf[v][ibool==1] = f2[v][ibool==1]
	return outf

interpFEM_reduce = mpi_create_op(_interpFEM_reduce, commute=True)

@cr('meshing.interpFEM')
def interpolateFEM(mesh,xyz,f,fact=2.0,f_incr=1.0,r_incr=1.0,
	global_max_iter=1,ball_max_iter=5,root=-1,target_mask=None):
	'''
	Interpolates a field f on this mesh to the given xyz points using the finite element machinery.
	The mass matrix on the mesh needs to be computed in order to obtain the element size.

	Inputs:
	- mesh:           source mesh (partitioned)
	- xyz:            target points
	- f:              source field (on source mesh)
	
	Optionals:
	- fact:            factor to multiply the source mesh max size of element (default: 2.0)
	- f_incr:          the box max size doubles at each iteration (default: 1.0)
	- r_incr:          the ball radius doubles at each iteration (default: 1.0)
	- global_max_iter: maximum iterations on the global algorithm (default: 1)
	- ball_max_iter:   maximum iterations on finding elements inside the ball (default: 5)
	- root:			   where to reduce the data to (default: -1)

	Returns:
	- field of xyz points with interpolated values
	'''
	# 1. Create output field
	tf = f.__class__(xyz=xyz,ptable=PartitionTable.new(1,npoints=1,nelems=1,has_master=mesh.partition_table.has_master)) # ptable not rellevant here
	for v in f.varnames:
		ndim  = 0 if len(f[v].shape) == 1 else f[v].shape[1]
		tf[v] = np.nan*np.zeros((xyz.shape[0],) if ndim==0 else (xyz.shape[0],ndim),dtype=f[v].dtype)
	tf['found'] = np.zeros((xyz.shape[0],),dtype=bool)

	# 2. Initialize ball search parameters
	radius = np.cbrt(np.nanmin(mesh.volume))
	if radius < 0: raiseError('Problems with the mass matrix... Negative radius found interpolation!')

	# 3. Discretized bounding box of the source mesh
	maxSize    = fact*np.cbrt(np.nanmax(mesh.volume))
	points     = np.vstack([mesh.xyz,mesh.xyz_center])
	if maxSize < 0: raiseError('Problems with the mass matrix... Negative maximum size found interpolation!')
	if np.any(np.isnan(mesh.xyz)): # Deal with an empty partition
		boundedIds = []
	else:
		box        = Geom.DiscreteBox(mesh.boundingBox,points,maxSize)
		mask       = box.areinside(xyz)
		boundedIds = np.where(mask == True)[0]

	## Initialize global iterations
	for itg in range(global_max_iter):	

		# 4. Computing bounding box efficiency parameters
		bounded_points_L   = len(boundedIds)
		bounded_points_MAX = mpi_reduce(bounded_points_L,op='max',all=True)
		if MPI_RANK == 0: bounded_points_L = 1e99
		bounded_points_MIN = mpi_reduce(bounded_points_L,op='min',all=True)

		real_points_L      = len(mesh.xyz)
		real_points_MAX    = mpi_reduce(real_points_L,op='max',all=True)
		if MPI_RANK == 0: real_points_L = 1e99
		real_points_MIN    = mpi_reduce(real_points_L,op='min',all=True) 

		# 5. Interpolation process
		# Loop over all the target nodes contained in the source bounding box
		for tnodeId in boundedIds:

			if target_mask is not None and target_mask[tnodeId]==0: continue
			
			# Parameters
			mindist = 1e99
			snodeId = -1

			tpoint = xyz[tnodeId,:]
			ball   = Geom.Ball(Geom.Point(tpoint[0],tpoint[1],tpoint[2]),radius)

			# Loop over a growing sphere to determine the target point's closest source nodes
			for it in range(ball_max_iter):
				ball.radius *= (1.+r_incr*it)
				if ball.radius > maxSize: break
				# Subset of source nodes within the sphere centered in the target point
				mask    = ball.areinside(mesh.xyz)

				ssubset = np.where(mask==True)[0]

				# We need at least one close node in the subset to find a candidate element
				if ssubset.shape[0] > 0: break

			candidate_elements = np.unique([e for s in ssubset for e in mesh.find_node_in_elems(s)])
			# If we found a node we found candidate elements, now see if 
			# one of these elements contains our point
			host = None
			for iel in candidate_elements:
				host = mesh._elemList[iel]
				if host.isinside(tpoint,mesh.xyz[host.nodes,:]):
					tf['found'][tnodeId] = True
					break
			if not tf['found'][tnodeId]: 
				continue

			# Find the coordinates of the point in local (element) coordinates
			stz = host.find_stz(tpoint,mesh.xyz[host.nodes,:])
			# Interpolate the variables from input field f to the output field new
			for v in f.varnames:
				tf[v][tnodeId] = host.interpolate(stz,f[v][host.nodes])

		# 6. Now we have a field (tfield) for all subdomains that contains information
		# on the interpolated values and the minimum distance to the target point
		#
		# We can now call a reduction algorithm that will populate all the points
		# of tfield
		tfg = tf.reduce(root=root,op=interpFEM_reduce)

		# 7. Find the points that are not found
		not_found = np.array([])
		if MPI_RANK == root or root < 0:
			not_found = np.where(tfg['found'] == False)[0]

		not_found = mpi_bcast(not_found,root=max(root,0))
		# All points have been found so stop the algorithm
		if not_found.shape[0] == 0: break

		# 8. Otherwise we are still missing some points, increase the box size
		# and try to look for them
		if np.any(np.isnan(mesh.xyz)): # Deal with an empty partition
			boundedIds = []
		else:
			maxSize   *= (1.+f_incr*(itg+1))
			box        = Geom.DiscreteBox(mesh.boundingBox,points,maxSize)
			mask       = box.areinside(xyz[not_found,:])
			boundedIds = not_found[mask]

		tf = tfg

	# 9. Produce a warning if some points are not found
	if MPI_RANK == root or root < 0:
		not_found = np.where(tfg['found'] == False)[0]
		if not_found.shape[0] > 0: 
			raiseWarning('Some points (%d):\n'%(not_found.shape[0]) + str(xyz[not_found,:]) + '\ncould not be found!')
		tfg.xyz[not_found] *= np.nan

	# Return
	tfg.delete('found') # delete found variable
	return tfg


@cr('meshing.interpFEMNN')
def interpolateFEMNearestNeighbour(mesh,xyz,f,fact=2.0,f_incr=1.0,r_incr=1.0,
	global_max_iter=1,ball_max_iter=5,root=-1,target_mask=None):
	'''
	Interpolates a field f on this mesh to the given xyz points using the shape functions.
	Finds the nearest element via a nearest neighbour approach.
	The mass matrix on the mesh needs to be computed in order to obtain the element size.

	Inputs:
	- mesh:            source mesh (partitioned)
	- xyz:             target points
	- f:               source field (on source mesh)
	
	Optionals:
	- fact:            factor to multiply the source mesh max size of element (default: 2.0)
	- f_incr:          the box max size doubles at each iteration (default: 1.0)
	- r_incr:          the ball radius doubles at each iteration (default: 1.0)
	- global_max_iter: maximum iterations on the global algorithm (default: 1)
	- ball_max_iter:   maximum iterations on finding elements inside the ball (default: 5)
	- root:			   where to reduce the data to (default: -1)

	Returns:
	- field of xyz points with interpolated values
	'''
	# 1. Create output field
	tf = f.__class__(xyz=xyz,ptable=PartitionTable.new(1,npoints=1,nelems=1,has_master=mesh.partition_table.has_master)) # ptable not rellevant here
	for v in f.varnames:
		ndim  = 0 if len(f[v].shape) == 1 else f[v].shape[1]
		tf[v] = np.nan*np.zeros((xyz.shape[0],) if ndim==0 else (xyz.shape[0],ndim),dtype=f[v].dtype)
	tf['dist'] = 1e99*np.ones((xyz.shape[0],),dtype=np.double)

	# 2. Initialize ball search parameters
	radius = np.cbrt(np.nanmin(mesh.volume))
	if radius < 0: raiseError('Problems with the mass matrix... Negative radius found interpolation!')

	# 3. Discretized bounding box of the source mesh
	maxSize    = fact*np.cbrt(np.nanmax(mesh.volume))
	points     = mesh.xyz_center
	if maxSize < 0: raiseError('Problems with the mass matrix... Negative maximum size found interpolation!')
	if np.any(np.isnan(mesh.xyz)): # Deal with an empty partition
		boundedIds = []
	else:
		box        = Geom.DiscreteBox(mesh.boundingBox,points,maxSize)
		mask       = box.areinside(xyz)
		boundedIds = np.where(mask == True)[0]

	## Initialize global iterations
	for itg in range(global_max_iter):

		# 4. Computing bounding box efficiency parameters
		bounded_points_L   = len(boundedIds)
		bounded_points_MAX = mpi_reduce(bounded_points_L,op='max',all=True)
		if MPI_RANK == 0: bounded_points_L = 1e99
		bounded_points_MIN = mpi_reduce(bounded_points_L,op='min',all=True)

		real_points_L      = len(mesh.xyz)
		real_points_MAX    = mpi_reduce(real_points_L,op='max',all=True)
		if MPI_RANK == 0: real_points_L = 1e99
		real_points_MIN    = mpi_reduce(real_points_L,op='min',all=True) 

		# 5. Interpolation process
		# Loop over all the target nodes contained in the source bounding box
		for tnodeId in boundedIds:
			# Parameters
			mindist = 1e99
			selemId = -1

			tpoint = xyz[tnodeId,:]
			ball   = Geom.Ball(Geom.Point(tpoint[0],tpoint[1],tpoint[2]),radius)

			# Loop over a growing sphere to determine the target point's closest source nodes
			for it in range(ball_max_iter):
				ball.radius *= (1.+r_incr*it)
				if ball.radius > maxSize: break
				# Subset of source nodes within the sphere centered in the target point
				mask    = ball.areinside(mesh.xyz_center)
				ssubset = np.where(mask==True)[0]

				# We need at least one close element in the subset to evaluate the distance
				if ssubset.shape[0] < 1: continue

				# Loop to determine the source element with minimum distance to the target point. 
				# The search is performed only within the sphere subset
				for elemId in ssubset:
					elem = mesh.xyz_center[elemId,:]
					dist = np.linalg.norm(tpoint-elem)
					if dist < mindist:
						mindist = dist
						selemId = elemId
				break
			# At this point we have a candidate element for the point
			# now check if the point is really inside the element
			host = mesh._elemList[selemId]
			if host.isinside(tpoint,mesh.xyz[host.nodes,:]):
				# Find the coordinates of the point in local (element) coordinates
				stz = host.find_stz(tpoint,mesh.xyz[host.nodes,:])
				# Interpolate the variables from input field f to the output field new
				for v in f.varnames:
					tf[v][tnodeId] = host.interpolate(stz,f[v][host.nodes])				
				tf['dist'][tnodeId] = mindist
			else:
				tf['dist'][tnodeId] = 1e99

		# 6. Now we have a field (tfield) for all subdomains that contains information
		# on the interpolated values and the minimum distance to the target point
		#
		# We can now call a reduction algorithm that will populate all the points
		# of tfield
		tfg = tf.reduce(root=root,op=interpNN_reduce)

		# 7. Find the points that are not found
		not_found = np.array([])
		if MPI_RANK == root or root < 0:
			not_found = np.where(tfg['dist'] >= 1e99)[0]
		
		not_found = mpi_bcast(not_found,root=max(root,0))
		# All points have been found so stop the algorithm
		if not_found.shape[0] == 0: break

		# 8. Otherwise we are still missing some points, increase the box size
		# and try to look for them
		if np.any(np.isnan(mesh.xyz)): # Deal with an empty partition
			boundedIds = []
		else:
			maxSize   *= (1.+f_incr*(itg+1))
			box        = Geom.DiscreteBox(mesh.boundingBox,points,maxSize)
			mask       = box.areinside(xyz[not_found,:])
			boundedIds = not_found[mask]
		
		tf = tfg

	# 9. Produce a warning if some points are still not found
	if MPI_RANK == root or root < 0:
		not_found = np.where(tfg['dist'] >= 1e99)[0]
		if not_found.shape[0] > 0: 
			raiseWarning('Some points (%d):\n'%(not_found.shape[0]) + str(xyz[not_found,:]) + '\ncould not be found!')
		tfg.xyz[not_found] *= np.nan

	# Return
	tfg.delete('dist') # delete dist variable
	return tfg
