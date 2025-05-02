#!/usr/bin/env python
#
# pyQvarsi, postproc.
#
# yplus routines.
#
# Last rev: 20/09/2021
from __future__ import print_function, division

import numpy as np, mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

from ..utils.common import raiseError
from ..             import Geom
from ..cr           import cr_start, cr_stop

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()


def get_bounding_box(xyz):
	'''
	Get the bounding box of a domain
	'''
	xmax, xmin = np.nanmax(xyz[:,0]), np.nanmin(xyz[:,0])
	ymax, ymin = np.nanmax(xyz[:,1]), np.nanmin(xyz[:,1])
	zmax, zmin = np.nanmax(xyz[:,2]), np.nanmin(xyz[:,2])	
	return np.array([xmin, xmax, ymin, ymax, zmin, zmax],dtype=np.double)


def get_global_bounding_box(xyz):
	'''
	Obtain the global bounding box
	of a parallel domain.
	'''
	local_bbox = get_bounding_box(xyz)
	# Serial run has all the domain - no need to continue
	if MPI_size == 1: return local_bbox 
	# We have a partitioned domain here
	# Gather to all the local boxes
	bboxes = np.array(MPI_comm.allgather(local_bbox),dtype=np.double)
	# At this point all processors should have
	# a list of the local boxes
	xmax, xmin = np.nanmax(bboxes[:,1]), np.nanmin(bboxes[:,0])
	ymax, ymin = np.nanmax(bboxes[:,3]), np.nanmin(bboxes[:,2])
	zmax, zmin = np.nanmax(bboxes[:,5]), np.nanmin(bboxes[:,4])	
	return np.array([xmin, xmax, ymin, ymax, zmin, zmax],dtype=np.double)


def build_domain_boxes(bbox,nbx,nby,nbz,fact):
	'''
	Splits the domain bouding box in separate boxes
	per each axis given a safety factor
	'''
	# Unpack bounding box
	xmin, xmax, ymin, ymax, zmin, zmax = bbox
	# Multiply domains by a factor
	xmax *= 1.+fact if xmax > 0 else 1.-fact
	ymax *= 1.+fact if ymax > 0 else 1.-fact
	zmax *= 1.+fact if zmax > 0 else 1.-fact
	xmin *= 1.-fact if xmin > 0 else 1.+fact
	ymin *= 1.-fact if ymin > 0 else 1.+fact
	zmin *= 1.-fact if zmin > 0 else 1.+fact
	# Compute deltas
	dbx   = (xmax-xmin)/nbx
	dby   = (ymax-ymin)/nby
	dbz   = (zmax-zmin)/nbz
	# Generate boxes
	boxes = []
	for ibx in range(nbx):
		xbox1, xbox2 = xmin+ibx*dbx, xmin+(ibx+1)*dbx
		for iby in range(nby):
			ybox1, ybox2 = ymin+iby*dby, ymin+(iby+1)*dby
			for ibz in range(nbz):
				zbox1, zbox2 = zmin+ibz*dbz, zmin+(ibz+1)*dbz
				boxes.append(Geom.SimpleCube(xbox1,xbox2,ybox1,ybox2,zbox1,zbox2))
	return boxes


def compute_yplus_u(mu,xyz,gradv,walld,wallid,boxes,boxes_ids,boxes_xyz,first):
	'''
	Compute yplus given a series of points
	'''
	yplus = -np.ones((xyz.shape[0],),dtype=np.double)

	# Find out the node that is on the wall
	xyzw = xyz.copy() # Theoretical position of the wall node
	xyzw[wallid==0,0] -= walld[wallid==0]
	xyzw[wallid==1,0] += walld[wallid==1]	
	xyzw[wallid==2,1] -= walld[wallid==2]
	xyzw[wallid==3,1] += walld[wallid==3]
	xyzw[wallid==4,2] -= walld[wallid==4]
	xyzw[wallid==5,2] += walld[wallid==5]
	
	# For each point find the u_tau
	for ip in range(xyzw.shape[0]):
		# Find the box that belongs to the point
		idbox = -1
		for ibox,box in enumerate(boxes):
			if box.isinside(xyzw[ip,:]):
				idbox = ibox
				break
		
		# Crash here! Boxes are on the global domain
		# so the ID of the box should always be found.
		if idbox < 0: raiseError('Point %d [%f,%f,%f] not found inside the boxes!'%(ip,xyzw[ip,0],xyzw[ip,1],xyzw[ip,2]))

		# Now we should have the id of the box, retrieve the points inside the box
		if first and len(boxes_xyz[idbox]) == 0:
			mask = boxes[idbox].areinside(xyz)
			boxes_ids[idbox] = np.where(mask)[0]
			boxes_xyz[idbox] = xyz[mask,:]

		# Now find the node id on a reduced search
		# case 1: len(boxes_xyz[idbox]) == 0 -> no points of this domain inside target box
		if len(boxes_xyz[idbox]) == 0: continue
		# Which points from idbox are equal to the wall point?
		iplist = np.where(np.all(np.abs(boxes_xyz[idbox] - xyzw[ip,:])<1e-12,axis=1))[0]
		# case 2: len(iplist) == 0 -> no point has been found in idbox
		if len(iplist) == 0: continue
		# At this point we should be able to obtain a node id
		node_id = boxes_ids[idbox][iplist[0]]
		
		# Compute u_tau
		if wallid[ip] == 0: yplus[ip] = np.sqrt(mu*abs(gradv[node_id,0]))*walld[ip]/mu # du/dx
		if wallid[ip] == 1: yplus[ip] = np.sqrt(mu*abs(gradv[node_id,0]))*walld[ip]/mu # du/dx
		if wallid[ip] == 2: yplus[ip] = np.sqrt(mu*abs(gradv[node_id,1]))*walld[ip]/mu # du/dy
		if wallid[ip] == 3: yplus[ip] = np.sqrt(mu*abs(gradv[node_id,1]))*walld[ip]/mu # du/dy
		if wallid[ip] == 4: yplus[ip] = np.sqrt(mu*abs(gradv[node_id,2]))*walld[ip]/mu # du/dz
		if wallid[ip] == 5: yplus[ip] = np.sqrt(mu*abs(gradv[node_id,2]))*walld[ip]/mu # du/dz

	return yplus


def yplus_xyz_3D(xyz,gradv,mu,x_fnt,x_bck,y_top,y_bot,z_rgt,z_lft,
	nbx=4,nby=4,nbz=4,fact=0.1):
	'''
	Compute the yplus given the mesh points and the points of
	the wall assuming that the mesh is regular on the xyz directions.

	This code is adapted for a parallel domain.

	INPUTS:
		> xyz:    positions of the nodes
		> gradv:  gradient of the flow velocity [da/dx,da/dy,da/dz]
		> mu:     viscosity
		> x_fnt:  front x wall coordinate or NaN to assume there is no wall.
		> x_bck:  back x wall coordinate or NaN to assume there is no wall.
		> y_top:  top y wall coordinate or NaN to assume there is no wall.
		> y_bot:  bottom y wall coordinate or NaN to assume there is no wall.
		> z_rgt:  right z wall coordinate or NaN to assume there is no wall.
		> z_lft:  left z wall coordinate or NaN to assume there is no wall.

	OUTPUTS:
		> yplus:  normalized wall distance
		> walld:  distance to the wall
	'''
	cr_start('yplus_xyz_3D',0)

	# Compute the global domain bounding box
	bbox = get_global_bounding_box(xyz)

	# Split the domain into boxes in order to
	# speed up the search algorithm
	boxes = build_domain_boxes(bbox,nbx,nby,nbz,fact)
	boxes_ids = [np.array([],np.int32)]*len(boxes)
	boxes_xyz = [np.array([],np.double)]*len(boxes)

	# Distance to the wall
	# This is OK in parallel since the wall position is
	# known for all processors
	walld6 = np.zeros((xyz.shape[0],6),dtype=np.double)
	walld6[:,0] = xyz[:,0] - x_bck
	walld6[:,1] = x_fnt    - xyz[:,0]
	walld6[:,2] = xyz[:,1] - y_bot
	walld6[:,3] = y_top    - xyz[:,1]
	walld6[:,4] = xyz[:,2] - z_lft
	walld6[:,5] = z_rgt    - xyz[:,2]
	
	# Distance to the wall
	walld  = np.nanmin(walld6   ,axis=1) if not np.any(np.isnan(xyz)) else np.nan*np.ones((xyz.shape[0],),dtype=np.double) 
	# Id of the closest wall, either +x,-x, etc.
	wallid = np.nanargmin(walld6,axis=1) if not np.any(np.isnan(xyz)) else -np.ones((xyz.shape[0],),dtype=np.int) 

	# Compute yplus for the points of the current subdomain
	yplus = compute_yplus_u(mu,xyz,gradv,walld,wallid,boxes,boxes_ids,boxes_xyz,True) if not np.any(np.isnan(xyz)) else np.nan*np.ones((xyz.shape[0],),dtype=np.double)

	# Manage the points that we could not find inside the current domain
	id_not_found = np.where(yplus < 0)[0]
	if MPI_size > 1:
		# Gather from all the cores the points that could not be found
		xyz_not_found    = MPI_comm.allgather(xyz[id_not_found,:])
		walld_not_found  = MPI_comm.allgather(walld[id_not_found])
		wallid_not_found = MPI_comm.allgather(wallid[id_not_found])
		# Per each core, look for the points that have not been found
		# and generate the yplus
		for icore in range(MPI_size):
			xyz_rank    = xyz_not_found[icore]
			walld_rank  = walld_not_found[icore]
			wallid_rank = wallid_not_found[icore]
			# Skip if there is no operation to do
			if xyz_rank.shape[0] == 0: continue 
			# All subdomains look for yplus given the xyz_rank
			yplus_rank = compute_yplus_u(mu,xyz_rank,gradv,walld_rank,
				wallid_rank,boxes,boxes_ids,boxes_xyz,False) if not MPI_rank == icore else -np.ones((xyz_rank.shape[0],),dtype=np.double)
			# Subdomain icore gathers all yplus computations
			yplus_gather = np.array(MPI_comm.gather(yplus_rank,root=icore),np.double)
			# Subdomain that gathers stores into yplus
			if icore == MPI_rank:
				yplus[id_not_found] = np.nanmax(yplus_gather,axis=0)
	else:
		# This is not a parallel run so crash if we could
		# not find any point using the algorithm
		if len(id_not_found) > 0: raiseError('Some points (%d) have not been found!'%len(id_not_found))

	cr_stop('yplus_xyz_3D',0)
	return yplus, walld

def findDistancesHexaVTK(mynode, myconnec, coord, surfMaskSet, uStreamwise, nDims = 3):
	'''
	Function to find the closest nodes to a given one in VTK connectivity in streamise,
	spanwise and normal direction.

	INPUTS:
		> mynode:       local numeration of node in wall surface
		> myconnec:     VTK local connectivity of node in wall surface
		> coord:        local coordinates
		> surfMaskSet:  list of wall node IDs
		> uStreamwise:  3 component array of streamwise velocity
		> nDims:        3 dimension
	OUTPUTS:
		> firstNode:    ID of the closest node in normal direction
		> iDist:        distance to the closest node in streamwise direction
		> jDist:        distance to the closest node in spanwise direction
		> kDist:        distance to the closest node in normal direction
	'''
	candidate_nodes = {0 : [1, 3, 4],
					  1 : [2, 5, 0],
					  2 : [1 ,3, 6],
					  3 : [0, 2, 7],
					  4 : [0, 5, 7],
					  5 : [1, 4, 6],
					  6 : [2, 5, 7],
					  7 : [3, 4, 6]
					}
	distSurf = np.zeros((2, nDims))
	dirSurf  = np.zeros((2, nDims))
	count = 0
	firstNode = None
	iNode = myconnec[mynode]
	for cand_node in candidate_nodes[mynode]:
		if myconnec[cand_node] not in surfMaskSet:
			# If cand_node not on the surface, it's the first node in normal direction
			firstNode = myconnec[cand_node]
			# k is normal direction
			kDist = np.linalg.norm(coord[firstNode] - coord[iNode])
		else:
			# The other two nodes lie on the surface
			auxNode = myconnec[cand_node]
			auxDist = coord[auxNode] - coord[iNode]
			distSurf[count] = auxDist
			dirSurf[count]  = distSurf[count]/np.linalg.norm(distSurf[count])
			count += 1
	# i is streamwise direction, j is spanwise
	projStreamwise = np.dot(dirSurf, np.array(uStreamwise)[:, np.newaxis]) * dirSurf
	norms  = np.linalg.norm(projStreamwise, axis=1)
	idNorm = np.argmax(norms)
	# i direction is the one closest to the streamwise projection on the local wall plane
	iDist  = distSurf[0]; jDist = distSurf[1]
	if idNorm == 0:
		iDist = np.linalg.norm(distSurf[0]); jDist = np.linalg.norm(distSurf[1])
	else:
		iDist = np.linalg.norm(distSurf[1]); jDist = np.linalg.norm(distSurf[0])
	return firstNode, iDist, jDist, kDist

def computeWallDistancesSOD2D(surf_mask, connecvtk, coord, uStreamwise):
	surfMaskSet = set(surf_mask)
	firstNodeMask = np.zeros(surf_mask.shape, dtype=np.int32)
	iDistVal  = np.zeros(surf_mask.shape)
	jDistVal  = np.zeros(surf_mask.shape)
	kDistVal  = np.zeros(surf_mask.shape)
	for iNode, wallNode in enumerate(surf_mask):
		myelem, mynode = np.argwhere(connecvtk == surf_mask[iNode])[0]
		myconnec = connecvtk[myelem,:]
		firstNode, iDist, jDist, kDist = findDistancesHexaVTK(mynode, myconnec, coord, surfMaskSet, uStreamwise)
		firstNodeMask[iNode] = firstNode
		iDistVal[iNode]  = iDist
		jDistVal[iNode]  = jDist
		kDistVal[iNode]  = kDist
	return firstNodeMask, iDistVal, jDistVal, kDistVal