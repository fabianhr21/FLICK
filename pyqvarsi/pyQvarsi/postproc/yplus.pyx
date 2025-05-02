#!/usr/bin/env cython
#
# pyQvarsi, postproc.
#
# yplus routines.
#
# References for cython and MPI:
#	https://gist.github.com/shigh/6708484
#	https://stackoverflow.com/questions/51557135/how-to-pass-an-mpi-communicator-from-python-to-c-via-cython
#
# Last rev: 20/09/2021
from __future__ import print_function, division

import numpy as np
cimport cython, numpy as np

from libc.math   cimport sqrt, fmax, fmin
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

# Fix as Open MPI does not support MPI-4 yet, and there is no nice way that I know to automatically adjust Cython to missing stuff in C header files.
# Source: https://github.com/mpi4py/mpi4py/issues/525
cdef extern from *:
	"""
    #include <mpi.h>
    
    #if (MPI_VERSION < 3) && !defined(PyMPI_HAVE_MPI_Message)
    typedef void *PyMPI_MPI_Message;
    #define MPI_Message PyMPI_MPI_Message
    #endif
    
    #if (MPI_VERSION < 4) && !defined(PyMPI_HAVE_MPI_Session)
    typedef void *PyMPI_MPI_Session;
    #define MPI_Session PyMPI_MPI_Session
    #endif"
	"""
from mpi4py        cimport MPI
from mpi4py         import MPI

from ..utils.common import raiseError
from ..cr           import cr_start, cr_stop


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef int find_nan(double[:] xyz):
	'''
	Filter NaNs in xyz
	'''
	cdef int out = np.isnan(xyz[0]) or np.isnan(xyz[1]) or np.isnan(xyz[2])
	return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef int is_point_in_box(double[:] xyz, double[:] box):
	'''
	Find if a point is inside a box
	'''
	return 1 if xyz[0]<=box[1] and xyz[0]>=box[0] and xyz[1]<=box[3] and \
	xyz[1]>=box[2] and xyz[2]<=box[5] and xyz[2]>=box[4] else 0

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef int find_boxid_for_point(double[:] xyz, double[:,:] boxes):
	'''
	Find if a point is inside the boxes and return the id
	of the box that contains the point
	'''
	cdef int idbox = -1, ibox, nboxes = boxes.shape[0]
	for ibox in range(nboxes):
		if is_point_in_box(xyz,boxes[ibox,:]):
			idbox = ibox
			break
	return idbox

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef int find_pointid_in_box(double[:] xyz1, double[:,:] xyz, int[:] box_ids,
	int istart, int iend, double epsi):
	'''
	Find the ID of a point that is inside the box
	'''
	cdef int ii, ibox, point_id = -1
	cdef double diff2
	for ii in range(istart,iend):
		ibox  = box_ids[ii]
		diff2 = (xyz[ibox,0]-xyz1[0])*(xyz[ibox,0]-xyz1[0]) + \
				(xyz[ibox,1]-xyz1[1])*(xyz[ibox,1]-xyz1[1]) + \
				(xyz[ibox,2]-xyz1[2])*(xyz[ibox,2]-xyz1[2])
		if diff2 < epsi:
			point_id = ii
			break
	return point_id

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.int32_t,ndim=1] find_not_found(double[:] array, double val):
	'''
	Find the positions of the array where the value
	has not been found
	'''
	cdef int ii, npoints = array.shape[0], not_found = 0
	cdef np.ndarray[np.int32_t,ndim=1] out
	cdef int *buff
	# Find points that are not valid and copy their
	# ids to the buffer
	buff = <int*>malloc(npoints*sizeof(int))
	for ii in range(npoints):
		if array[ii] <= val:
			buff[not_found] = ii
			not_found += 1
	# Copy the buffer to the output
	if not_found > 0:
		out = np.ndarray((not_found,),np.int32)
		memcpy(&out[0],buff,not_found*sizeof(int))
	else:
		out = np.array([],np.int32)
	free(buff)
	return out


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=1] get_bounding_box(double[:,:] xyz):
	'''
	Get the bounding box of a domain
	'''	
	cdef int ip, n = xyz.shape[0]
	cdef np.ndarray[np.double_t,ndim=1] out = np.ndarray((6,),np.double)
	# Compute max and min from the domain
	out[1], out[0] = xyz[0,0], xyz[0,0]
	out[3], out[2] = xyz[0,1], xyz[0,1]
	out[5], out[4] = xyz[0,2], xyz[0,2]
	for ip in range(1,n):
		out[0] = fmin(out[0],xyz[ip,0])
		out[2] = fmin(out[2],xyz[ip,1])
		out[4] = fmin(out[4],xyz[ip,2])
		out[1] = fmax(out[1],xyz[ip,0])
		out[3] = fmax(out[3],xyz[ip,1])
		out[5] = fmax(out[5],xyz[ip,2])
	# Return
	return out


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=1] get_global_bounding_box(double[:,:] xyz, MPI.Comm comm):
	'''
	Obtain the global bounding box
	of a parallel domain.
	'''
	cdef int ii, MPI_size = comm.Get_size()
	cdef np.ndarray[np.double_t,ndim=1] local_bbox
	cdef np.ndarray[np.double_t,ndim=2] bboxes
	cdef np.ndarray[np.double_t,ndim=1] out = np.ndarray((6,),np.double)
	# Obtain the local bounding box for the subdomain
	local_bbox = get_bounding_box(xyz)
	# Serial run has all the domain - no need to continue
	if MPI_size == 1: return local_bbox
	# We have a partitioned domain here
	# Gather to all the local boxes
	bboxes = np.array(comm.allgather(local_bbox),np.double)
	# At this point all processors should have
	# a list of the local boxes
	out[1], out[0] = bboxes[0,1], bboxes[0,0]
	out[3], out[2] = bboxes[0,3], bboxes[0,2]
	out[5], out[4] = bboxes[0,5], bboxes[0,4]
	for ii in range(1,MPI_size):
		out[0] = fmin(out[0],bboxes[ii,0])
		out[2] = fmin(out[2],bboxes[ii,2])
		out[4] = fmin(out[4],bboxes[ii,4])
		out[1] = fmax(out[1],bboxes[ii,1])
		out[3] = fmax(out[3],bboxes[ii,3])
		out[5] = fmax(out[5],bboxes[ii,5])
	return out


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] build_domain_boxes(double[:] bbox,int nbx,int nby,int nbz,double fact):
	'''
	Splits the domain bouding box in separate boxes
	per each axis given a safety factor
	'''
	cdef int ibx, iby, ibz, ibox, nboxes = nbx*nby*nbz
	cdef double xmax, xmin, ymax, ymin, zmax, zmin, dbx, dby, dbz
	cdef double xbox1, xbox2, ybox1, ybox2, zbox1, zbox2
	cdef np.ndarray[np.double_t,ndim=2] boxes = np.zeros((nboxes,6),dtype=np.double)
	# Unpack bounding box
	xmin = bbox[0]
	ymin = bbox[2]
	zmin = bbox[4]
	xmax = bbox[1]
	ymax = bbox[3]
	zmax = bbox[5]
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
	ibox = 0
	for ibx in range(nbx):
		xbox1, xbox2 = xmin+ibx*dbx, xmin+(ibx+1)*dbx
		for iby in range(nby):
			ybox1, ybox2 = ymin+iby*dby, ymin+(iby+1)*dby
			for ibz in range(nbz):
				zbox1, zbox2 = zmin+ibz*dbz, zmin+(ibz+1)*dbz
				boxes[ibox,0] = xbox1
				boxes[ibox,1] = xbox2
				boxes[ibox,2] = ybox1
				boxes[ibox,3] = ybox2
				boxes[ibox,4] = zbox1
				boxes[ibox,5] = zbox2
				ibox += 1
	return boxes


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=1] compute_yplus_u(double mu, double[:,:] xyz, double[:,:] xyzd, 
	double[:,:] gradv, double[:] walld, int[:] wallid, double[:,:] boxes,
	int[:] boxes_np, int[:,:] boxes_ids, double epsi, int first):
	'''
	Compute yplus given a series of points
	'''
	cdef int ip, idbox, ii, ic, istart, iend, npoints = xyz.shape[0], nboxes = boxes.shape[0]
	cdef double u_tau

	cdef np.ndarray[np.double_t,ndim=1] xyzw    = np.ndarray((3,),dtype=np.double)
	cdef np.ndarray[np.int32_t,ndim=1]  node_id = np.zeros((nboxes,),dtype=np.int32)
	cdef np.ndarray[np.double_t,ndim=1] yplus   = -np.ones((xyz.shape[0],),dtype=np.double)

	# Loop the points
	for ip in range(npoints):
		# Treat NaNs
		if find_nan(xyz[ip,:]): continue

		# Find the coordinates of the node on the wall
		for ii in range(3): xyzw[ii] = xyz[ip,ii]
		if wallid[ip] == 0: xyzw[0] -= walld[ip]
		if wallid[ip] == 1: xyzw[0] += walld[ip]
		if wallid[ip] == 2: xyzw[1] -= walld[ip]
		if wallid[ip] == 3: xyzw[1] += walld[ip]
		if wallid[ip] == 4: xyzw[2] -= walld[ip]
		if wallid[ip] == 5: xyzw[2] += walld[ip]

		# Find the box that belongs to the point
		idbox = find_boxid_for_point(xyzw,boxes)
		# Crash here! Boxes are on the global domain
		# so the ID of the box should always be found.
		if idbox < 0: raiseError('Point %d [%f,%f,%f] not found inside the boxes!'%(ip,xyzw[0],xyzw[1],xyzw[2]))

		# Now we should have the id of the box, retrieve the points inside the box
		if first and boxes_np[idbox] == 0:
			ic = 0
			for ii in range(xyzd.shape[0]):
				if is_point_in_box(xyzd[ii,:],boxes[idbox,:]):
					boxes_np[idbox]    += 1
					boxes_ids[idbox,ic] = ii
					ic += 1

		# Now find the node id on a reduced search
		# case 1: boxes_np[idbox] == 0 -> no points of this domain inside target box
		if boxes_np[idbox] == 0: continue

		# Local search close to the previous node_id
		istart = 0               if ip == 0 else max(node_id[idbox]-1000,0)
		iend   = boxes_np[idbox] if ip == 0 else min(node_id[idbox]+1000,boxes_np[idbox])
		node_id[idbox] = find_pointid_in_box(xyzw,xyzd,boxes_ids[idbox,:],istart,iend,epsi)

		# If not found look at the remaining end
		if node_id[idbox] < 0:
			node_id[idbox] = find_pointid_in_box(xyzw,xyzd,boxes_ids[idbox,:],
				iend,boxes_np[idbox],epsi)

		# If still not found look at the ones at the beginning
		if node_id[idbox] < 0:
			node_id[idbox] = find_pointid_in_box(xyzw,xyzd,boxes_ids[idbox,:],
				0,istart,epsi)

		# case 2: node_id[idbox] < 0 -> no point has been found in idbox
		if node_id[idbox] < 0: continue # Point not in this subdomain

		# Friction velocity
		u_tau = 0.
		ic    = boxes_ids[idbox,node_id[idbox]]
		if wallid[ip] == 0: u_tau = sqrt(mu*abs(gradv[ic,0]))
		if wallid[ip] == 1: u_tau = sqrt(mu*abs(gradv[ic,0]))
		if wallid[ip] == 2: u_tau = sqrt(mu*abs(gradv[ic,1]))
		if wallid[ip] == 3: u_tau = sqrt(mu*abs(gradv[ic,1]))
		if wallid[ip] == 4: u_tau = sqrt(mu*abs(gradv[ic,2]))
		if wallid[ip] == 5: u_tau = sqrt(mu*abs(gradv[ic,2]))

		# Compute yplus
		yplus[ip] = u_tau*walld[ip]/mu
	return yplus


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def yplus_xyz_3D(double[:,:] xyz, double[:,:] gradv, double mu, 
	double[:] x_fnt, double[:] x_bck, double[:] y_top,
	double[:] y_bot, double[:] z_rgt, double[:] z_lft,
	int nbx=4, int nby=4, int nbz=4, double fact=0.1):
	'''
	Compute the yplus given the mesh points and the points of
	the wall assuming that the mesh is regular on the xyz directions.

	Warning: this code might have to be adapted for a parallel domain!

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

	cdef MPI.Comm comm = MPI.COMM_WORLD
	
	cdef int ii, ic, ip, icore, npoints = xyz.shape[0], nboxes = nbx*nby*nbz, not_found
	cdef int MPI_size = comm.Get_size(), MPI_rank = comm.Get_rank()
	cdef object xyz_not_found, walld_not_found, wallid_not_found, yplus_gather

	cdef int[:]      id_not_found
	cdef double[:]   bbox
	cdef double[:,:] boxes

	cdef np.ndarray[np.double_t,ndim=1] walld  = np.zeros((npoints,),dtype=np.double)
	cdef np.ndarray[np.int32_t,ndim=1]  wallid = -np.ones((npoints,),dtype=np.int32)
	cdef np.ndarray[np.double_t,ndim=1] walld6 = np.zeros((6,),dtype=np.double)

	cdef np.ndarray[np.int32_t,ndim=1]  boxes_np  = np.zeros((nboxes,),dtype=np.int32)
	cdef np.ndarray[np.int32_t,ndim=2]  boxes_ids = -np.ones((nboxes,npoints),dtype=np.int32)
		
	cdef np.ndarray[np.double_t,ndim=2] xyz_to_gather
	cdef np.ndarray[np.double_t,ndim=1] yplus, walld_to_gather, yplus_rank
	cdef np.ndarray[np.int32_t,ndim=1]  wallid_to_gather

	# Compute the global domain bounding box
	bbox = get_global_bounding_box(xyz,comm)

	# Split the domain into boxes in order to
	# speed up the search algorithm
	boxes = build_domain_boxes(bbox,nbx,nby,nbz,fact)

	# Distance to the wall
	# This is OK in parallel since the wall position is
	# known for all processors
	for ip in range(npoints):
		# Treat NaNs
		if find_nan(xyz[ip,:]): 
			walld[ip] = np.nan
			continue

		# Compute the distance to the wall for point ip
		walld6[0] = xyz[ip,0] - x_bck[ip]
		walld6[1] = x_fnt[ip] - xyz[ip,0]
		walld6[2] = xyz[ip,1] - y_bot[ip]
		walld6[3] = y_top[ip] - xyz[ip,1]
		walld6[4] = xyz[ip,2] - z_lft[ip]
		walld6[5] = z_rgt[ip] - xyz[ip,2]

		# Find the minimum distance and to which wall
		walld[ip] = 1e20
		for ii in range(6):
			if np.isnan(walld6[ii]): continue # Skip NaNs
			if walld6[ii]<walld[ip]:
				# Distance to the wall
				walld[ip]  = walld6[ii] 
				wallid[ip] = ii
		if wallid[ip] < 0: raiseError('Wall ID not found for point %d!'%ip)

	# Compute yplus for the points of the current subdomain
	yplus = compute_yplus_u(mu,xyz,xyz,gradv,walld,wallid,boxes,boxes_np,boxes_ids,1e-6,1)

	# Manage the points that we could not find inside the current domain
	id_not_found = find_not_found(yplus,-1)
	not_found    = id_not_found.shape[0]
	if MPI_size > 1:
		# Prepare arrays to gather
		xyz_to_gather    = np.ndarray((not_found,3),np.double)
		walld_to_gather  = np.ndarray((not_found,),np.double)
		wallid_to_gather = np.ndarray((not_found,),np.int32)
		for ii in range(not_found):
			ic = id_not_found[ii]
			xyz_to_gather[ii,0]  = xyz[ic,0]
			xyz_to_gather[ii,1]  = xyz[ic,1]
			xyz_to_gather[ii,2]  = xyz[ic,2]
			walld_to_gather[ii]  = walld[ic]
			wallid_to_gather[ii] = wallid[ic]
		# Gather from all the cores the points that could not be found
		xyz_not_found    = comm.allgather(xyz_to_gather)
		walld_not_found  = comm.allgather(walld_to_gather)
		wallid_not_found = comm.allgather(wallid_to_gather)
		# Per each core, look for the points that have not been found
		# and generate the yplus
		for icore in range(MPI_size):
			# Skip if there is no operation to do
			if xyz_not_found[icore].shape[0] == 0: continue 
			# All subdomains look for yplus given the xyz_rank
			yplus_rank = compute_yplus_u(mu,xyz_not_found[icore],xyz,gradv,
				walld_not_found[icore],wallid_not_found[icore],boxes,boxes_np,boxes_ids,1e-6,0)
			# Subdomain icore gathers all yplus computations
			yplus_gather = comm.gather(yplus_rank,root=icore)
			# Subdomain that gathers stores into yplus
			if icore == MPI_rank:
				for ip in range(MPI_size):
					for ii in range(not_found):
						ic = id_not_found[ii]
						yplus[ic] = fmax(yplus[ic],yplus_gather[ip][ii])
	else:
		# This is not a parallel run so crash if we could
		# not find any point using the algorithm
		if not_found > 0: raiseError('Some points (%d) have not been found!'%len(id_not_found))

	cr_stop('yplus_xyz_3D',0)
	return yplus, walld

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef inline tuple[int, double, double, double] findDistancesHexaVTK(int mynode, int[:] myconnec, double[:, :] coord, set surfMaskSet, double[:] uStreamwise, int nDims=3):
	cdef dict candidate_nodes = {
		0: [1, 3, 4],
		1: [2, 5, 0],
		2: [1, 3, 6],
		3: [0, 2, 7],
		4: [0, 5, 7],
		5: [1, 4, 6],
		6: [2, 5, 7],
		7: [3, 4, 6]
	}
	cdef:
		double distSurf[2][3]
		double dirSurf[2][3]
		int count = 0
		int firstNode = -1
		int iNode = myconnec[mynode]
		int cand_node
		int auxNode
		double kDist
		double auxDist[3]
		double iDist
		double jDist
		double projStreamwise[2][3]
		double norms[2]
		int idNorm
		int i
		int j

	for cand_node in candidate_nodes[mynode]:
		if myconnec[cand_node] not in surfMaskSet:
			firstNode = myconnec[cand_node]
			kDist = 0.0
			for i in range(nDims):
				kDist += (coord[firstNode, i] - coord[iNode, i]) ** 2
			kDist = sqrt(kDist)
		else:
			auxNode = myconnec[cand_node]
			for i in range(nDims):
				auxDist[i] = coord[auxNode, i] - coord[iNode, i]
			for i in range(nDims):
				distSurf[count][i] = auxDist[i]
			norm = 0.0
			for i in range(nDims):
				norm += distSurf[count][i] ** 2
			norm = sqrt(norm)
			for i in range(nDims):
				dirSurf[count][i] = distSurf[count][i] / norm
			count += 1

	for i in range(2):
		for j in range(nDims):
			projStreamwise[i][j] = 0.0
			for k in range(nDims):
				projStreamwise[i][j] += dirSurf[i][k] * uStreamwise[k]
			projStreamwise[i][j] *= dirSurf[i][j]

	for i in range(2):
		norms[i] = 0.0
		for j in range(nDims):
			norms[i] += projStreamwise[i][j] ** 2
		norms[i] = sqrt(norms[i])

	idNorm = 0
	if norms[1] > norms[0]:
		idNorm = 1

	if idNorm == 0:
		iDist = 0.0
		jDist = 0.0
		for i in range(nDims):
			iDist += distSurf[0][i] ** 2
			jDist += distSurf[1][i] ** 2
		iDist = sqrt(iDist)
		jDist = sqrt(jDist)
	else:
		iDist = 0.0
		jDist = 0.0
		for i in range(nDims):
			iDist += distSurf[1][i] ** 2
			jDist += distSurf[0][i] ** 2
		iDist = sqrt(iDist)
		jDist = sqrt(jDist)

	return firstNode, iDist, jDist, kDist

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cpdef computeWallDistancesSOD2D(int[:] surf_mask, int[:, :] connecvtk, double[:, :] coord, double[:] uStreamwise):
	cdef:
		set surfMaskSet = set(surf_mask)
		cdef np.ndarray[int, ndim=1] firstNodeMask = np.zeros(surf_mask.shape[0], dtype=np.int32)
		cdef np.ndarray[double, ndim=1] iDistVal = np.zeros(surf_mask.shape[0], dtype=np.double)
		cdef np.ndarray[double, ndim=1] jDistVal = np.zeros(surf_mask.shape[0], dtype=np.double)
		cdef np.ndarray[double, ndim=1] kDistVal = np.zeros(surf_mask.shape[0], dtype=np.double)
		int iNode
		int myelem
		int mynode
		int firstNode
		double iDist
		double jDist
		double kDist
		int nNode = len(surf_mask)

	for iNode in range(nNode):
		wallNode = surf_mask[iNode]
		for myelem in range(connecvtk.shape[0]):
			for mynode in range(connecvtk.shape[1]):
				if connecvtk[myelem, mynode] == surf_mask[iNode]:
					break
			if connecvtk[myelem, mynode] == surf_mask[iNode]:
				break
		myconnec = connecvtk[myelem, :]
		firstNode, iDist, jDist, kDist = findDistancesHexaVTK(mynode, myconnec, coord, surfMaskSet, uStreamwise)
		firstNodeMask[iNode] = firstNode
		iDistVal[iNode] = iDist
		jDistVal[iNode] = jDist
		kDistVal[iNode] = kDist

	return np.array(firstNodeMask), np.array(iDistVal), np.array(jDistVal), np.array(kDistVal)
