from __future__ import print_function, division

import mpi4py 
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import os, re, glob, subprocess, math, numpy as np
import pyAlya

import torch
import torch.nn.functional as F
import numpy as np

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def read_instants(CASEDIR):
	steps = []
	for file in glob.glob(CASEDIR+"*AVVEL*post*"):
		steps.append(int(re.split('[- .]',file)[-4]))
	steps.sort()
	return steps

def averaging(instants,mesh,VARLIST,CASEDIR,CASESTR):

	# Read the first instant of the list
	_, header = pyAlya.Field.read(CASESTR,VARLIST,instants[0],mesh.xyz,basedir=CASEDIR)

	# Initialize field at zero
	avgField = pyAlya.Field(xyz = mesh.xyz, AVVEL = mesh.newArray(ndim=3))
	time     = header.time
	for instant in instants[1:]:
		field, header = pyAlya.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=CASEDIR)

#		if mpi_rank == 0: continue # Skip master

		# Compute time-weighted average (Welford's online algorithm)
		dt   = header.time - time # weight
		time = header.time        # sum_weight
		avgField["AVVEL"] += pyAlya.stats.addS1(avgField["AVVEL"], field["AVVEL"], w=dt/time)

	return avgField, time 

def averaging2(instants,mesh,VARLIST,CASEDIR,CASESTR):

	# Read the first instant of the list
	_, header = pyAlya.Field.read(CASESTR,VARLIST,instants[0],mesh.xyz,basedir=CASEDIR)

	# Initialize the average
	avgField = pyAlya.Field(xyz = mesh.xyz, AVVEL = mesh.newArray(ndim=3), AVPRE = mesh.newArray())
	time = header.time
	total_time=0.0
	for instant in instants[1:]:
		field, header = pyAlya.Field.read(CASESTR,VARLIST,instant,mesh.xyz,basedir=CASEDIR)

		# Compute time-weighted average 
		dt   = header.time - time # weight
		time = header.time        # time to compute weigth in the next iteration
		for v in avgField.varnames:
			avgField[v]   += field[v] * dt
		total_time += dt
	return avgField / total_time, total_time

def plane_generation(Length,nx,ny):

	# Generate partition table
	ptable = pyAlya.PartitionTable.new(1,nelems=(nx-1)*(ny-1),npoints=nx*ny)

	# Generate points
	points = np.array([
		[-Length,-Length,0.0],
		[ Length,-Length,0.0],
		[ Length, Length,0.0],
		[-Length, Length,0.0]
		],dtype='double')

	# Generate plane mesh
	#return pyAlya.Mesh.plane(points[0],points[1],points[3],nx,ny,ngauss=1,ptable=ptable,create_elemList=False)
	#return pyAlya.Mesh.plane(points[0],points[1],points[3],nx,ny,ngauss=1,ptable=ptable)
	return pyAlya.Mesh.plane(points[0],points[1],points[3],nx,ny,ngauss=1,ptable=ptable,compute_massMatrix=False)

'''
def plane_generation(Length,nx,ny):

	# Generate partition table
	ptable = pyAlya.PartitionTable.new(1,nelems=(nx-1)*(ny-1),npoints=nx*ny)

	# Generate points
	points = np.array([
		[0.0,0.0,0.0],
		[Length,0.0,0.0],
		[Length,Length,0.0],
		[0.0,Length,0.0]
		],dtype='double')

	# Generate plane mesh
	return pyAlya.Mesh.plane(points[0],points[1],points[3],nx,ny,ngauss=1,ptable=ptable,create_elemList=False)
'''

def meteo_fields(avgField,mesh):

	# Compute the gradients of the velocity and the pressure
	avgField['GRAVZ'] = mesh.gradient(avgField['AVVEL'])[:,[0,2,4,5,8]]
	avgField['GRAPZ'] = mesh.gradient(avgField['AVPRE'])[:,2]
	return avgField

def fields_h5(int_xyz,mesh,avgField,target_mask,POSTDIR,metadata,basename,nanval=0.0):

	# Field generation
	pyAlya.cr_start('fields_h5',0)

	# Interpolation
	print(0,'Initiating interpolation',flush=True)
	int_field = mesh.interpolate(int_xyz,avgField,method='FEM',fact=3.,ball_max_iter=5,global_max_iter=1,target_mask=target_mask)

	target_mask = target_mask.astype(int)

	# Change NaNs to a value
	for v in int_field.varnames:
		int_field[v][~target_mask] = nanval
	pyAlya.pprint(0,'Interpolated',flush=True)
	pyAlya.cr_stop('fields_h5',0)
	return int_field

def compute_u_grads(fields,N_POINTS,D_LENGTH):

	GRID_SPACING=(D_LENGTH*2)/(N_POINTS-1)

	U=fields['AVVEL'][:,0]
	V=fields['AVVEL'][:,1]
	channels=1

	U=np.nan_to_num(U,nan=0.0)
	V=np.nan_to_num(V,nan=0.0)
        
	U=np.reshape(U,(N_POINTS,N_POINTS), order='C')
	U=np.flip(U,axis=0)
	V=np.reshape(V,(N_POINTS,N_POINTS), order='C')
	V=np.flip(V,axis=0)

	U=torch.from_numpy(U.copy())
	V=torch.from_numpy(V.copy())

	U=torch.unsqueeze(U,dim=0)
	V=torch.unsqueeze(V,dim=0)

	U.double()
	V.double()

	dx=GRID_SPACING
	dy=GRID_SPACING

	grad_x_weights = torch.tensor([
 		[0.0, 0.0, 0.0],
    	[-1.0/(2.0*dx), 0.0,1.0/(2.0*dx)],
    	[0.0, 0.0, 0.0]
	], dtype=torch.float64)

	grad_y_weights = torch.tensor([
	    [0.0, 1.0/(2.0*dy), 0.0],
	    [0.0, 0.0,0.0],
	    [0.0, -1.0/(2.0*dy), 0.0]
	], dtype=torch.float64)

	grad_x_weights = grad_x_weights.expand(channels, 1, 3, 3)
	grad_y_weights = grad_y_weights.expand(channels, 1, 3, 3)

	grad_Ux = F.conv2d(U, grad_x_weights, groups=U.shape[0], padding=1)
	grad_Vy = F.conv2d(V, grad_y_weights, groups=V.shape[0], padding=1)

	dim_x=N_POINTS
	dim_y=N_POINTS

	grad_Ux[0,:,0]=(U[0,:,1]-U[0,:,0])/dx
	grad_Ux[0,:,dim_x-1]=(U[0,:,dim_x-1]-U[0,:,dim_x-2])/dx

	grad_Vy[0,0,:]=(V[0,0,:]-V[0,1,:])/dy
	grad_Vy[0,dim_y-1,:]=(V[0,dim_y-2,:]-V[0,dim_y-1,:])/dy

	grad_Wz=(-1.0*grad_Ux)+(-1.0*grad_Vy)

	GRDUX=np.flip(grad_Ux[0].numpy(),axis=0)
	GRDUX=GRDUX.flatten(order='C')
	GRDVY=np.flip(grad_Vy[0].numpy(),axis=0)
	GRDVY=GRDVY.flatten(order='C')
	GRDWZ=np.flip(grad_Wz[0].numpy(),axis=0)
	GRDWZ=GRDWZ.flatten(order='C')

	return {'GRDUX':GRDUX,'GRDVY':GRDVY,'GRDWZ':GRDWZ}

def supress_nans(fields,VARLIST):

	#print("fields shape=",fields)
	#print("var=",VARLIST[0])

	field=fields[VARLIST[0]]
	size=len(field)
	dimx=math.sqrt(size)

	#print(f"rank={mpi_rank} type={type(field)} shape={field.shape} shape1 {field.shape[0]} shape2 {field.shape[1]}")
	#print("field 0=",field[0])

	#isNaN=True
	#while isNaN:
		#isNaN=False
		#for idx,i in np.ndenumerate(field):
	for idx in range (field.shape[0]):

		size_L=field.shape[0]

		#print("size_L=",size_L)
		#print("field val=",field[idx]," field shape=",field[idx].shape)
			
			
		#coord=fields['xyz'][idx]
		#print(f"rank={mpi_rank} loc_idx={idx} coord{coord}")
			
		#point=np.array([-52.0,90.0,0.0])
		#isPoint=False
		#if np.linalg.norm(coord-point)<1e-1: isPoint=True
			
		#if isPoint: print(f"punt {coord} localitzat al rank={mpi_rank}")
		#if isPoint and mpi_rank==0: print(f"punt {coord} localitzat al rank={mpi_rank}")
			
		if field.shape[1]==1: val=field[idx]
		if field.shape[1]==3: val=field[idx,0]
			
		#if isPoint and mpi_rank==0: print("valor al punt=",val," not eq=", val==val, "mask=",fields['MASK'][idx])
			
		if val==val or fields['MASK'][idx]==0: 
			#if isPoint and mpi_rank==0: print("is continuing!!! valor al punt=",val," not eq=", val==val, "mask=",fields['MASK'][idx])
			continue

		#if mpi_rank==0: print("is not continuing at point=",coord," val==val=",val==val," mask=",fields['MASK'][idx]," avvel=",fields['AVVEL'][idx])
		#isNaN=True

		Eidx=idx+1
		Widx=idx-1

		if Widx>-1 and Widx<=size_L and field[Widx][0]==field[Widx][0]: 
			for v in VARLIST: fields[v][idx]=fields[v][Widx]
			if mpi_rank==0: print("west value=",fields['AVVEL'][Widx]) 
		elif Eidx>-1 and Eidx<=size_L and field[Eidx][0]==field[Eidx][0]:
			for v in VARLIST: fields[v][idx]=fields[v][Eidx]
			if mpi_rank==0: print("east value=",fields['AVVEL'][Eidx]) 
				
		#if isPoint: exit(0)
			

