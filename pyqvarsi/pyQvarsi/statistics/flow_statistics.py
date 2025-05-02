#!/usr/bin/env python
#
# pyQvarsi, statistics.
#
# Statistics routines.
#
# Last rev: 11/11/2020
from __future__ import print_function, division

import numpy as np

from ..   import vmath as math
from ..cr import cr


@cr('stats.tripleCorr')
def tripleCorrelation(infield1,infield2,infield3):
	'''
	Computes the triple correlation.

	EXAMPLE USAGE
		velo3 = tripleCorrelation(veloc,veloc,veloc)

	IN:
		> infield1(nnod,nd1): input field to be averaged
		> infield2(nnod,nd2): input field to be averaged
		> infield3(nnod,nd3): input field to be averaged

	OUT:
		> output field(nnod,nd1*nd2*nd3)
	'''
	nd1, nd2, nd3 = infield1.shape[1], infield1.shape[1], infield1.shape[1]
	outfield = np.zeros((infield1.shape[0],nd1*nd2*nd3),dtype=infield1.dtype)
	iout = 0
	for i1 in range(nd1):
		for i2 in range(nd2):
			for i3 in range(nd3):
				outfield[:,iout] = infield1[:,i1]*infield2[:,i2]*infield3[:,i3]
				iout += 1
	return outfield


@cr('stats.ReynoldsST')
def reynoldsStressTensor(*args):
	'''
	Computes the Reynolds stress tensor.

	EXAMPLE USAGE
		mode 1 (from velocity fluctuations):
			Rij = reynoldsStressTensor(vfluct)
		mode 2 (from averaged quantities):
			Rij = reynoldsStressTensor(avvel,avve2,avvxy)

	IN:
		> veloc(nnod,3): velocity field (u,v,w)
		> velo2(nnod,3): velocity product (uu,vv,ww)
		> velxy(nnod,3): velocity cross product (uv,uw,vw)

	OUT:
		> output field(nnod,9)
	'''
	if len(args) == 1: # mode 1 (from velocity vector)
		return math.outer(args[0],args[0])
	if len(args) == 3: # mode 2 (from averaged quantities)
		Rij = np.zeros((args[0].shape[0],9),dtype=args[0].dtype)
		Rij[:,0] = args[1][:,0] - args[0][:,0]*args[0][:,0] # r_11 = avg(uu) - avg(u)*avg(u)
		Rij[:,1] = args[2][:,0] - args[0][:,0]*args[0][:,1] # r_12 = avg(uv) - avg(u)*avg(v)
		Rij[:,2] = args[2][:,1] - args[0][:,0]*args[0][:,2] # r_13 = avg(uw) - avg(u)*avg(w)
		Rij[:,3] = Rij[:,1]                                 # r_21 = avg(vu) - avg(v)*avg(u) = r_12
		Rij[:,4] = args[1][:,1] - args[0][:,1]*args[0][:,1] # r_22 = avg(vv) - avg(v)*avg(v)
		Rij[:,5] = args[2][:,2] - args[0][:,1]*args[0][:,2] # r_23 = avg(vw) - avg(v)*avg(w)
		Rij[:,6] = Rij[:,2]                                 # r_31 = avg(wu) - avg(w)*avg(u) = r_13
		Rij[:,7] = Rij[:,5]                                 # r_32 = avg(wv) - avg(w)*avg(v) = r_23
		Rij[:,8] = args[1][:,2] - args[0][:,2]*args[0][:,2] # r_33 = avg(ww) - avg(w)*avg(w)
		return Rij
	raiseError('Wrong number of input arguments %d in reynoldsStressTensor!'%len(args))


@cr('stats.strainTensor')
def strainTensor(gradv):
	'''
	Compute the strain tensor from the velocity gradient.
		S_ij = 0.5*(du_i/dx_j+du_j/dx_i)

	IN:
		> gradv(nnod,9): gradient of velocity

	OUT:
		> S_ij(nnod,9): strain tensor
	'''
	aux = np.zeros((gradv.shape[0],9),dtype=np.double)
	aux[:,0] = math.linopScaf(0.5,gradv[:,0],0.5,gradv[:,0]) # S_11
	aux[:,1] = math.linopScaf(0.5,gradv[:,1],0.5,gradv[:,3]) # S_12
	aux[:,2] = math.linopScaf(0.5,gradv[:,2],0.5,gradv[:,6]) # S_13
	aux[:,3] = math.linopScaf(0.5,gradv[:,3],0.5,gradv[:,1]) # S_21
	aux[:,4] = math.linopScaf(0.5,gradv[:,4],0.5,gradv[:,4]) # S_22
	aux[:,5] = math.linopScaf(0.5,gradv[:,5],0.5,gradv[:,7]) # S_23
	aux[:,6] = math.linopScaf(0.5,gradv[:,6],0.5,gradv[:,2]) # S_31
	aux[:,7] = math.linopScaf(0.5,gradv[:,7],0.5,gradv[:,5]) # S_32
	aux[:,8] = math.linopScaf(0.5,gradv[:,8],0.5,gradv[:,8]) # S_33
	return aux


@cr('stats.vortTensor')
def vorticityTensor(gradv):
	'''
	Compute the vorticity tensor from the velocity gradient.
		O_ij = 0.5*(du_i/dx_j-du_j/dx_i)

	IN:
		> gradv(nnod,9): gradient of velocity

	OUT:
		> O_ij(nnod,9): vorticity tensor
	'''
	aux = np.zeros((gradv.shape[0],9),dtype=np.double)
	aux[:,0] = math.linopScaf(0.5,gradv[:,0],-0.5,gradv[:,0]) # O_11
	aux[:,1] = math.linopScaf(0.5,gradv[:,1],-0.5,gradv[:,3]) # O_12
	aux[:,2] = math.linopScaf(0.5,gradv[:,2],-0.5,gradv[:,6]) # O_13
	aux[:,3] = math.linopScaf(0.5,gradv[:,3],-0.5,gradv[:,1]) # O_21
	aux[:,4] = math.linopScaf(0.5,gradv[:,4],-0.5,gradv[:,4]) # O_22
	aux[:,5] = math.linopScaf(0.5,gradv[:,5],-0.5,gradv[:,7]) # O_23
	aux[:,6] = math.linopScaf(0.5,gradv[:,6],-0.5,gradv[:,2]) # O_31
	aux[:,7] = math.linopScaf(0.5,gradv[:,7],-0.5,gradv[:,5]) # O_32
	aux[:,8] = math.linopScaf(0.5,gradv[:,8],-0.5,gradv[:,8]) # O_33
	return aux


@cr('stats.TKE')
def TKE(arg):
	'''
	Computes the turbulent kinetic energy (TKE).

	EXAMPLE USAGE
		mode 1 (assume avve2):
			k = TKE(avve2)
		mode 2 (assume R_ij):
			k = TKE(R_ij)

	IN (either):
		> velo2(nnod,3): velocity product (uu,vv,ww)
		> R_ij(nnod,9):  Reynolds stress tensor 

	OUT:
		> output field(nnod)
	'''
	if arg.shape[1] == 3: # Assume avve2
		return 0.5*math.dot(arg,arg)
	if arg.shape[1] == 9: # Assume R_ij
		return 0.5*math.trace(arg)
	raiseError('Wrong number of input arguments %d in TKE!'%len(arg))


@cr('stats.dissipation')
def dissipation(nu,s_ij):
	'''
	Computes the turbulent dissipation (epsilon).

	EXAMPLE USAGE
		epsilon = dissipation(mu,s_ij)

	IN:
		> mu:          Kinematic viscosity
		> sij(nnod,9): Fluctuating strain tensor

	OUT:
		> output field(nnod)
	'''
	return 2.*nu*math.doubleDot(s_ij,s_ij)


@cr('stats.taylorMicro')
def taylorMicroscale(nu,k,e,small=1e-6):
	'''
	Computes the Taylor microscale.

	EXAMPLE USAGE
		epsilon = taylorMicroscale(mu,rho,k,epsilon)

	IN:
		> nu:      Kinematic viscosity
		> k(nnod): Turbulent kinetic energy
		> e(nnod): Turbulent dissipation

	OUT:
		> output field(nnod)
	'''
	return np.sqrt(np.abs(10.*nu*k/(e+small)))


@cr('stats.kolmLS')
def kolmogorovLengthScale(nu,e,small=1e-6):
	'''
	Computes the Kolmogorov length scale.

	EXAMPLE USAGE
		epsilon = computeKolmogorovLengthScale(mu,rho,epsilon)

	IN:
		> nu:      Kinematic viscosity
		> e(nnod): Turbulent dissipation

	OUT:
		> output field(nnod)
	'''
	return np.abs( nu*nu*nu/(e+small) )**0.25


@cr('stats.kolmTS')
def kolmogorovTimeScale(nu,e,small=1e-6):
	'''
	Computes the Kolmogorov time scale.

	EXAMPLE USAGE
		epsilon = computeKolmogorovTimeScale(mu,rho,epsilon)

	IN:
		> nu:      Kinematic viscosity
		> e(nnod): Turbulent dissipation

	OUT:
		> output field(nnod)
	'''
	return np.sqrt( np.abs(nu/(e+small)) )