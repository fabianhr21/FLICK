#!/usr/bin/env python
#
# pyQvarsi, statistics.
#
# RANS equations budgets.
# Here are functions to compute the budgets from the Reynolds
# stress equations as defined in the HiFiTurb document
#
#	R_ij,t + C_ij = P_ij + D1_ij + D2_ij + D3_ij + Phi_ij - Epsi_ij
#
# where,
#	C_ij    = (R_ij <u>_k),k                  (convection) 
#	P_ij    = -(R_ik<u>_j,k + R_jk<u>_i,k)    (production)
#	D1_ij   = -rho*(<u'_iu'_ju'_k>),_k        (turbulent diffusion 1)	
#	D2_ij   = -(<p'(u'_id_ij + u'_jd_ik)>)_,k (turbulent diffusion 2)
#	D3_ij   = mu*(R_ij)_,kk                   (molecular diffusion)
#	Phi_ij  = 2<p'S'_ij>                      (pressure strain correlation)
#	Epsi_ij = 2*mu*<u'_i,k u'_j,k>            (dissipation)
#
# Last rev: 16/02/2021
from __future__ import print_function, division

import numpy as np

from ..   import vmath as math
from ..cr import cr


@cr('budgets.convection')
def convectionBudget(avvel,gradRij):
	'''
	Computes the convection ((R_ij <u>_k),k) from 
	the Reynolds stress equations as defined in the 
	HiFiTurb document.

	No accumulation is needed.

	IN:
		> avvel(nnod,3):    averaged velocity 
		> gradRij(nnod,27): gradient of the Reynolds stress tensor

	OUT:
		> C_ij(nnod,9):     convection
	'''
	# dR_ij/dx*<u> + dR_ij/dy*<v> + dR_ij/dz*<w>
	return math.scaTensProd(avvel[:,0],gradRij[:,[0,3,6,9,12,15,18,21,24]]) + \
	       math.scaTensProd(avvel[:,1],gradRij[:,[1,4,7,10,13,16,19,22,25]]) + \
	       math.scaTensProd(avvel[:,2],gradRij[:,[2,5,8,11,14,17,20,23,26]])


@cr('budgets.production')
def productionBudget(R_ij,gradv):
	'''
	Computes the production (-(R_ik<u>_j,k + R_jk<u>_i,k)) from 
	the Reynolds stress equations as defined in the 
	HiFiTurb document.

	No accumulation is needed.

	IN:
		> R_ij(nnod,9):  Reynolds stress tensor
		> gradv(nnod,9): gradient of averaged velocity 

	OUT:
		> P_ij(nnod,9):  production
	'''
	return -math.linopArrf(1,math.matmul(R_ij,math.transpose(gradv)),1,math.matmul(gradv,math.transpose(R_ij)))


@cr('budgets.D1')
def turbulentDiffusion1Budget(rho,avve3,mesh,on_Gauss=False):
	'''
	Computes the turbulent diffusion (-rho*(<u'_iu'_ju'_k>),_k) from
	the Reynolds stress equations as defined in the 
	HiFiTurb document.

	No accumulation is needed.

	IN:
		> rho:            density
		> avve3(nnod,27): averaged velocity triple correlation
		> mesh:			  mesh class to compute the derivatives

	OUT:
		> D1_ij(nnod,9):  turbulent diffusion 1
	'''
	return -rho*mesh.divergence(avve3,on_Gauss=on_Gauss)


@cr('budgets.D1')
def turbulentDiffusion2Budget(avpve,mesh,on_Gauss=False):
	'''
	Computes the turbulent diffusion (-(<p'(u'_id_ij + u'_jd_ik)>)_,k) from
	the Reynolds stress equations as defined in the 
	HiFiTurb document.

	No accumulation is needed.

	IN:
		> avpve(nnod,3): averaged pressure velocity correlation
		> mesh:			 mesh class to compute the derivatives

	OUT:
		> D2_ij(nnod,9):  turbulent diffusion 2
	'''
	# Compute the Theta_ijk
	gavpv = mesh.gradient(avpve,on_Gauss=on_Gauss)
	return -math.linopArrf(1,math.matmul(gavpv,math.identity(gavpv)),1,math.matmul(math.transpose(gavpv),math.identity(gavpv)))


@cr('bugdets.D3')
def molecularDiffusionBudget(mu,R_ij,mesh,on_Gauss=False):
	'''
	Computes the molecular diffusion (mu*(R_ij)_,kk) from
	the Reynolds stress equations as defined in the 
	HiFiTurb document.

	No accumulation is needed.

	IN:
		> mu:            dynamic viscosity
		> R_ij(nnod,9):  Reynolds stress tensor
		> mesh:			 mesh class to compute the derivatives

	OUT:
		> D3_ij(nnod,9): molecular diffusion
	'''
	return mu*mesh.laplacian(R_ij,on_Gauss=on_Gauss)


@cr('bugdets.pStrain')
def pressureStrainBudget(pfluct,s_ij):
	'''
	Computes the pressure strain correlation (p'S'_ij) from 
	the Reynolds stress equations as defined in the 
	HiFiTurb document.

	Accumulation needs to be done apart using the addS1 routine.

	IN:
		> pfluct(nnod,):  pressure fluctuations  
		> s_ij(nnod,9):   fluctuating strain tensor

	OUT:
		> Phi_ij(nnod,9): pressure strain correlation
	'''
	return 2.*math.scaTensProd(pfluct,s_ij)


@cr('bugdets.dissi')
def dissipationBudget(mu,gradv):
	'''
	Computes the dissipation (2*mu*u'_i,k u'_j,k) from 
	the Reynolds stress equations as defined in the 
	HiFiTurb document.

	Accumulation needs to be done apart using the addS1 routine.

	IN:
		> mu:              dynamic viscosity
		> gradv(nnod,9):   gradient of the fluctuating velocity

	OUT:
		> Epsi_ij(nnod,9): dissipation
	'''
	# 2*mu*u'_i,k u'_j,k = 2*grad(u')*grad(u')^t
	return 2.*mu*math.matmul(gradv,math.transpose(gradv)) if not isinstance(mu,np.ndarray) else 2.*math.scaTensProd(mu,math.matmul(gradv,math.transpose(gradv)))