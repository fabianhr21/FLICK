#!/usr/bin/env python
#
# pyQvarsi, postproc.
#
# Vortex detection algorithms.
#
# Last rev: 29/09/2020
from __future__ import division, print_function

import mpi4py
import numpy as np

from ..                           import vmath as math
from ..cr                         import cr
from ..statistics.flow_statistics import strainTensor, vorticityTensor
from ..utils.common               import raiseError
from ..utils.parallel             import mpi_reduce


@cr('vorticity')
def vorticity(gradv):
	'''
	Compute the vorticity from the velocity gradient.

	IN:
			> gradv(nnod,{4,6,9}): gradient of velocity

	OUT:
			> vorticity(nnod,{1,3}): vorticity vectorial or scalar field
	'''
	if gradv.shape[1] < 7:  # 2x2 or 3x2
		vorticity = gradv[:, 2] - gradv[:, 1]  # dv/dx - du/dy
	elif gradv.shape[1] == 9:
		vorticity = np.zeros((gradv.shape[0], 3), dtype=np.double)
		vorticity[:, 0] = gradv[:, 7] - gradv[:, 5]  # dw/dy - dv/dz
		vorticity[:, 1] = gradv[:, 2] - gradv[:, 6]  # du/dz - dw/dx
		vorticity[:, 2] = gradv[:, 3] - gradv[:, 1]  # dv/dx - du/dy
	else:
		raiseError('Velocity gradient tensor has wrong shape.')
	return vorticity


@cr('QCriterion')
def QCriterion(gradv):
	'''
	Compute the Q criterion from the velocity gradient.

	IN:
			> gradv(nnod,9): gradient of velocity

	OUT:
			> Q-criterion(nnod,): Q-criterion scalar field
	'''
	return - (gradv[:, 0]*gradv[:, 0]+gradv[:, 4]*gradv[:, 4]+gradv[:, 8]*gradv[:, 8])/2. \
		   - (gradv[:, 1]*gradv[:, 3]+gradv[:, 2]
		   * gradv[:, 6]+gradv[:, 5]*gradv[:, 7])


@cr('Lambda2Criterion')
def Lambda2Criterion(gradv):
	'''
	Compute the lambda2 criterion from the velocity gradient.

	IN:
			> gradv(nnod,9): gradient of velocity

	OUT:
			> l2-criterion(nnod,): l2-criterion scalar field
	'''
	# Compute strain and vorticity tensor
	S = strainTensor(gradv)
	O = vorticityTensor(gradv)
	# Compute tensor S*S + O*O
	R = math.linopArrf(1, math.matmul(S, S), 1, math.matmul(O, O))
	# Compute the eigenvalues of R so that l1 > l2 > l3
	l2 = math.eigenvalues(R)[:, 1]
	return l2


@cr('OmegaCriterion')
def OmegaCriterion(gradv, epsilon=0.0001, modified=False):
	'''
	Omega-Criterion for vortex identification from:
			C. Liu, Y. Wang, Y. Yang, and Z. Duan, "New omega vortex identification method"
			Sci. China Physics, Mech. Astron., vol. 59, no. 8, p. 684711, Aug. 2016.

	For setting the value of epsilon:
			X. Dong, Y. Wang, X. Chen, Y. Dong, Y. Zhang, and C. Liu, "Determination of epsilon for Omega vortex identification method"
			J. Hydrodyn., vol. 30, no. 4, pp. 541-548, 2018.

	IN:
			> gradv(nnod,9): gradient of velocity

	OUT:
			> Omega-criterion(nnod,): Omega-criterion scalar field
	'''
	# Compute a
	a = 0.5*(gradv[:, 1] + gradv[:, 3])*(gradv[:, 1] + gradv[:, 3]) + \
		0.5*(gradv[:, 2] + gradv[:, 6])*(gradv[:, 2] + gradv[:, 6]) + \
		0.5*(gradv[:, 5] + gradv[:, 7])*(gradv[:, 5] + gradv[:, 7]) + \
		gradv[:, 0]*gradv[:, 0] + gradv[:, 4] * \
		gradv[:, 4] + gradv[:, 8]*gradv[:, 8]
	# Compute b
	b = 0.5*(gradv[:, 1] - gradv[:, 3])*(gradv[:, 1] - gradv[:, 3]) + \
		0.5*(gradv[:, 2] - gradv[:, 6])*(gradv[:, 2] - gradv[:, 6]) + \
		0.5*(gradv[:, 5] - gradv[:, 7])*(gradv[:, 5] - gradv[:, 7])
	# Compute maximum epsilon
	eps = epsilon*mpi_reduce(np.nanmax(b-a), op='nanmax', all=True)
	# Return Omega
	out = b/math.maxVal(a+b,eps) if not modified else (b+eps)/(a+b+2.*eps)
	return out


def _RortexCriterionShur(gradv):
    '''
    Rortex-Criterion for vortex identification from:
            Liu, C., Gao, Y., Tian, S., Dong, X., 2018. Rortex—A new vortex vector definition and vorticity tensor and vector decompositions. 
            Physics of Fluids 30, 035103. https://doi.org/10.1063/1.5023001

            Gao, Y., Liu, C., 2018. Rortex and comparison with eigenvalue-based vortex identification criteria. 
            Physics of Fluids 30, 085107. https://doi.org/10.1063/1.5040112

    IN:
            > gradv(nnod,9): gradient of velocity

    OUT:
            > Rortex-criterion(nnod,3): Rortex-criterion vector
    '''    
    # Compute the schur decomposition of the transposed gradient
    # and sorting the eigenvalues
    S, Q = math.schur(math.transpose(gradv))
    # det(Q) can only be 1 or -1
    Qt    = math.transpose(Q)
    detQ  = math.det(Q)
    # For det(Q) > 0 just rewrite Q
    mask    = detQ > 0.
    Q[mask] = Qt[mask]
    # For det(Q) < 0 rotate Q using Rmat
    mask = detQ < 0.
    Rmat = np.zeros((gradv.shape[0],9),np.double)
    Rmat[:,0] =  1.
    Rmat[:,4] =  1.
    Rmat[:,8] = -1.
    Q[mask] = math.matmul(Rmat[mask,:],Qt[mask])
    del Rmat
    # Rewrite gradV
    gradV = math.transpose(math.matmul(Q,math.matmul(gradv,math.transpose(Q))))
    # Axis of rotation
    rrot = np.zeros((gradv.shape[0],3),np.double)
    rrot[:,2] = 1.
    r = math.tensVecProd(math.transpose(Q),rrot)
    # alpha and beta
    alpha = 0.5*np.sqrt((gradV[:, 4]-gradV[:, 0])*(gradV[:, 4]-gradV[:, 0]) +
                        (gradV[:, 1]+gradV[:, 3])*(gradV[:, 1]+gradV[:, 3]))
    beta  = 0.5*(gradV[:, 1]-gradV[:, 3])
    # Rortex magnitude
    Rm = np.zeros((gradv.shape[0],), np.double)
    id1 = np.logical_and((alpha*alpha - beta*beta) < 0., beta > 0.)
    id2 = np.logical_and((alpha*alpha - beta*beta) < 0., beta < 0.)
    Rm[id1] = 2.*(beta[id1] - alpha[id1])
    Rm[id2] = 2.*(beta[id2] + alpha[id2])
    # Rortex vector
    rortex = math.scaVecProd(Rm,r)
    return rortex

def _RortexCriterionFast(gradv):
	'''
	Rortex-Criterion for vortex identification from:
			Liu, C., Gao, Y., Tian, S., Dong, X., 2018. Rortex—A new vortex vector definition and vorticity tensor and vector decompositions. 
			Physics of Fluids 30, 035103. https://doi.org/10.1063/1.5023001

			Gao, Y., Liu, C., 2018. Rortex and comparison with eigenvalue-based vortex identification criteria. 
			Physics of Fluids 30, 085107. https://doi.org/10.1063/1.5040112

	IN:
			> gradv(nnod,9): gradient of velocity

	OUT:
			> Rortex-criterion(nnod,3): Rortex-criterion vector
	'''    
	# Calculation of characteristic equation parameters P, Q, R
	P   = -math.trace(gradv)
	Q   = gradv[:, 0]*gradv[:, 4] + gradv[:, 4]*gradv[:, 8] + gradv[:, 0]*gradv[:, 8]
	R   = -(gradv[:, 0]*(gradv[:, 4]*gradv[:, 8] - gradv[:, 5]*gradv[:, 7]) 
		  - gradv[:, 1]*(gradv[:, 3]*gradv[:, 8] - gradv[:, 5]*gradv[:, 6]) 
		  + gradv[:, 2]*(gradv[:, 3]*gradv[:, 7] - gradv[:, 4]*gradv[:, 6]))

	# Calculation of condition real eigenvalue existence condition.
	# CONDIITON: If T^2 > S^3 there is one real eigenvalue (lambda3) in dv
	S   = (P*P - 3.*Q)/9.
	T   = (2.*P*P*P - 9.*P*Q + 27.*R)/54.
	ids = T*T - S*S*S
	mask = ids > 0

	# Calculation of the real eigenvalue of dv tensor
	A 		= np.zeros((gradv.shape[0],), np.double)
	B       = np.zeros((gradv.shape[0],), np.double)
	lambda3 = np.zeros((gradv.shape[0],), np.double)
	A[mask]       = -np.sign(T[mask])*( np.abs(T[mask]) + np.sqrt(ids[mask]) )**(1./3.)
	B[A != 0.]    = S[A != 0.]/A[A != 0.]
	lambda3[mask] = A[mask] + B[mask] - P[mask]/3.
	del P, Q, R, S, T, A, B
	
	# Re-definition of the velocity gradient tensor
	gradv[:,0] = gradv[:,0] - lambda3
	gradv[:,4] = gradv[:,4] - lambda3
	gradv[:,8] = gradv[:,8] - lambda3

	# Calcuation of the minors of the velocity gradient tensor
	delta = np.zeros((gradv.shape[0],3), np.double)
	delta[:,0] = np.abs(gradv[:,4]*gradv[:,8] - gradv[:,5]*gradv[:,7])
	delta[:,1] = np.abs(gradv[:,0]*gradv[:,8] - gradv[:,2]*gradv[:,6])
	delta[:,2] = np.abs(gradv[:,0]*gradv[:,4] - gradv[:,1]*gradv[:,3])
	ids = np.argmax(delta,axis=1)
	
	# Normalization of the real eigenvector
	r_str = np.zeros((gradv.shape[0],3), np.double)
	H     = np.zeros((gradv.shape[0],), np.double)
	# x direction is the dominant
	mask = ids == 0
	H[mask]       = gradv[mask,4]*gradv[mask,8] - gradv[mask,5]*gradv[mask,7]
	r_str[mask,0] = 1
	r_str[mask,1] = ( -gradv[mask,8]*gradv[mask,3] + gradv[mask,5]*gradv[mask,6] )/H[mask]
	r_str[mask,2] = ( -gradv[mask,4]*gradv[mask,6] + gradv[mask,7]*gradv[mask,3] )/H[mask]
	# y direction is the dominant
	mask = ids == 1
	H[mask]       = gradv[mask,0]*gradv[mask,8] - gradv[mask,2]*gradv[mask,6]
	r_str[mask,0] = ( -gradv[mask,8]*gradv[mask,1] + gradv[mask,2]*gradv[mask,7] )/H[mask]
	r_str[mask,1] = 1
	r_str[mask,2] = ( -gradv[mask,0]*gradv[mask,7] + gradv[mask,6]*gradv[mask,1] )/H[mask]
	# z direction is the dominant
	mask = ids == 2
	H[mask]       = gradv[mask,0]*gradv[mask,4] - gradv[mask,1]*gradv[mask,3]
	r_str[mask,0] = ( -gradv[mask,4]*gradv[mask,2] + gradv[mask,1]*gradv[mask,5] )/H[mask]
	r_str[mask,1] = ( -gradv[mask,0]*gradv[mask,5] + gradv[mask,3]*gradv[mask,2] )/H[mask]
	r_str[mask,2] = 1

	# Normalized real eigenvector
	eigv_norm = math.scaVecProd(1./math.vecNorm(r_str),r_str)
	del H, r_str, delta

	# Calculation of rotation matrix Q_str using Rodrigues-Formula
	aux      = np.zeros((gradv.shape[0],3), np.double)
	aux[:,2] = 1.
	c     = math.dot(eigv_norm, aux)
	phi   = np.arccos(c)
	c_phi = np.cos(phi)
	s_phi = np.sin(phi)
	Q_str = np.zeros((gradv.shape[0], 9))
	Q_str[:,0] = ( c_phi + eigv_norm[:,0]*eigv_norm[:,0]*( 1 - c_phi ) )
	Q_str[:,1] = ( eigv_norm[:,0]*eigv_norm[:,1]*( 1 - c_phi ) - eigv_norm[:,2]*s_phi )
	Q_str[:,2] = ( eigv_norm[:,0]*eigv_norm[:,2]*( 1 - c_phi ) + eigv_norm[:,1]*s_phi )
	Q_str[:,3] = ( eigv_norm[:,1]*eigv_norm[:,0]*( 1 - c_phi ) + eigv_norm[:,2]*s_phi )
	Q_str[:,4] = ( c_phi + eigv_norm[:,1]*eigv_norm[:,1]*( 1 - c_phi ) )
	Q_str[:,5] = ( eigv_norm[:,1]*eigv_norm[:,2]*( 1 - c_phi ) - eigv_norm[:,0]*s_phi )
	Q_str[:,6] = ( eigv_norm[:,2]*eigv_norm[:,0]*( 1 - c_phi ) - eigv_norm[:,1]*s_phi )
	Q_str[:,7] = ( eigv_norm[:,2]*eigv_norm[:,1]*( 1 - c_phi ) + eigv_norm[:,0]*s_phi )
	Q_str[:,8] = ( c_phi + eigv_norm[:,2]*eigv_norm[:,2]*( 1 - c_phi ) )

	# Calculation of the local velocity gradient tensor: gradV
	gradV = math.matmul(gradv,Q_str)
	gradV = math.matmul(math.transpose(Q_str), gradV) 
	
	# Calculation of alpha and beta and condition 
	# CONDITION: If alpha^2 < beta^2 there is local fluid rotation in the XY plane
	alpha = 0.5*np.sqrt( ( gradV[:,4] - gradV[:,0] )*( gradV[:,4] - gradV[:,0] ) +
						 ( gradV[:,3] + gradV[:,1] )*( gradV[:,3] + gradV[:,1] ) )
	beta  = 0.5*( gradV[:,3] - gradV[:,1] )
	mask  = alpha*alpha - beta*beta < 0
	del aux, c, phi, c_phi, s_phi, Q_str

	# Calculation of rortex magnitude
	rtx_mag       = np.zeros((gradv.shape[0],), np.double)
	rtx_mag[mask] = 2.*( np.abs(beta[mask]) - alpha[mask] )
	
	# Setting the rortex vector: rortex magnitude towards the normalized real eigenvector direction
	rortex        = math.scaVecProd(rtx_mag,eigv_norm)
	return rortex

@cr('RortexCriterion')
def RortexCriterion(gradv, method='fast'):
	'''
	Rortex-Criterion for vortex identification from:
			Liu, C., Gao, Y., Tian, S., Dong, X., 2018. Rortex—A new vortex vector definition and vorticity tensor and vector decompositions. 
			Physics of Fluids 30, 035103. https://doi.org/10.1063/1.5023001

			Gao, Y., Liu, C., 2018. Rortex and comparison with eigenvalue-based vortex identification criteria. 
			Physics of Fluids 30, 085107. https://doi.org/10.1063/1.5040112

	IN:
			> gradv(nnod,9): gradient of velocity

	OUT:
			> Rortex-criterion(nnod,3): Rortex-criterion vector
	'''
	return _RortexCriterionFast(gradv) if method.lower() == 'fast' else _RortexCriterionShur(gradv)


@cr('OmegaRortexCriterion')
def OmegaRortexCriterion(gradv, epsilon=0.001, modified=False):
	'''
	OmegaRortex-Criterion for vortex identification from:
			Liu, C., Gao, Y., Dong, X., Wang, Y., Liu, J., Zhang, Y., Cai, X., Gui, N., 2019. 
			Third generation of vortex identification methods: Omega and Liutex/Rortex based systems. J Hydrodyn 31, 205–223. https://doi.org/10.1007/s42241-019-0022-4

	IN:
			> gradv(nnod,9): gradient of velocity

	OUT:
			> Rortex-criterion(nnod,3): Rortex-criterion vector
	'''
	rortex = np.zeros((gradv.shape[0], 3), np.double)
	# Compute the schur decomposition of the transposed gradient
	# and sorting the eigenvalues
	S, Q = math.schur(math.transpose(gradv))
	# det(Q) can only be 1 or -1
	gradV = math.transpose(S)
	detQ = math.det(Q)
	# Rewrite gradV
	gradV[detQ < 0., 2] *= -1.
	gradV[detQ < 0., 5] *= -1.
	# alpha and beta
	alpha = 0.5*np.sqrt((gradV[:, 4]-gradV[:, 0])*(gradV[:, 4]-gradV[:, 0]) +
						(gradV[:, 1]+gradV[:, 3])*(gradV[:, 1]+gradV[:, 3]))
	beta = 0.5*(gradV[:, 1]-gradV[:, 3])
	beta2 = beta*beta
	alpha2 = alpha*alpha
	# Compute maximum epsilon
	eps = epsilon*mpi_reduce(np.nanmax(beta2-alpha2), op='nanmax', all=True)
	# Return Omega
	out = beta2/(alpha2+beta2+eps) if not modified else (beta2 +
														 eps)/(alpha2+beta2+2.*eps)
	return out