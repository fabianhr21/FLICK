#!/usr/bin/env python
#
# pyQvarsi, statistics.
#
# Accumulator routines.
#
# Last rev: 16/02/2021
from __future__ import print_function, division

import numpy as np

from ..             import vmath as math
from ..cr           import cr
from ..utils.common import raiseError


@cr('stats.addS1')
def addS1(outfield,infield,w):
	'''
	Adds the first order statistics implementing
	the Welford online algorithm.
	
	EXAMPLE USAGE
		avvel += addS1(avvel,veloc,w=dt/time)

	IN:
		> outfield(nnod,): output field, also returned by this function
		> infield(nnod,):  input field to be averaged
		> w:               weight (dt/time or 1/instant for a normal average)

	OUT:
		> output field(nnod,)
	'''
	return math.linopArrf(w,infield,-w,outfield) if len(infield.shape) > 1 else math.linopScaf(w,infield,-w,outfield)


@cr('stats.addS2')
def addS2(outfield,infield1,infield2,w):
	'''
	Adds the second order statistics implementing
	the Welford online algorithm.
	
	EXAMPLE USAGE
		nd = 0: avpr2 += addS2(avpr2,press,press,w=dt/time)
		nd = 3: avpve += addS2(avpve,press,veloc,w=dt/time)
		nd = 9: avve2 += addS2(avve2,veloc,veloc,w=dt/time)

	IN:
		> outfield(nnod,nd):  output field, also returned by this function
		> infield1(nnod,nd1): input field to be averaged
		> infield2(nnod,nd2): input field to be averaged
		> w:                  weight (dt/time or 1/instant for a normal average)

	nd can be either 0, 3 or 9. If:
		- nd == 0 then nd1 = 0 and nd2 = 0 (i.e., pp)
		- nd == 3 then nd1 = 0 and nd2 = 3 (i.e., pu,pv,pw)
		- nd == 9 then nd1 = 3 and nd2 = 3 (i.e., uu,uv,vv,uw,vw,ww)

	OUT:
		> output field(nnod,nd)
	'''
	if len(outfield.shape) == 1:
		return w*(infield1*infield2 - outfield)
	elif outfield.shape[1] == 3:
		aux = np.zeros(outfield.shape,dtype=outfield.dtype)
		aux[:,0] = w*(infield1*infield2[:,0] - outfield[:,0])
		aux[:,1] = w*(infield1*infield2[:,1] - outfield[:,1])
		aux[:,2] = w*(infield1*infield2[:,2] - outfield[:,2])
		return aux
	elif outfield.shape[1] == 9:
		aux = np.zeros(outfield.shape,dtype=outfield.dtype)
		aux[:,0] = w*(infield1[:,0]*infield2[:,0] - outfield[:,0]) # uu
		aux[:,1] = w*(infield1[:,0]*infield2[:,1] - outfield[:,1]) # uv
		aux[:,2] = w*(infield1[:,0]*infield2[:,2] - outfield[:,2]) # uw
		aux[:,3] = w*(infield1[:,1]*infield2[:,0] - outfield[:,3]) # vu
		aux[:,4] = w*(infield1[:,1]*infield2[:,1] - outfield[:,4]) # vv
		aux[:,5] = w*(infield1[:,1]*infield2[:,2] - outfield[:,5]) # vw
		aux[:,6] = w*(infield1[:,2]*infield2[:,0] - outfield[:,6]) # wu
		aux[:,7] = w*(infield1[:,2]*infield2[:,1] - outfield[:,7]) # wv
		aux[:,8] = w*(infield1[:,2]*infield2[:,2] - outfield[:,8]) # ww
		return aux
	raiseError('addS2 statistic not recognized!')


@cr('stats.addS3')
def addS3(outfield,infield1,infield2,infield3,w):
	'''
	Adds the triple correlation implementing
	the Welford online algorithm.

	EXAMPLE USAGE
		avve3 += addS2(avve3,veloc,veloc,veloc,w=dt/time)

	IN:
		> outfield(nnod,nd):  output field, also returned by this function
		> infield1(nnod,nd1): input field to be averaged
		> infield2(nnod,nd1): input field to be averaged
		> infield3(nnod,nd1): input field to be averaged
		> w:                  weight (dt/time or 1/instant for a normal average)

	OUT:
		> output field(nnod,nd)
	'''
	if not outfield.shape[1] == 27 or not infield1.shape[1] == 3 or not infield2.shape[1] == 3 or not infield3.shape[1] == 3:
		raiseError('addS3 dimensions do not agree!')
	aux = np.zeros(outfield.shape,dtype=outfield.dtype)
	for i1 in range(infield1.shape[1]):
		for i2 in range(infield2.shape[1]):
			for i3 in range(infield3.shape[1]):
				ind = i1+3*i2+9*id3
				aux[:,ind] = w*(infield1[:,i1]*infield2[:,i2]*infield3[:,i3] - outfield[:,ind])
	return aux