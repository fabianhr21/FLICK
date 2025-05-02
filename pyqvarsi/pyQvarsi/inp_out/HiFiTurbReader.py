#!/usr/bin/env python
#
# HiFiTurb Reader
#
# Last rev: 01/03/2021
from __future__ import print_function, division

import numpy as np, h5py


class HiFiTurbDB_Reader(object):
	'''
	Reader class for HiFiTurb dataset.
	'''
	def __init__(self,filename,return_matrix=True,parallel=False):
		'''
		Class constructor
		'''
		self._fname    = filename
		self._isopen   = False
		self._outtype  = return_matrix
		self._parallel = parallel
		self._istart   = -1
		self._iend     = -1
		# Obtain the MPI comm World if parallel
		if parallel:
			from mpi4py import MPI
			self._comm = MPI.COMM_WORLD
		# Open file for reading and keep it opened
		self.open()

#	def __del__(self):
#		'''
#		Class destructor
#		'''
#		self.close() # close the file

	def __len__(self):
		return self.info['n_nodes']

	def __str__(self):
		info = self.info
		retstr  = 'HiFiTurb Database on <%s>' % self._fname
		retstr += ' (opened)\n' if self._isopen else ' (closed)\n'
		if self._isopen:
			retstr += '-> Model %s\n' % (info['model'])
			retstr += '-> Dimensions %d nodes, %d elements\n' % (info['n_nodes'],info['n_elems'])
			retstr += '-> Density   %f\n' % self.density
			retstr += '-> Viscosity %f\n' % self.viscosity
		return retstr

	def open(self):
		'''
		Open the file if it is not opened
		'''
		if not self._isopen:
			self._file   = h5py.File(self._fname,'r') if not self._parallel else h5py.File(self._fname,'r',driver='mpio',comm=self._comm)
			self._isopen = True

			if self._parallel:
				self._istart, self._iend = worksplit(0,self.info['n_nodes'])
			else:
				self._istart, self._iend = 0, self.info['n_nodes']

	def close(self):
		'''
		Close the file if it is opened
		'''
		if self._isopen:
			self._file.close()
			self._isopen = False
			self._istart = -1
			self._iend   = -1

	def set_return(self,value):
		self._outtype = value

	# Read the diferent parameters from the HiFiTurb database as
	# class properties. New parameters can be added as properties.
	# The arrays are returned as numpy arrays and the data read will
	# be deleted due to memory issues.

	# First read the information part and return it as a dictionary
	@property
	def info(self):
		'''
		Information (metadata) regarding the dataset
		'''
		out  = {}
		data = self._file['01_Info']
		# Read the data and store it in out
		for var in data.keys():
			out[var] = data[var][0] # assume here everything of size (1,)
		return out
	@property
	def density(self):
		return float(self._file['01_Info']['density'][0])
	@property
	def viscosity(self):
		return float(self._file['01_Info']['viscosity'][0])	

	# Now read the rest of the parameters
	@property
	def points(self):
		'''
		Grid points or nodes
		'''
		return np.array(self._file['03_Nodes']['Nodes'][self._istart:self._iend],dtype=np.double)

	# Inputs
	@property
	def pressure(self):
		'''
		Pressure
		'''
		inp  = np.array(self._file['02_Entries']['Inputs'][self._istart:self._iend])
		data = inp[:,0].copy()
		del inp
		return data.astype(np.double)
	@property
	def velocity(self):
		'''
		Velocity [u,v,w]
		'''
		inp  = np.array(self._file['02_Entries']['Inputs'][self._istart:self._iend,:])
		data = inp[:,[1,2,3]].copy()
		del inp
		return data.astype(np.double)
	@property
	def shear_stress(self):
		'''
		Shear stress
		either [t_11,t_12,t_22,t_13,t_23,t_33]
		or     [t_11,t_12,t_13;t_12,t_22,t_23;t_13,t_23,t_33]
		'''
		inp  = np.array(self._file['02_Entries']['Inputs'][self._istart:self._iend,:])
		idx  = [4,5,6,7,8,9] if not self._outtype else [4,5,7,5,6,8,7,8,9]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def reynolds_stress(self):
		'''
		Reynolds stress
		either [r_11,r_12,r_22,r_13,r_23,r_33]
		or     [r_11,r_12,r_13;r_12,r_22,r_23;r_13,r_23,r_33]
		'''
		inp  = np.array(self._file['02_Entries']['Inputs'][self._istart:self._iend,:])
		idx  = [10,11,12,13,14,15] if not self._outtype else [10,11,13,11,12,14,13,14,15]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def pressure_gradient(self):
		'''
		Pressure gradient [px,py,pz]
		'''
		inp  = np.array(self._file['02_Entries']['Inputs'][self._istart:self._iend,:])
		data = inp[:,[16,20,24]].copy()
		del inp
		return data.astype(np.double)
	@property
	def velocity_gradient(self):
		'''
		Velocity gradient
		either [ux,vx,wx;uy,vy,wy;uz,vz,wz]
		or     [ux,uy,uz;vx,vy,vz;wx,wy,wz]
		'''
		inp  = np.array(self._file['02_Entries']['Inputs'][self._istart:self._iend,:])
		idx  = [17,18,19,20,21,22,23,24,25] if not self._outtype else [17,21,25,18,22,26,19,23,27]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def wall_distance(self):
		'''
		Wall distance
		'''
		inp  = np.array(self._file['02_Entries']['Inputs'][self._istart:self._iend,:])
		data = inp[:,26].copy()
		del inp
		return data.astype(np.double)
	@property
	def yplus(self):
		'''
		y+
		'''
		inp  = np.array(self._file['02_Entries']['Inputs'][self._istart:self._iend,:])
		data = inp[:,27].copy()
		del inp
		return data.astype(np.double)

	# Reynolds stress budgets
	@property
	def production(self):
		'''
		Production budget
		either [p_11,p_12,p_22,p_13,p_23,p_33]
		or     [p_11,p_12,p_13;p_12,p_22,p_23;p_13,p_23,p_33]
		'''
		inp  = np.array(self._file['02_Entries']['01_Output']['Production'][self._istart:self._iend,:])
		idx  = [0,1,2,3,4,5] if not self._outtype else [0,1,3,1,2,4,3,4,5]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def convection(self):
		'''
		Convection budget
		either [c_11,c_12,c_22,c_13,c_23,c_33]
		or     [c_11,c_12,c_13;c_12,c_22,c_23;c_13,c_23,c_33]
		'''
		inp  = np.array(self._file['02_Entries']['01_Output']['Convection'][self._istart:self._iend,:])
		idx  = [0,1,2,3,4,5] if not self._outtype else [0,1,3,1,2,4,3,4,5]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def dissipation(self):
		'''
		Dissipation budget
		either [d_11,d_12,d_22,d_13,d_23,d_33]
		or     [d_11,d_12,d_13;d_12,d_22,d_23;d_13,d_23,d_33]
		'''
		inp  = np.array(self._file['02_Entries']['01_Output']['Dissipation'][self._istart:self._iend,:])
		idx  = [0,1,2,3,4,5] if not self._outtype else [0,1,3,1,2,4,3,4,5]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def pressure_strain(self):
		'''
		Pressure strain budget
		either [ps_11,ps_12,ps_22,ps_13,ps_23,ps_33]
		or     [ps_11,ps_12,ps_13;ps_12,ps_22,ps_23;ps_13,ps_23,ps_33]
		'''
		inp  = np.array(self._file['02_Entries']['01_Output']['PressureStrain'][self._istart:self._iend,:])
		idx  = [0,1,2,3,4,5] if not self._outtype else [0,1,3,1,2,4,3,4,5]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def molecular_diffusion(self):
		'''
		Molecular diffusion budget
		either [md_11,md_12,md_22,md_13,md_23,md_33]
		or     [md_11,md_12,md_13;md_12,md_22,md_23;md_13,md_23,md_33]
		'''
		inp  = np.array(self._file['02_Entries']['01_Output']['MolecularDiffusion'][self._istart:self._iend,:])
		idx  = [0,1,2,3,4,5] if not self._outtype else [0,1,3,1,2,4,3,4,5]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def turbulent_diffusion01(self):
		'''
		Turbulent diffusion 01 budget
		either [td01_11,td01_12,td01_22,td01_13,td01_23,td01_33]
		or     [td01_11,td01_12,td01_13;td01_12,td01_22,td01_23;td01_13,td01_23,td01_33]
		'''
		inp  = np.array(self._file['02_Entries']['01_Output']['TurbulentDiffusion01'][self._istart:self._iend,:])
		idx  = [0,1,2,3,4,5] if not self._outtype else [0,1,3,1,2,4,3,4,5]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def turbulent_diffusion02(self):
		'''
		Turbulent diffusion 02 budget
		either [td02_11,td02_12,td02_22,td02_13,td02_23,td02_33]
		or     [td02_11,td02_12,td02_13;td02_12,td02_22,td02_23;td02_13,td02_23,td02_33]
		'''
		inp  = np.array(self._file['02_Entries']['01_Output']['TurbulentDiffusion02'][self._istart:self._iend,:])
		idx  = [0,1,2,3,4,5] if not self._outtype else [0,1,3,1,2,4,3,4,5]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)

	# Other quantities
	@property
	def pressure_velocity(self):
		'''
		Pressure velocity [pu,pv,pw]
		'''
		return np.array(self._file['02_Entries']['01_Output']['PressureVelocity'][self._istart:self._iend,:],dtype=np.double)
	@property
	def pressure_autocorrelation(self):
		'''
		Pressure autocorrelation
		'''
		inp  = np.array(self._file['02_Entries']['01_Output']['AdditionalQuantities'][self._istart:self._iend,:])
		data = inp[:,0].copy()
		del inp
		return data.astype(np.double)
	@property
	def taylor_microscale(self):
		'''
		Taylor microscale
		'''
		inp  = np.array(self._file['02_Entries']['01_Output']['AdditionalQuantities'][self._istart:self._iend,:])
		data = inp[:,1].copy()
		del inp
		return data.astype(np.double)	
	@property
	def kolmogorov_lengthscale(self):
		'''
		Kolmogorov lengthscale
		'''
		inp  = np.array(self._file['02_Entries']['01_Output']['AdditionalQuantities'][self._istart:self._iend,:])
		data = inp[:,2].copy()
		del inp
		return data.astype(np.double)
	@property
	def kolmogorov_timescale(self):
		'''
		Kolmogorov timescale
		'''
		inp  = np.array(self._file['02_Entries']['01_Output']['AdditionalQuantities'][self._istart:self._iend,:])
		data = inp[:,3].copy()
		del inp
		return data.astype(np.double)
	@property
	def velocity_triple_correlation(self):
		'''
		Velocity triple correlation
		'''
		return np.array(self._file['02_Entries']['01_Output']['TripleCorrelation'][self._istart:self._iend,:],dtype=np.double)
	# EARSM quantities
	@property
	def PsK(self):
		'''
		PsK
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		data = inp[:,0].copy()
		del inp
		return data.astype(np.double)
	@property
	def anisotropy(self):
		'''
		Anisotropy tensor
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		idx  = [1,2,3,4,5,6] if not self._outtype else [1,2,4,2,3,5,4,5,6]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def sigma(self):
		'''
		Sigma
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		data = inp[:,7].copy()
		del inp
		return data.astype(np.double)	
	@property
	def r(self):
		'''
		r
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		data = inp[:,8].copy()
		del inp
		return data.astype(np.double)
	@property
	def III_S(self):
		'''
		III_S
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		data = inp[:,9].copy()
		del inp
		return data.astype(np.double)
	@property
	def IV(self):
		'''
		IV
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		data = inp[:,10].copy()
		del inp
		return data.astype(np.double)
	@property
	def V(self):
		'''
		V
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		data = inp[:,11].copy()
		del inp
		return data.astype(np.double)
	@property
	def II_a(self):
		'''
		II_a
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		data = inp[:,12].copy()
		del inp
		return data.astype(np.double)
	@property
	def III_a(self):
		'''
		II_a
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		data = inp[:,13].copy()
		del inp
		return data.astype(np.double)
	@property
	def T1(self):
		'''
		T1 tensor
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		idx  = [13,14,15,16,17,18] if not self._outtype else [13,14,16,14,15,17,16,17,18]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def T2(self):
		'''
		T2 tensor
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		idx  = [19,20,21,22,23,24] if not self._outtype else [19,20,22,20,21,23,22,23,24]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def T3(self):
		'''
		T3 tensor
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		idx  = [25,26,27,28,29,30] if not self._outtype else [19,20,22,20,21,23,22,23,24]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def T4(self):
		'''
		T4 tensor
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		idx  = [31,32,33,34,35,36] if not self._outtype else [31,32,34,32,33,35,34,35,36]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def T5(self):
		'''
		T5 tensor
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		idx  = [37,38,39,40,41,42] if not self._outtype else [37,38,40,38,39,41,40,41,42]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def beta(self):
		'''
		beta1 to beta5
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM'][self._istart:self._iend,:])
		data = inp[:,43:].copy()
		del inp
		return data.astype(np.double)

	# EARSM extra quantities
	@property
	def velocity_laplacian(self):
		'''
		Laplacian of velocity
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM_Extra'][self._istart:self._iend,:])
		data = inp[:,:3].copy()
		del inp
		return data.astype(np.double)
	@property
	def Sij_material_derivative(self):
		'''
		Material derivative of Sij/s
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM_Extra'][self._istart:self._iend,:])
		idx  = [3,4,5,6,7,8] if not self._outtype else [3,4,6,4,5,7,6,7,8]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def rotational_vorticity_tensor(self):
		'''
		Oadr
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM_Extra'][self._istart:self._iend,:])
		idx  = [9,10,11,12,13,14] if not self._outtype else [9,10,12,10,11,13,12,13,14]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def II_Or(self):
		'''
		II_Or
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM_Extra'][self._istart:self._iend,:])
		data = inp[:,15].copy()
		del inp
		return data.astype(np.double)
	@property
	def T2r(self):
		'''
		T2r
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM_Extra'][self._istart:self._iend,:])
		idx  = [15,16,17,18,19,20] if not self._outtype else [15,16,18,16,17,19,18,19,20]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def sigma_vk(self):
		'''
		sigma_vk
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM_Extra'][self._istart:self._iend,:])
		data = inp[:,21].copy()
		del inp
		return data.astype(np.double)
	@property
	def g_plus(self):
		'''
		g_plus
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM_Extra'][self._istart:self._iend,:])
		data = inp[:,22].copy()
		del inp
		return data.astype(np.double)
	@property
	def g_y(self):
		'''
		g_y
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM_Extra'][self._istart:self._iend,:])
		data = inp[:,23].copy()
		del inp
		return data.astype(np.double)
	@property
	def g_t(self):
		'''
		g_t
		'''
		inp  = np.array(self._file['02_Entries']['02_EARSM']['EARSM_Extra'][self._istart:self._iend,:])
		data = inp[:,24].copy()
		del inp
		return data.astype(np.double)

	# K-Omega residual
	@property
	def Rk(self):
		'''
		Rk
		'''
		inp  = np.array(self._file['02_Entries']['04_KO_RESIDUAL'][self._istart:self._iend,:])
		data = inp[:,0].copy()
		del inp
		return data.astype(np.double)
	@property
	def Romega(self):
		'''
		Romega
		'''
		inp  = np.array(self._file['02_Entries']['04_KO_RESIDUAL'][self._istart:self._iend,:])
		data = inp[:,1].copy()
		del inp
		return data.astype(np.double)
	@property
	def dK(self):
		'''
		dK
		'''
		inp  = np.array(self._file['02_Entries']['04_KO_RESIDUAL'][self._istart:self._iend,:])
		data = inp[:,2].copy()
		del inp
		return data.astype(np.double)
	@property
	def domega(self):
		'''
		domega
		'''
		inp  = np.array(self._file['02_Entries']['04_KO_RESIDUAL'][self._istart:self._iend,:])
		data = inp[:,3].copy()
		del inp
		return data.astype(np.double)
	@property
	def dKO(self):
		'''
		dKO
		'''
		inp  = np.array(self._file['02_Entries']['04_KO_RESIDUAL'][self._istart:self._iend,:])
		data = inp[:,4].copy()
		del inp
		return data.astype(np.double)

	# DRSM variables
	@property
	def A1(self):
		'''
		A1 tensor
		'''
		inp  = np.array(self._file['02_Entries']['03_DRSM']['DRSM'][self._istart:self._iend,:])
		idx  = [8,9,10,11,12,13] if not self._outtype else [8,9,11,9,10,12,11,12,13]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def A2(self):
		'''
		A2 tensor
		'''
		inp  = np.array(self._file['02_Entries']['03_DRSM']['DRSM'][self._istart:self._iend,:])
		idx  = [14,15,16,17,18,19] if not self._outtype else [14,15,17,15,16,18,17,18,19]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def A3(self):
		'''
		A3 tensor
		'''
		inp  = np.array(self._file['02_Entries']['03_DRSM']['DRSM'][self._istart:self._iend,:])
		idx  = [20,21,22,23,24,25] if not self._outtype else [20,21,23,21,22,24,23,24,25]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def A4(self):
		'''
		A4 tensor
		'''
		inp  = np.array(self._file['02_Entries']['03_DRSM']['DRSM'][self._istart:self._iend,:])
		idx  = [26,27,28,29,30,31] if not self._outtype else [26,27,29,27,28,30,28,29,30]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def A5(self):
		'''
		A5 tensor
		'''
		inp  = np.array(self._file['02_Entries']['03_DRSM']['DRSM'][self._istart:self._iend,:])
		idx  = [32,33,34,35,36,37] if not self._outtype else [32,33,35,33,34,36,35,36,37]
		data = inp[:,idx].copy()
		del inp
		return data.astype(np.double)
	@property
	def c_prime(self):
		'''
		Cp1 to Cp5
		'''
		inp  = np.array(self._file['02_Entries']['03_DRSM']['DRSM'][self._istart:self._iend,:])
		data = inp[:,38:].copy()
		del inp
		return data.astype(np.double)


def worksplit(istart,iend):
	'''
	Divide the number of points between the processors
	as equally as possible. Return the range of start
	and end.
	'''
	from mpi4py import MPI
	mpi_rank = MPI.COMM_WORLD.Get_rank()
	mpi_size = MPI.COMM_WORLD.Get_size()

	istart_l, iend_l = 0, iend
	irange = iend - istart
	if (mpi_size < irange):
		# We split normally among processes assuming no remainder
		rangePerProcess = int(np.floor(irange/mpi_size))
		istart_l        = istart   + mpi_rank*rangePerProcess
		iend_l          = istart_l + rangePerProcess
		# Handle the remainder
		remainder = irange - rangePerProcess*mpi_size
		if remainder > mpi_rank:
			istart_l += mpi_rank
			iend_l   += mpi_rank+1;
		else:
			istart_l += remainder
			iend_l   += remainder
	else:
		# Each process will forcefully conduct one instant.
		istart_l = mpi_rank   if mpi_rank < iend else iend
		iend_l   = mpi_rank+1 if mpi_rank < iend else iend

	return istart_l, iend_l	