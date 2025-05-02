#!/usr/bin/env python
#
# pyQvarsi, setup.
#
# Setup and cythonize code.
#
# Last rev: 17/02/2021
from __future__ import print_function, division

import os, sys, numpy as np, mpi4py
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

with open('README.md') as f:
	readme = f.read()


## Read compilation options
options = {}
options_file = 'options.cfg' if os.path.isfile('options.cfg') else 'config/options_default.cfg'
with open(options_file) as f:
	for line in f.readlines():
		if '#' in line or len(line) == 1: continue # Skip comment
		linep = line.split('=')
		options[linep[0].strip()] = linep[1].strip()
		if options[linep[0].strip()] == 'ON':  options[linep[0].strip()] = True
		if options[linep[0].strip()] == 'OFF': options[linep[0].strip()] = False
options['MODULES_COMPILED'] = options['MODULES_COMPILED'].lower().split(',')


## Set up compiler options and flags
if not options['OVERRIDE_COMPILERS']:
	CC  = 'mpicc'   if options['FORCE_GCC'] or not os.system('which icc > /dev/null') == 0 else 'mpiicc'
	CXX = 'mpicxx'  if options['FORCE_GCC'] or not os.system('which icc > /dev/null') == 0 else 'mpiicpc'
	FC  = 'mpifort' if options['FORCE_GCC'] or not os.system('which icc > /dev/null') == 0 else 'mpiifort'
else:
	CC  = options['CC']
	CXX = options['CXX']
	FC  = options['FC']

CFLAGS   = ''
CXXFLAGS = ' -std=c++11'
FFLAGS   = ''
DFLAGS   = ' -DNPY_NO_DEPRECATED_API'
if not options['OVERRIDE_COMPILERS']:
	if CC == 'mpicc':
		# Using GCC as a compiler
		CFLAGS   += ' -O0 -g -rdynamic -fPIC' if options['DEBUGGING'] else ' -O%s -ffast-math -fPIC' % options['OPTL']
		CXXFLAGS += ' -O0 -g -rdynamic -fPIC' if options['DEBUGGING'] else ' -O%s -ffast-math -fPIC' % options['OPTL']
		FFLAGS   += ' -O0 -g -rdynamic -fPIC' if options['DEBUGGING'] else ' -O%s -ffast-math -fPIC' % options['OPTL']
		# Vectorization flags
		if options['VECTORIZATION']:
			CFLAGS   += ' -march=native -ftree-vectorize'
			CXXFLAGS += ' -march=native -ftree-vectorize'
			FFLAGS   += ' -march=native -ftree-vectorize'
		# OpenMP flag
		if options['OPENMP_PARALL']:
			CFLAGS   += ' -fopenmp'
			CXXFLAGS += ' -fopenmp'
	else:
		# Using GCC as a compiler
		CFLAGS   += ' -O0 -g -traceback -fPIC' if options['DEBUGGING'] else ' -O%s -fPIC' % options['OPTL']
		CXXFLAGS += ' -O0 -g -traceback -fPIC' if options['DEBUGGING'] else ' -O%s -fPIC' % options['OPTL']
		FFLAGS   += ' -O0 -g -traceback -fPIC' if options['DEBUGGING'] else ' -O%s -fPIC' % options['OPTL']
		# Vectorization flags
		if options['VECTORIZATION']:
			CFLAGS   += ' -x%s -mtune=%s' % (options['HOST'],options['TUNE'])
			CXXFLAGS += ' -x%s -mtune=%s' % (options['HOST'],options['TUNE'])
			FFLAGS   += ' -x%s -mtune=%s' % (options['HOST'],options['TUNE'])
		# OpenMP flag
		if options['OPENMP_PARALL']:
			CFLAGS   += ' -qopenmp'
			CXXFLAGS += ' -qopenmp'
else:
	CFLAGS   += options['CFLAGS']
	CXXFLAGS += options['CXXFLAGS']
	FFLAGS   += options['FFLAGS']


## Set up environment variables
os.environ['CC']       = CC
os.environ['CXX']      = CXX
os.environ['CFLAGS']   = CFLAGS + DFLAGS
os.environ['CXXFLAGS'] = CXXFLAGS + DFLAGS
os.environ['LDSHARED'] = CC + ' -shared'


## Libraries and includes
libraries     = ['m']

# OSX needs to also link with python3.8 for reasons...
if sys.platform == 'darwin': libraries += [f'python{sys.version_info[0]}.{sys.version_info[1]}']


## Modules
# Define each one of the external modules that the tool is comprised of
Module_FEM_lib  = Extension('pyQvarsi.FEM.lib',
							sources      = ['pyQvarsi/FEM/lib.pyx'],
					   		language     = 'c',
					   		include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_FEM_div  = Extension('pyQvarsi.FEM.div',
							sources      = ['pyQvarsi/FEM/div.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_FEM_dgp  = Extension('pyQvarsi.FEM.divgp',
							sources      = ['pyQvarsi/FEM/divgp.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_FEM_grad = Extension('pyQvarsi.FEM.grad',
							sources      = ['pyQvarsi/FEM/grad.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_FEM_grgp = Extension('pyQvarsi.FEM.gradgp',
							sources      = ['pyQvarsi/FEM/gradgp.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_FEM_int  = Extension('pyQvarsi.FEM.integral',
							sources      = ['pyQvarsi/FEM/integral.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_FEM_mass = Extension('pyQvarsi.FEM.mass',
							sources      = ['pyQvarsi/FEM/mass.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_FEM_utils = Extension('pyQvarsi.FEM.utils',
							sources      = ['pyQvarsi/FEM/utils.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_FEM_quadratures = Extension('pyQvarsi.FEM.quadratures',
									sources = ['pyQvarsi/FEM/quadratures.pyx'],
									language = 'c',
									include_dirs = [np.get_include()],
									libraries = libraries,
								)
Module_Math_utils = Extension('pyQvarsi.vmath.utils',
							sources      = ['pyQvarsi/vmath/utils.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_Math_vect  = Extension('pyQvarsi.vmath.vector',
							sources      = ['pyQvarsi/vmath/vector.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_Math_tens  = Extension('pyQvarsi.vmath.tensor',
							sources      = ['pyQvarsi/vmath/tensor.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_Geom_basic = Extension('pyQvarsi.Geom.basic',
							sources      = ['pyQvarsi/Geom/basic.pyx','pyQvarsi/Geom/src/geometry.cpp'],
							language     = 'c++',
							include_dirs = ['pyQvarsi/Geom/src',np.get_include()],
					   		libraries    = libraries,
						   )
Module_meshing_mesh = Extension('pyQvarsi.meshing.mesh',
							sources      = ['pyQvarsi/meshing/mesh.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_meshing_centers = Extension('pyQvarsi.meshing.cellcenters',
							sources      = ['pyQvarsi/meshing/cellcenters.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_postproc_yplus = Extension('pyQvarsi.postproc.yplus',
							sources      = ['pyQvarsi/postproc/yplus.pyx'],
							language     = 'c',
							include_dirs = [np.get_include(),mpi4py.get_include()],
					   		libraries    = libraries,
						   )
Module_IO_ensight  = Extension('pyQvarsi.inp_out.EnsightIO',
							sources      = ['pyQvarsi/inp_out/EnsightIO.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_solvers_lumped = Extension('pyQvarsi.solvers.lumped',
							sources      = ['pyQvarsi/solvers/lumped.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_solvers_appInv = Extension('pyQvarsi.solvers.approxInverse',
							sources      = ['pyQvarsi/solvers/approxInverse.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_solvers_conjgrad= Extension('pyQvarsi.solvers.conjgrad',
							sources      = ['pyQvarsi/solvers/conjgrad.pyx'],
							language     = 'c',
							include_dirs = [np.get_include()],
					   		libraries    = libraries,
						   )
Module_periodic_periodic= Extension('pyQvarsi.periodic.periodic',
							sources      = ['pyQvarsi/periodic/periodic.pyx','pyQvarsi/periodic/src/D3class.cpp'],
							language     = 'c++',
							include_dirs = ['pyQvarsi/periodic/src',np.get_include()],
					   		libraries    = libraries,
						   )

# Build modules
Module_FEM  = [Module_FEM_lib, Module_FEM_grad, Module_FEM_grgp, Module_FEM_div, Module_FEM_dgp, Module_FEM_int, Module_FEM_mass, Module_FEM_utils, Module_FEM_quadratures] if 'fem' in options['MODULES_COMPILED'] else []
Module_Math = [Module_Math_utils, Module_Math_vect, Module_Math_tens] if 'math' in options['MODULES_COMPILED'] else []
Module_Geom = [Module_Geom_basic] if 'geometry' in options['MODULES_COMPILED'] else []
Module_IO   = [Module_IO_ensight] if 'io' in options['MODULES_COMPILED'] else []
Module_Mesh = [Module_meshing_mesh, Module_meshing_centers] if 'mesh' in options['MODULES_COMPILED'] else []
Module_Post = [Module_postproc_yplus] if 'postproc' in options['MODULES_COMPILED'] else []
Module_Solv = [Module_solvers_lumped, Module_solvers_appInv, Module_solvers_conjgrad] if 'solvers' in options['MODULES_COMPILED'] else []
Module_Peri = [Module_periodic_periodic] if 'periodic' in options['MODULES_COMPILED'] else []


## Decide which modules to compile
modules_list = Module_FEM  + Module_Math + Module_Geom + Module_IO + Module_Mesh + Module_Post + Module_Solv + Module_Peri if options['USE_COMPILED'] else []


## Scripts to install
scripts_list = [os.path.join('bin',f) for f in os.listdir('bin')]


## Main setup
setup(
	name="pyQvarsi",
	version="3.1.0",
	ext_modules=cythonize(modules_list,
		language_level = str(sys.version_info[0]), # This is to specify python 3 synthax
		annotate       = True # This is to generate a report on the conversion to C code
	),
    long_description=readme,
    url='https://gitlab.com/ArnauMiro/pyQvarsi.git',
    packages=find_packages(exclude=('examples','Examples','doc','docs')),
	install_requires=['numpy>=1.21.6','scipy>=1.5.4','matplotlib','cython>=3.0.0','mpi4py>=4.0.0','nfft'],
	scripts=scripts_list
)
