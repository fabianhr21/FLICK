#!/usr/bin/env python
#
# PLOMB: LOMB-SCARGLE PERIODOGRAM
#
from __future__ import print_function, division

import sys, numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

with open('README.md') as f:
	readme = f.read()


## Libraries and includes
include_dirs  = []
extra_objects = []
libraries     = ['m']

# OSX needs to also link with python for reasons...
if sys.platform == 'darwin': libraries += ['python%d.%d'%(sys.version_info[0],sys.version_info[1])]


## Modules
Module_plomb = Extension('pyplomb.cy_plomb',
						sources       = ['pyplomb/cy_plomb.pyx','pyplomb/src/dplomb.c','pyplomb/src/splomb.c','pyplomb/src/utils.c'],
						language      = 'c',
						include_dirs  = include_dirs + ['pyplomb/src',np.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
)


## Decide which modules to compile
modules_list = [ Module_plomb ]


## Main setup
setup(
	name             = 'pyplomb',
	version          = '1.5.0',
	author           = 'Benet Eiximeno, Arnau Miro',
	author_email     = 'benet.eiximeno@bsc.es, arnau.mirojane@bsc.es',
	maintainer       = 'Arnau Miro',
	maintainer_email = 'arnau.mirojane@bsc.es',
	ext_modules = cythonize(modules_list,
		language_level = str(sys.version_info[0]), # This is to specify python 3 synthax
		annotate       = True                      # This is to generate a report on the conversion to C code
	),
	long_description = readme,
	url              = 'https://gitlab.com/ArnauMiro/pyplomb',
	packages         = find_packages(exclude=('Examples')),
	install_requires = ['numpy','cython>=3.0.0']
)