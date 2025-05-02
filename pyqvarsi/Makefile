# Compile pyQvarsi
#   Compile with g++ or Intel C++ Compiler
#   Compile with the most aggressive optimization setting (O3)
#   Use the most pedantic compiler settings: must compile with no warnings at all
#
# The user may override any desired internal variable by redefining it via command-line:
#   make CXX=g++ [...]
#   make OPTL=-O2 [...]
#   make FLAGS="-Wall -g" [...]
#
# Arnau Miro 2021

# Include user-defined build configuration file if exists, otherwise the default file.
ifneq ("$(wildcard options.cfg)","")
    include options.cfg
     $(info Using options.cfg)
else
    include config/options_default.cfg
    $(info Using config/options_default.cfg)
endif

# Compilers
#
# Automatically detect if the intel compilers are installed and use
# them, otherwise default to the GNU compilers
ifeq ($(OVERRIDE_COMPILERS),OFF)
	ifeq ($(FORCE_GCC),ON) 
		# Forcing the use of GCC
		# C Compiler
		CC = mpicc
		# C++ Compiler
		CXX = mpicxx
		# Fortran Compiler
		FC = mpifort
	else
		ifeq (,$(shell which icc))
			# C Compiler
			CC = mpicc
			# C++ Compiler
			CXX = mpicxx
			# Fortran Compiler
			FC = mpifort
		else
			# C Compiler
			CC = mpiicc
			# C++ Compiler
			CXX = mpiicpc
			# Fortran Compiler
			FC = mpiifort
		endif
	endif
endif


# Compiler flags
#
ifeq ($(OVERRIDE_COMPILERS),OFF)
	ifeq ($(CC),mpicc)
		# Using GCC as a compiler
		ifeq ($(DEBUGGING),ON)
			# Debugging flags
			CFLAGS   += -O0 -g -rdynamic -fPIC
			CXXFLAGS += -O0 -g -rdynamic -fPIC
			FFLAGS   += -O0 -g -rdynamic -fPIC
		else
			CFLAGS   += -O$(OPTL) -ffast-math -fPIC
			CXXFLAGS += -O$(OPTL) -ffast-math -fPIC
			FFLAGS   += -O$(OPTL) -ffast-math -fPIC
		endif
		# Vectorization flags
		ifeq ($(VECTORIZATION),ON)
			CFLAGS   += -march=native -ftree-vectorize
			CXXFLAGS += -march=native -ftree-vectorize
			FFLAGS   += -march=native -ftree-vectorize
		endif
		# OpenMP flag
		ifeq ($(OPENMP_PARALL),ON)
			CFLAGS   += -fopenmp
			CXXFLAGS += -fopenmp
		endif
	else
		# Using INTEL as a compiler
		ifeq ($(DEBUGGING),ON)
			# Debugging flags
			CFLAGS   += -O0 -g -traceback -fPIC
			CXXFLAGS += -O0 -g -traceback -fPIC
			FFLAGS   += -O0 -g -traceback -fPIC
		else
			CFLAGS   += -O$(OPTL) -fPIC
			CXXFLAGS += -O$(OPTL) -fPIC
			FFLAGS   += -O$(OPTL) -fPIC
		endif
		# Vectorization flags
		ifeq ($(VECTORIZATION),ON)
			CFLAGS   += -x$(HOST) -mtune=$(TUNE)
			CXXFLAGS += -x$(HOST) -mtune=$(TUNE)
			FFLAGS   += -x$(HOST) -mtune=$(TUNE)
		endif
		# OpenMP flag
		ifeq ($(OPENMP_PARALL),ON)
			CFLAGS   += -qopenmp
			CXXFLAGS += -qopenmp
		endif
	endif
endif
# C standard
CFLAGS   += -std=c99
# C++ standard
CXXFLAGS += -std=c++11


# Defines
#
DFLAGS = -DNPY_NO_DEPRECATED_API


# One rule to compile them all, one rule to find them,
# One rule to bring them all and in the compiler link them.
all: requirements pyplomb python install
	@echo ""
	@echo "pyQvarsi deployed successfully"


# Python
#
python: setup.py pyplomb
	@CC="${CC}" CFLAGS="${CFLAGS} ${DFLAGS}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" LDSHARED="${CC} -shared" ${PYTHON} $< build_ext --inplace
	@echo "Python programs deployed successfully"

pyplomb: pyQvarsi/postproc/pyplomb
	@git submodule update --init --recursive
	@cd $<; CC="${CC}" CFLAGS="${CFLAGS} ${DFLAGS}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" LDSHARED="${CC} -shared" ${PYTHON} setup.py build_ext --inplace
	@echo "PyPlomb compiled successfully"

requirements: requirements.txt
	@${PIP} install -r $<


install: requirements
	@CC="${CC}" CFLAGS="${CFLAGS} ${DFLAGS}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" LDSHARED="${CC} -shared" ${PIP} install --no-deps --use-pep517 .

install_dev: requirements
	@CC="${CC}" CFLAGS="${CFLAGS} ${DFLAGS}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" LDSHARED="${CC} -shared" ${PIP} install --no-deps --use-pep517 -e .

wheel: requirements
	@CC="${CC}" CFLAGS="${CFLAGS} ${DFLAGS}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" LDSHARED="${CC} -shared" ${PIP} wheel --no-deps --use-pep517 -e .

package-build:
	@CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" LDSHARED="${CC} -shared" ${PYTHON} -m build


# Generic object makers
#
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $< $(DFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(DFLAGS)

%.o: %.f
	$(FC) $(FFLAGS) -c -o $@ $< $(DFLAGS)

%.o: %.f90
	$(FC) $(FFLAGS) -c -o $@ $< $(DFLAGS)


# Clean
#
clean:
	-@cd pyQvarsi; rm -f *.o $(wildcard **/*.o) $(wildcard **/*/*.o)
	-@cd pyQvarsi; rm -f *.pyc $(wildcard **/*.pyc) $(wildcard **/*/*.pyc)
	-@cd pyQvarsi; rm -rf __pycache__
	-@cd pyQvarsi; rm -rf statistics/__pycache__  
	-@cd pyQvarsi; rm -rf periodic/__pycache__  
	-@cd pyQvarsi; rm -rf FEM/__pycache__ FEM/*.c FEM/*.cpp FEM/*.html
	-@cd pyQvarsi; rm -rf vmath/__pycache__ vmath/*.c vmath/*.cpp vmath/*.html
	-@cd pyQvarsi; rm -rf inp_out/__pycache__ inp_out/*.c inp_out/*.cpp inp_out/*.html
	-@cd pyQvarsi; rm -rf Geom/__pycache__ Geom/*.c Geom/*.cpp Geom/*.html
	-@cd pyQvarsi; rm -rf utils/__pycache__ utils/*.c utils/*.cpp utils/*.html
	-@cd pyQvarsi; rm -rf meshing/__pycache__ meshing/*.c meshing/*.cpp meshing/*.html
	-@cd pyQvarsi; rm -rf postproc/__pycache__ postproc/*.c postproc/*.cpp postproc/*.html
	-@cd pyQvarsi; rm -rf solvers/__pycache__ solvers/*.c postproc/*.cpp solvers/*.html
	-@cd pyQvarsi; rm -rf periodic/__pycache__ periodic/*.c periodic/*.cpp periodic/*.html
	-@cd pyQvarsi/postproc/pyplomb; make clean

cleanall: clean
	-@rm -rf build
	-@cd pyQvarsi; rm FEM/*.so
	-@cd pyQvarsi; rm vmath/*.so
	-@cd pyQvarsi; rm Geom/*.so
	-@cd pyQvarsi; rm inp_out/*.so
	-@cd pyQvarsi; rm postproc/*.so
	-@cd pyQvarsi; rm meshing/*.so
	-@cd pyQvarsi; rm solvers/*.so
	-@cd pyQvarsi; rm periodic/*.so
	-@cd pyQvarsi/postproc/pyplomb; make cleanall

uninstall: cleanall
	@${PIP} uninstall pyQvarsi
	-@rm -rf pyQvarsi.egg-info
