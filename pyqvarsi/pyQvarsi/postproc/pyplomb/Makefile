# Compile PYPLOMB
#   Compile with the most aggressive optimization setting (O3)
#   Use the most pedantic compiler settings: must compile with no warnings at all
#
# The user may override any desired internal variable by redefining it via command-line:
#   make CXX=g++ [...]
#   make OPTL=-O2 [...]
#   make FLAGS="-Wall -g" [...]
#

# Options
#
DEBUGGING     = OFF
VECTORIZATION = ON
OPENMP_PARALL = OFF

# Compilers
#
# Automatically detect if the intel compilers are installed and use
# them, otherwise default to the GNU compilers
CC     = gcc
CXX    = g++
FC     = gfortran
PYTHON = python3
PIP    = pip3

# Compiler flags
#
OPTL = 3
ifeq ($(DEBUGGING),ON)
	CFLAGS   += -O0 -g -rdynamic -fPIC
	CXXFLAGS += -O0 -g -rdynamic -fPIC
	FFLAGS   += -O0 -g -rdynamic -fPIC
else
	CFLAGS   += -O$(OPTL) -fPIC
	CXXFLAGS += -O$(OPTL) -fPIC
	FFLAGS   += -O$(OPTL) -fPIC
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
# C standard
CFLAGS   += -std=c99
# C++ standard
CXXFLAGS += -std=c++11


# Defines
#
DFLAGS = -DNPY_NO_DEPRECATED_API


# One rule to compile them all, one rule to find them,
# One rule to bring them all and in the compiler link them.
all: requirements python install
	@echo ""
	@echo "plomb deployed successfully"


# Python
#
python: setup.py
	@CC="${CC}" CFLAGS="${CFLAGS} ${DFLAGS}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" LDSHARED="${CC} -shared" ${PYTHON} $< build_ext --inplace
	@echo "Python programs deployed successfully"

requirements: requirements.txt
	@${PIP} install -r $<

install:
	@CC="${CC}" CFLAGS="${CFLAGS} ${DFLAGS}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" LDSHARED="${CC} -shared" ${PIP} install --no-deps .

install_dev:
	@CC="${CC}" CFLAGS="${CFLAGS} ${DFLAGS}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" LDSHARED="${CC} -shared" ${PIP} install --no-deps -e .

wheel:
	@CC="${CC}" CFLAGS="${CFLAGS} ${DFLAGS}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" LDSHARED="${CC} -shared" ${PIP} wheel --no-deps -e .

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
	-@cd pyplomb; rm -f *.c *.html *.o $(wildcard **/*.o) 
	-@cd pyplomb; rm -f *.pyc $(wildcard **/*.pyc) $(wildcard **/*/*.pyc)
	-@cd pyplomb; rm -rf __pycache__

cleanall: clean
	-@rm -rf build
	-@cd pyplomb; rm *.so

uninstall: cleanall
	@${PIP} uninstall pyplomb
	-@rm -rf pyplomb.egg-info
