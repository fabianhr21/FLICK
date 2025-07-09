#!/bin/bash

# Compile city4CFD
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    libmpfr-dev \
    libgmp-dev \
    libboost-all-dev \
    libeigen3-dev \
    libomp-dev \
    libgdal-dev

# Set FLIC to current directory
export FLIC=$(pwd)

# Compile City4CFD
cd City4CFD
mkdir -p build && cd build
cmake .. -DCGAL_DIR=$FLIC/City4CFD/cgal-6.0.1
make -j4

# Create symlink to city4cfd binary
cd ../../pre-process/geo4CFD/
ln -sf ../../City4CFD/build/city4cfd .
cd ../../

# Compile pyqvarsi
#cd pyqvarsi
#make
