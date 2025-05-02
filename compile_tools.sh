#!/bin/bash

# Compile city4CFD
sudo apt-get install libmpfr-dev libgmp-dev libboost-all-dev libeigen3-dev libomp-dev libgdal-dev
cd City4CFD
mkdir build && cd build
cmake .. -DCGAL_DIR=../cgal-6.0.1
make -j 4
cd ../../pre-process/geo4CFD/
ln -sf ../../City4CFD/build/city4cfd .
cd ../../

cd ../..
# Compile pyqvarsi
cd pyqvarsi
make
