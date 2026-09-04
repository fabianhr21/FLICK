#!/bin/bash
# compile_tools.sh — Build City4CFD (required) with bundled CGAL 6.0.1, and pyqvarsi.
#
# City4CFD is required for BIM/OSM-to-CFD geometry preparation.
# CGAL 6.0.1 is bundled as a git submodule at City4CFD/cgal-6.0.1 — no separate
# CGAL installation needed.
#
# Usage (from repo root):
#   bash scripts/compile_tools.sh
#
# Requirements: CMake >= 3.15, GCC >= 9, Ubuntu 20.04+

set -e

# Initialize all submodules (City4CFD, City4CFD/cgal-6.0.1, pyqvarsi)
echo "Initializing submodules..."
git submodule update --init --recursive

# Install system dependencies (Ubuntu/Debian)
echo "Installing system libraries..."
sudo apt-get update -qq
sudo apt-get install -y \
    libmpfr-dev \
    libgmp-dev \
    libboost-all-dev \
    libeigen3-dev \
    libomp-dev \
    libgdal-dev

# Set FLICK root (resolved from script location, not CWD)
export FLIC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Compile City4CFD against bundled CGAL 6.0.1
echo "Building City4CFD..."
cd "$FLIC/City4CFD"
mkdir -p build && cd build
cmake .. -DCGAL_DIR="$FLIC/City4CFD/cgal-6.0.1"
make -j2
echo "City4CFD built successfully."

# Create symlink so geo4cfd can find the binary
cd "$FLIC/flick_urban/geo4cfd/"
ln -sf "../../City4CFD/build/city4cfd" city4cfd
echo "Symlink created at flick_urban/geo4cfd/city4cfd"

# Compile pyqvarsi
# cd "$FLIC/pyqvarsi"
# make
