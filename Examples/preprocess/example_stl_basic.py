"""Basic pre-processing example using grid_of_cubes.stl."""
import os
import sys

# Allow running directly: python Examples/preprocess/example_stl_basic.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

STL_PATH = os.path.join(os.path.dirname(__file__), "data/grid_of_cubes.stl")

if not os.path.exists(STL_PATH):
    raise FileNotFoundError(f"Test geometry not found: {STL_PATH}")

print(f"STL geometry: {STL_PATH}")
print("Run STL2GeoTool with desired arguments to generate H5 output.")
print("Example (with MPI):")
print(f"  mpirun -n 4 python -m flick_urban.nn.preprocess.stl2geo {STL_PATH} -o output/")
