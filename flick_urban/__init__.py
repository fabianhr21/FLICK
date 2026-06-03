"""
flick_urban — Fast and Light Inference for Climate Knowledge.

Urban wind modelling pipeline:
  - nn.preprocess:  STL geometry → HDF5 feature maps (MASK, HEGT, WDST)
  - nn:             UNet/Generator2D inference on feature maps → U, V wind fields
  - nn.postprocess: tile-stitching, georeferencing, and visualisation
  - geo4cfd:        LiDAR/BIM → CFD-ready mesh via City4CFD + ANSA

Example
-------
>>> from flick_urban.nn.preprocess.geometry import calculate_bounding_box
>>> calculate_bounding_box('Examples/preprocess/data/grid_of_cubes.stl')
"""

__version__ = "0.1.0"

from . import nn, geo4cfd
