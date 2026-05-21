"""
flick_urban — Fast and Light Inference for Climate Knowledge.

Urban wind modelling pipeline with three stages:
  - preprocess: STL geometry → HDF5 feature maps (MASK, HEGT, WDST)
  - nn:         UNet/Generator2D inference on feature maps → U, V wind fields
  - postprocess: tile-stitching, georeferencing, and visualisation

Example
-------
>>> from flick_urban.preprocess.geometry import calculate_bounding_box
>>> calculate_bounding_box('Examples/preprocess/data/grid_of_cubes.stl')
"""

__version__ = "0.1.0"

from . import preprocess, nn, postprocess
