"""
flick_urban.nn.preprocess — STL geometry preprocessing.

Converts STL city models into HDF5 feature maps used for NN inference.
Key modules:
  - geometry:          rotation, bounding box, wall distance, plane generation
  - stl2geo:           main MPI+GPU preprocessing script
  - gpu_geometry_opt:  optimised GPU geometry extractor (production)
  - features:          appends U/V fields to existing HDF5 files
  - simpli_stl:        mesh decimation via pymeshlab
  - stl_from_bim:      IFC/BIM → STL conversion via ifcopenshell
  - utils:             CFD field averaging utilities (requires pyQvarsi/pyAlya)
"""
