#!/usr/bin/env python
#
# pyQvarsi, FEM Module.
#
# Small FEM module to compute derivatives and possible other
# simple stuff from Alya output for postprocessing purposes.
#
# Last rev: 30/09/2020

__VERSION__ = 3.1

# Import basic element class
from .lib import Element1D, Element2D, Element3D, createElementByType, defineHighOrderElement

# Import element types from the library (1D elements)
from .lib import Bar

# Import element types from the library (2D elements)
from .lib import LinearTriangle, LinearQuadrangle

# Import element types from the library (3D elements)
from .lib import LinearTetrahedron, LinearPyramid, LinearPrism, TrilinearBrick, TriQuadraticBrick

# Import FEM operations
from .grad        import gradient2D, gradient3D
from .gradgp      import gradient2Dgp, gradient3Dgp
from .div         import divergence2D, divergence3D
from .divgp       import divergence2Dgp, divergence3Dgp
from .mass        import mass_matrix_lumped, mass_matrix_consistent
from .integral    import integralSurface, integralVolume
from .utils       import cellCenters, nodes2Gauss, gauss2Nodes, nodes_per_element, connectivity, quad2tri
from .quadratures import legendre, dlegendre, d2legendre, d3legendre, quadrature_GaussLobatto

del lib, grad, div, mass, integral, utils
