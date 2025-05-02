#!/usr/bin/env python
#
# pyQvarsi, GEOM Module.
#
# Geometric entities for mesh and field selection.
#
# Last rev: 12/01/2021

__VERSION__ = 3.1

from .basic    import Point, Ball, Polygon
# 1D entities
from .entities import Line 
# 2D entities
from .entities import SimpleRectangle, Rectangle, Plane, Polygon2D
# 3D entities
from .entities import SimpleCube, Cube, Polygon3D
# Other entities
from .entities import Collection, DiscreteBox

del basic, entities