#!/usr/bin/env python
#
# pyQvarsi.
#
# Python interface for postprocessing in Alya.
#
# Last rev: 21/04/2021

__VERSION__ = 3.1

## Import other modules
from . import FEM, inp_out as io, vmath as math, Geom, statistics as stats, postproc, utils, periodic, solvers, meshing, plotting


## Import pyQvarsi modules
from .mesh            import Mesh, MeshAlya, MeshSOD2D
from .field           import Field, FieldAlya, FieldSOD2D, fieldFastReduce
from .partition_table import PartitionTable
from .communicator    import Communicator
from .checkpoint      import Checkpoint
from .utils.common    import pprint, raiseError, raiseWarning, truncate, printArray, run_subprocess
from .cr              import cr, cr_start, cr_stop, cr_time, cr_reset, cr_info
from .mem             import mem, mem_start, mem_stop, mem_value, mem_reset, mem_info

del mesh, field, partition_table, communicator, checkpoint