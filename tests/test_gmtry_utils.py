import importlib.util
import os
import sys
import types
import numpy as np
import pytest

@pytest.fixture()
def gm():
    mpi_module = types.ModuleType('mpi4py')
    mpi_module.rc = types.SimpleNamespace(recv_mprobe=False)
    class DummyComm:
        def Get_rank(self):
            return 0
        def Get_size(self):
            return 1
        def allgather(self, x):
            return [x]
    mpi_module.MPI = types.SimpleNamespace(COMM_WORLD=DummyComm())
    sys.modules['mpi4py'] = mpi_module
    sys.modules['mpi4py.MPI'] = mpi_module.MPI

    py_module = types.ModuleType('pyQvarsi')
    py_module.PartitionTable = types.SimpleNamespace(new=lambda *a, **k: None)
    py_module.MeshAlya = types.SimpleNamespace(plane=lambda *a, **k: None)
    class Field(dict):
        pass
    py_module.Field = Field
    sys.modules['pyQvarsi'] = py_module

    trimesh_module = types.ModuleType('trimesh')
    trimesh_module.load_mesh = lambda *a, **k: None
    sys.modules['trimesh'] = trimesh_module

    path = os.path.join(os.path.dirname(__file__), '..', 'pre-process', 'gmtry_utils.py')
    spec = importlib.util.spec_from_file_location('gmtry_utils', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_rotation_matrix_identity(gm):
    R = gm.rotation_matrix_around_z(270)
    assert np.allclose(R, np.eye(3))

def test_calculate_bounding_box(gm):
    stl_path = os.path.join(os.path.dirname(__file__), '..', 'grid_of_cubes.stl')
    min_c, max_c = gm.calculate_bounding_box(stl_path)
    assert np.allclose(min_c, [0.0, 0.0, 0.0])
    assert np.allclose(max_c, [390.0, 390.0, 30.0])
