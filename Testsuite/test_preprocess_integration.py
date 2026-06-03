"""
Integration tests: real STL geometry processing on grid_of_cubes.stl (CPU only).
mpi4py, pyQvarsi, and trimesh are mocked at import time — no GPU required.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import pytest
from stl import mesh as stl_mesh

STL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'Examples', 'preprocess', 'data', 'grid_of_cubes.stl'
)


@pytest.fixture(scope='module')
def geometry_module():
    """Load flick_urban/preprocess/geometry.py with mpi4py/pyQvarsi/trimesh mocked."""
    # mpi4py mock
    mpi_mod = types.ModuleType('mpi4py')
    mpi_mod.rc = types.SimpleNamespace(recv_mprobe=False)

    class _DummyComm:
        def Get_rank(self): return 0
        def Get_size(self): return 1
        def allgather(self, x): return [x]
        def bcast(self, x, root=0): return x
        def Barrier(self): pass

    mpi_mod.MPI = types.SimpleNamespace(COMM_WORLD=_DummyComm())
    sys.modules['mpi4py'] = mpi_mod
    sys.modules['mpi4py.MPI'] = mpi_mod.MPI

    # pyQvarsi mock
    pq_mod = types.ModuleType('pyQvarsi')
    pq_mod.PartitionTable = types.SimpleNamespace(new=lambda *a, **k: None)
    pq_mod.MeshAlya = types.SimpleNamespace(plane=lambda *a, **k: None)
    pq_mod.Field = dict
    sys.modules['pyQvarsi'] = pq_mod

    # trimesh mock (only move_stl_to_origin_trimesh uses it; not called here)
    tm_mod = types.ModuleType('trimesh')
    tm_mod.load_mesh = lambda *a, **k: None
    sys.modules['trimesh'] = tm_mod

    path = os.path.join(
        os.path.dirname(__file__), '..', 'flick_urban', 'nn', 'preprocess', 'geometry.py'
    )
    spec = importlib.util.spec_from_file_location('geometry_integration', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stl_file_exists():
    """grid_of_cubes.stl is present in the repo."""
    assert os.path.isfile(STL_PATH), f"Test STL not found: {STL_PATH}"


def test_bounding_box_grid_of_cubes(geometry_module):
    """grid_of_cubes.stl has known bounds [0,0,0] → [390,390,30]."""
    min_c, max_c = geometry_module.calculate_bounding_box(STL_PATH)
    assert np.allclose(min_c, [0.0, 0.0, 0.0], atol=1e-3)
    assert np.allclose(max_c, [390.0, 390.0, 30.0], atol=1e-3)


def test_move_to_origin_is_idempotent(geometry_module, tmp_path):
    """grid_of_cubes.stl is already at origin — moving keeps bounds unchanged."""
    out = str(tmp_path / 'cubes_moved.stl')
    geometry_module.move_stl_to_origin(STL_PATH, out)
    min_c, max_c = geometry_module.calculate_bounding_box(out)
    assert np.allclose(min_c, [0.0, 0.0, 0.0], atol=1e-3)
    assert np.allclose(max_c, [390.0, 390.0, 30.0], atol=1e-3)


def test_rotate_0deg_preserves_bounds(geometry_module, tmp_path):
    """Rotating grid_of_cubes.stl by 0° does not change the bounding box."""
    out = str(tmp_path / 'cubes_rot0')
    geometry_module.rotate_geometry(STL_PATH, out, 0.0)
    min_c, max_c = geometry_module.calculate_bounding_box(out + '.stl')
    orig_min, orig_max = geometry_module.calculate_bounding_box(STL_PATH)
    assert np.allclose(min_c, orig_min, atol=1e-2)
    assert np.allclose(max_c, orig_max, atol=1e-2)


def test_rotate_90deg_square_same_spans(geometry_module, tmp_path):
    """grid_of_cubes.stl is 390×390 (square); rotating 90° keeps XY spans equal."""
    out = str(tmp_path / 'cubes_rot90')
    geometry_module.rotate_geometry(STL_PATH, out, 90.0)
    min_c, max_c = geometry_module.calculate_bounding_box(out + '.stl')
    orig_min, orig_max = geometry_module.calculate_bounding_box(STL_PATH)

    orig_xspan = orig_max[0] - orig_min[0]
    orig_yspan = orig_max[1] - orig_min[1]
    rot_xspan = max_c[0] - min_c[0]
    rot_yspan = max_c[1] - min_c[1]

    # 390×390 square → same spans after 90° rotation
    assert np.isclose(rot_xspan, orig_xspan, atol=1.0), \
        f"xspan: {rot_xspan:.2f} vs {orig_xspan:.2f}"
    assert np.isclose(rot_yspan, orig_yspan, atol=1.0), \
        f"yspan: {rot_yspan:.2f} vs {orig_yspan:.2f}"


def test_rotate_45deg_bounding_box_expands(geometry_module, tmp_path):
    """Rotating a 2×1 rectangle 45° expands its XY bounding box."""
    # Build a 2×1 rectangle
    data = np.zeros(2, dtype=stl_mesh.Mesh.dtype)
    data['vectors'][0] = [[0, 0, 0], [2, 0, 0], [0, 1, 0]]
    data['vectors'][1] = [[2, 0, 0], [2, 1, 0], [0, 1, 0]]
    rect_path = str(tmp_path / 'rect.stl')
    stl_mesh.Mesh(data).save(rect_path)

    out = str(tmp_path / 'rect_rot45')
    geometry_module.rotate_geometry(rect_path, out, 45.0)
    rot_min, rot_max = geometry_module.calculate_bounding_box(out + '.stl')
    rot_xspan = rot_max[0] - rot_min[0]

    # 45° rotation of a 2×1 rect → xspan > 1.0 (original smaller dimension)
    assert rot_xspan > 1.0, f"Expected xspan > 1.0 after 45° rotation, got {rot_xspan:.3f}"
