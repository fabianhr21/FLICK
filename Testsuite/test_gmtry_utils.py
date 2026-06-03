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

    path = os.path.join(os.path.dirname(__file__), '..', 'flick_urban', 'nn', 'preprocess', 'geometry.py')
    spec = importlib.util.spec_from_file_location('gmtry_utils', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_rotation_matrix_identity(gm):
    R = gm.rotation_matrix_around_z(0)
    assert np.allclose(R, np.eye(3))

def test_rotation_matrix_90(gm):
    R = gm.rotation_matrix_around_z(90)
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(R, expected, atol=1e-10)

def test_calculate_bounding_box(gm):
    stl_path = os.path.join(os.path.dirname(__file__), '..', 'Examples', 'preprocess', 'data', 'grid_of_cubes.stl')
    min_c, max_c = gm.calculate_bounding_box(stl_path)
    assert np.allclose(min_c, [0.0, 0.0, 0.0])
    assert np.allclose(max_c, [390.0, 390.0, 30.0])


def test_move_stl_to_origin(gm, tmp_path):
    """move_stl_to_origin translates mesh so min bound becomes [0,0,0]."""
    from stl import mesh as stl_mesh

    # Build a 2-triangle mesh offset from origin
    offset = np.array([10.0, 20.0, 0.0])
    data = np.zeros(2, dtype=stl_mesh.Mesh.dtype)
    data['vectors'][0] = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]]) + offset
    data['vectors'][1] = np.array([[1, 1, 0], [2, 0, 0], [2, 1, 0]]) + offset
    in_path = str(tmp_path / 'in.stl')
    out_path = str(tmp_path / 'out.stl')
    stl_mesh.Mesh(data).save(in_path)

    gm.move_stl_to_origin(in_path, out_path)

    result = stl_mesh.Mesh.from_file(out_path)
    all_verts = np.concatenate([result.v0, result.v1, result.v2])
    min_coords = all_verts.min(axis=0)
    assert np.allclose(min_coords[:2], [0.0, 0.0], atol=1e-5), \
        f"Expected min [0,0,*], got {min_coords}"


def test_rotate_geometry_90deg(gm, tmp_path):
    """rotate_geometry 90° on a 2×1 rectangle → xspan≈1, yspan≈2."""
    from stl import mesh as stl_mesh

    # 2×1 rectangle at origin
    data = np.zeros(2, dtype=stl_mesh.Mesh.dtype)
    data['vectors'][0] = [[0, 0, 0], [2, 0, 0], [0, 1, 0]]
    data['vectors'][1] = [[2, 0, 0], [2, 1, 0], [0, 1, 0]]
    in_path = str(tmp_path / 'rect.stl')
    out_base = str(tmp_path / 'rect_rot')
    stl_mesh.Mesh(data).save(in_path)

    gm.rotate_geometry(in_path, out_base, 90)

    result = stl_mesh.Mesh.from_file(out_base + '.stl')
    all_verts = np.concatenate([result.v0, result.v1, result.v2])
    xspan = all_verts[:, 0].max() - all_verts[:, 0].min()
    yspan = all_verts[:, 1].max() - all_verts[:, 1].min()

    assert np.isclose(xspan, 1.0, atol=1e-3), f"Expected xspan≈1, got {xspan}"
    assert np.isclose(yspan, 2.0, atol=1e-3), f"Expected yspan≈2, got {yspan}"


def test_rotate_geometry_0deg_identity(gm, tmp_path):
    """rotate_geometry 0° on grid_of_cubes.stl → bounds unchanged."""
    stl_path = os.path.join(os.path.dirname(__file__), '..', 'Examples', 'preprocess', 'data', 'grid_of_cubes.stl')
    out_base = str(tmp_path / 'cubes_rot0')
    gm.rotate_geometry(stl_path, out_base, 0)

    orig_min, orig_max = gm.calculate_bounding_box(stl_path)
    rot_min, rot_max = gm.calculate_bounding_box(out_base + '.stl')
    assert np.allclose(orig_min, rot_min, atol=1e-2)
    assert np.allclose(orig_max, rot_max, atol=1e-2)
