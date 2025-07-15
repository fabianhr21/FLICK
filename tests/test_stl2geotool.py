import importlib.util
import runpy
import sys
import os
import types
import numpy as np
import h5py
import shutil
from stl import mesh
import argparse


def test_STL2GeoTool_gpu(tmp_path):
    # create minimal stl
    data = np.zeros(1, dtype=mesh.Mesh.dtype)
    data['vectors'][0] = np.array([[0,0,0],[1,0,0],[0,1,0]])
    m = mesh.Mesh(data)
    stl_path = tmp_path / 'cube.stl'
    m.save(str(stl_path))
    cwd = os.getcwd()
    os.chdir(tmp_path)

    # stub mpi4py
    mpi_mod = types.ModuleType('mpi4py')
    mpi_mod.rc = types.SimpleNamespace(recv_mprobe=False)
    class DummyComm:
        def Get_rank(self):
            return 0
        def Get_size(self):
            return 1
        def allgather(self, x):
            return [x]
        def bcast(self, x, root=0):
            return x
        def Barrier(self):
            pass
    mpi_mod.MPI = types.SimpleNamespace(COMM_WORLD=DummyComm())
    sys.modules['mpi4py'] = mpi_mod
    sys.modules['mpi4py.MPI'] = mpi_mod.MPI
    original_parse = argparse.ArgumentParser.parse_known_args
    def fake_parse(self, *args, **kwargs):
        parsed, extras = original_parse(self, *args, **kwargs)
        if isinstance(parsed.wind_direction, int):
            parsed.wind_direction = [parsed.wind_direction]
        return parsed, extras
    argparse.ArgumentParser.parse_known_args = fake_parse

    # stub pyQvarsi
    py_mod = types.ModuleType('pyQvarsi')
    class Field(dict):
        def __init__(self, xyz=None, ptable=None):
            super().__init__()
            self.xyz = xyz
            self.ptable = ptable
        def save(self, filename, mpio=False):
            with h5py.File(filename, 'w') as f:
                for k, v in self.items():
                    f.create_dataset(k, data=np.asarray(v))
    py_mod.Field = Field
    py_mod.PartitionTable = types.SimpleNamespace(new=lambda *a, **k: None)
    py_mod.MeshAlya = types.SimpleNamespace(plane=lambda *a, **k: None)
    py_mod.pprint = lambda *a, **k: None
    py_mod.cr_info = lambda: None
    py_mod.utils = types.SimpleNamespace(is_rank_or_serial=lambda x: True)
    sys.modules['pyQvarsi'] = py_mod

    # stub gmtry_utils
    gm_mod = types.ModuleType('gmtry_utils')
    def plane_generation(*a, **k):
        class DummyMesh:
            def __init__(self):
                self.xyz = np.zeros((1,3))
            def save(self, filename, mpio=False):
                with h5py.File(filename, 'w') as f:
                    f.create_dataset('dummy', data=[0])
        return DummyMesh()
    def geometrical_magnitudes(*a, **k):
        f = Field()
        f['MASK'] = [0]
        return f
    gm_mod.save_scalarfield = lambda *a, **k: None
    gm_mod.plane_generation = plane_generation
    gm_mod.geometrical_magnitudes = geometrical_magnitudes
    gm_mod.calculate_bounding_box = lambda *a, **k: (np.zeros(3), np.ones(3))
    gm_mod.append_UV_features = lambda *a, **k: None
    gm_mod.move_stl_to_origin = lambda *a, **k: None
    def rotate_geom(src, dst, angle):
        dest = dst + '.stl'
        if os.path.abspath(src) != os.path.abspath(dest):
            shutil.copy(src, dest)
    gm_mod.rotate_geometry = rotate_geom
    sys.modules['gmtry_utils'] = gm_mod

    calls = {}
    gpu_mod = types.ModuleType('gpu_gmtry_utils')
    def geometrical_magnitudes_gpu(*a, **k):
        calls['gpu'] = True
        f = Field()
        f['MASK'] = [1]
        return f
    gpu_mod.geometrical_magnitudes_gpu = geometrical_magnitudes_gpu
    gpu_mod.geometrical_data_extractor_gpu = lambda *a, **k: None
    sys.modules['gpu_gmtry_utils'] = gpu_mod

    script = os.path.join(cwd, 'pre-process', 'STL2GeoTool.py')
    sys.argv = [
        'STL2GeoTool.py',
        '-dataset_path', str(tmp_path) + '/',
        '-stl_basename', 'cube',
        '-output_path', str(tmp_path) + '/',
        '-step_size', '1',
        '-n_points', '1',
        '-p_overlap', '1',
        '-wind_direction', '0',
        '-use_gpu', 'True',
    ]
    runpy.run_path(script, run_name='__main__')
    argparse.ArgumentParser.parse_known_args = original_parse
    os.chdir(cwd)
    out_file = tmp_path / 'output0-cube' / 'cube-0-geodata.h5'
    assert out_file.exists()
    assert calls.get('gpu')
