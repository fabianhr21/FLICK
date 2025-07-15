import importlib.util
import sys
import types
import os
import numpy as np


def test_overlap_matrix():
    sys.modules['STL2GeoTool_loop'] = types.SimpleNamespace(WIND_DIRECTION=[0], p_overlap=0.5)
    spec = importlib.util.spec_from_file_location('overlap', os.path.join('post-process', 'overlap.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    matrices = [np.full((2, 2), i) for i in range(1, 5)]
    result = module.overlap_matrix(
        matrices,
        N_points=2,
        step=1,
        overlap=1,
        y_dir=4,
        x_frames=2,
        x_factor=1.5,
        y_factor=1.5,
    )
    expected = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 3.0, 4.0, 0.0],
        [3.0, 3.0, 4.0, 0.0],
        [1.0, 1.0, 2.0, 0.0],
    ])
    assert np.allclose(result, expected)
