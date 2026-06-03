import numpy as np
from flick_urban.nn.postprocess.overlap import overlap_matrix


def test_overlap_matrix():
    matrices = [np.full((2, 2), i) for i in range(1, 5)]
    result = overlap_matrix(
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
