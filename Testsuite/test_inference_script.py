"""
Tests for flick_urban.nn — Generator2D forward pass and inference utilities.
No model weights required; tests architecture and data-loading logic only.
"""
import types
import numpy as np
import pytest


def _make_args(num_res_blocks=2):
    """Minimal args namespace matching Generator2D and inference API."""
    return types.SimpleNamespace(
        x_features=['MASK', 'HEGT', 'WDST'],
        y_features=['U', 'V'],
        e_features=[],
        num_res_blocks=num_res_blocks,
        input_xdim=256,
        input_ydim=256,
        target_xdim=256,
        target_ydim=256,
        scaling_x=120.0,
        scaling_y=16.0,
    )


def test_generator2d_forward_pass():
    """Generator2D produces (1, 2, 256, 256) output from (1, 3, 256, 256) input."""
    import torch
    from flick_urban.nn.models import Generator2D

    model = Generator2D(_make_args())
    model.eval()

    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x.float())

    assert out.shape == (1, 2, 256, 256), f"Unexpected shape: {out.shape}"
    # tanh output must be in [-1, 1]
    assert out.min().item() >= -1.0
    assert out.max().item() <= 1.0


def test_generator2d_output_dtype():
    """Generator2D output is float32."""
    import torch
    from flick_urban.nn.models import Generator2D

    model = Generator2D(_make_args())
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x.float())
    assert out.dtype == torch.float32


def test_get_args_defaults():
    """get_args([]) returns expected defaults without consuming sys.argv."""
    from flick_urban.nn.inference import get_args
    args = get_args([])
    assert args.input_xdim == 256
    assert args.input_ydim == 256
    assert args.scaling_x == 120.0
    assert args.scaling_y == 16.0
    assert 'MASK' in args.x_features


def test_load_input_sample(tmp_path):
    """load_input_sample returns correctly shaped tensors from an H5 file."""
    import h5py
    import torch
    from flick_urban.nn.inference import get_args, load_input_sample

    args = get_args([])
    N = args.input_xdim

    h5_path = tmp_path / 'sample-0-geodata.h5'
    with h5py.File(h5_path, 'w') as f:
        grp = f.create_group('FIELD/VARIABLES')
        for name in ['MASK', 'HEGT', 'WDST', 'U', 'V', 'GRDUX', 'GRDVY', 'GRDWZ']:
            grp.create_dataset(name, data=np.zeros((N, N)))

    x, output = load_input_sample(args, str(h5_path))

    assert x.shape == (1, len(args.x_features), N, N)
    assert output['y'].shape == (1, len(args.y_features), N, N)
    assert isinstance(x, torch.Tensor)
