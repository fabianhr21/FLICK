import importlib.util
import sys
import types
import os
import numpy as np
import h5py
import torch


def test_inference_script_runs(tmp_path):
    sys.modules['STL2GeoTool_loop'] = types.SimpleNamespace(WIND_DIRECTION=[0], p_overlap=0.5)
    sys.path.append('wind-nn')
    spec = importlib.util.spec_from_file_location('inference', os.path.join('wind-nn', 'inference-script.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    N = module.INPUT_XDIM
    file = tmp_path / 'sample-0-geodata.h5'
    with h5py.File(file, 'w') as f:
        grp = f.create_group('FIELD/VARIABLES')
        for name in ['MASK', 'HEGT', 'WDST', 'U', 'V', 'GRDUX', 'GRDVY', 'GRDWZ']:
            grp.create_dataset(name, data=np.zeros((N, N)))

    args = module.get_args()
    args.dataset_path = str(tmp_path) + '/'
    args.data_sample_basename = 'sample'
    args.verbose = 0

    model = module.UNet_wind(args)
    x, y = module.load_input_sample(args, 0)
    out = model(x.float())
    assert out.shape[2:] == (args.target_xdim, args.target_ydim)
