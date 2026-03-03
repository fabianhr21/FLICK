#!/usr/bin/env python
"""
Standalone inference script for the FLICK webapp.
Runs UNet_wind on all -geodata.h5 files in a preprocessed directory
and writes <basename>-<idx>-UMAG.csv to the output directory.

Usage:
    python run_inference.py \
        --preprocess_dir /path/to/output45.0-input/ \
        --basename input \
        --model_path /path/to/weights/ \
        --model_basename model \
        --output_dir /path/to/infer_output/ \
        --n_points 256
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch

INPUT_FEAT = ['MASK', 'HEGT', 'WDST']
SCALING_X = 120.0


def get_args():
    p = argparse.ArgumentParser(description='FLICK webapp inference')
    p.add_argument('--preprocess_dir', required=True,
                   help='Folder containing *-geodata.h5 files from STL2GeoTool')
    p.add_argument('--basename', required=True,
                   help='STL basename used during preprocessing (e.g. "input")')
    p.add_argument('--model_path', required=True,
                   help='Directory containing the weights file')
    p.add_argument('--model_basename', default='model',
                   help='Weights file stem (without .pt extension)')
    p.add_argument('--output_dir', required=True,
                   help='Where to write UMAG CSV files')
    p.add_argument('--n_points', type=int, default=256,
                   help='Grid dimension (must match trained model, default 256)')
    return p.parse_args()


def load_h5_sample(file_path: str, n_points: int) -> torch.Tensor:
    """Load MASK/HEGT/WDST from an HDF5 geodata file into a (1, 3, N, N) tensor."""
    with h5py.File(file_path, 'r') as f:
        X = np.empty((len(INPUT_FEAT), n_points, n_points), dtype=np.float64)
        for i, key in enumerate(INPUT_FEAT):
            scale = 1.0 / SCALING_X if key in ('WDST', 'HEGT') else 1.0
            arr = scale * np.array(f[f'/FIELD/VARIABLES/{key}'], dtype=np.float64)
            arr = np.nan_to_num(arr, nan=0.0)
            arr = np.reshape(arr, (n_points, n_points), order='C')
            arr = np.flip(arr, axis=0).copy()
            X[i] = arr
    return torch.unsqueeze(torch.from_numpy(X), 0)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Add wind-nn/ to sys.path so UNet_wind can be imported
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, os.path.join(repo_root, 'wind-nn'))
    from Unet_model import UNet_wind  # noqa: E402 (local import after path setup)

    class ModelArgs:  # minimal namespace expected by UNet_wind.__init__
        x_features = INPUT_FEAT
        y_features = ['U', 'V']
        e_features = []
        input_xdim = args.n_points
        input_ydim = args.n_points
        target_xdim = args.n_points
        target_ydim = args.n_points
        scaling_x = SCALING_X
        scaling_y = 16.0
        verbose = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[inference] device={device}')

    model_file = os.path.join(args.model_path, f'{args.model_basename}.pt')
    if not os.path.exists(model_file):
        print(f'[inference] ERROR: weights not found at {model_file}')
        sys.exit(1)

    model = UNet_wind(ModelArgs())
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.float().eval().to(device)
    print(f'[inference] model loaded from {model_file}')

    pattern = os.path.join(args.preprocess_dir, f'{args.basename}-*-geodata.h5')
    h5_files = sorted(glob.glob(pattern))
    if not h5_files:
        print(f'[inference] ERROR: no H5 files matching: {pattern}')
        sys.exit(1)

    for h5_path in h5_files:
        fname = os.path.basename(h5_path)
        # Strip "<basename>-" prefix and "-geodata.h5" suffix to get the tile index
        idx_str = fname[len(args.basename) + 1:].replace('-geodata.h5', '')

        x = load_h5_sample(h5_path, args.n_points).to(device)
        with torch.no_grad():
            ypred = model(x.float())

        mask = (x[0][0] >= 1.0).cpu().numpy()
        Ux = ypred[0][0].cpu().numpy() * mask
        Uy = ypred[0][1].cpu().numpy() * mask
        umag = np.sqrt(Ux ** 2 + Uy ** 2)

        prefix = f'{args.basename}-{idx_str}'
        np.savetxt(
            os.path.join(args.output_dir, f'{prefix}-UMAG.csv'),
            umag,
            delimiter=',',
        )
        print(f'[inference] saved {prefix}-UMAG.csv')


if __name__ == '__main__':
    main()
