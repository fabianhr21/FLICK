"""
features.py — Append U/V velocity fields to existing HDF5 geodata files.

Runs after stl2geo.py to add placeholder or CFD-derived U/V features
to the HDF5 files produced by the preprocessing pipeline.

Usage
-----
    python -m flick_urban.preprocess.features \\
        -stl_basename grid_of_cubes -output_path ./output/ -wind_direction 0
"""
from flick_urban.preprocess.geometry import append_UV_features
import os
import argparse

STL_BASENAME = 'grid_of_cubes'


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Append U/V features to HDF5 geodata files')
    parser.add_argument('-stl_basename', default=STL_BASENAME, help='STL basename (used to locate output files)')
    parser.add_argument('-wind_direction', type=float, default=0.0, help='wind direction angle in degrees')
    parser.add_argument('-output_path', default='./output/', help='root output folder from stl2geo')
    args, _ = parser.parse_known_args(argv)
    return args


if __name__ == '__main__':
    args = get_args()
    STL_BASENAME = args.stl_basename
    print(f"STL_BASENAME: {STL_BASENAME}")

    OUTPUT_DIR = args.output_path + f'output{int(args.wind_direction) if args.wind_direction == int(args.wind_direction) else args.wind_direction}-{STL_BASENAME}/'
    files = os.listdir(OUTPUT_DIR)
    h5_files = [f for f in files if f.endswith('.h5')]

    for idx in range(len(h5_files)):
        path = f"{OUTPUT_DIR}{STL_BASENAME}-{idx}"
        print(f"Adding U/V features to {path}-geodata.h5")
        append_UV_features(path)
        print(f"Done: {path}-geodata.h5")
