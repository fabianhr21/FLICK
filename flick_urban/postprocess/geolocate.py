"""
geolocate.py — Georeference wind field predictions to real-world coordinates.

Reads Ground Control Point (GCP) metadata from the preprocessing step and
maps pixel-space U/V fields to UTM/geographic coordinates.

Usage
-----
    python -m flick_urban.postprocess.geolocate \\
        -basename grid_of_cubes -dataset_path ./final_output/ \\
        -gcp_path ./output/ -wind_direction 0
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

basename = 'grid_of_cubes'
GCP_path = './output/'
DATASET_PATH = './final_output/'
OUTPUT_PATH = './output/'


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Georeference wind field predictions')
    parser.add_argument('-basename',       default=basename,      help='STL basename used in preprocessing')
    parser.add_argument('-dataset_path',   default=DATASET_PATH,  help='path to final_output directory')
    parser.add_argument('-gcp_path',       default=GCP_path,      help='path to directory containing GCP.csv')
    parser.add_argument('-output_path',    default=OUTPUT_PATH,   help='output directory for georeferenced results')
    parser.add_argument('-wind_direction', type=float, default=0.0, help='wind direction angle in degrees')
    args, _ = parser.parse_known_args(argv)
    return args


def vel_magNdir(U, V, W=0):
    """Compute velocity magnitude and 2-D direction from U, V components."""
    VMAG = np.sqrt(U**2 + V**2 + W**2)
    VDIR2D = np.arctan2(V, U)
    return VMAG, VDIR2D


def GEOLOCATE(U_path, V_path, GCP_PATH):
    """Map pixel-space wind fields to real-world coordinates using GCP metadata.

    Parameters
    ----------
    U_path   : str — path to U.csv
    V_path   : str — path to V.csv
    GCP_PATH : str — path to GCP.csv produced by stl2geo.py

    Returns
    -------
    wind_field_df : pd.DataFrame with columns pixel_x, pixel_y, real_x, real_y,
                    wind_value, U, V, dir
    rows, cols    : int — spatial dimensions of the field
    """
    GCP = pd.read_csv(GCP_PATH)
    U = np.loadtxt(U_path, delimiter=',')
    V = np.loadtxt(V_path, delimiter=',')

    min_x = GCP['min_x'].values[0]
    max_x = GCP['max_x'].values[0]
    min_y = GCP['min_y'].values[0]
    max_y = GCP['max_y'].values[0]

    VMAG, VDIR = vel_magNdir(U, V)
    rows, cols = VMAG.shape

    x_coords = np.linspace(min_x, max_x, cols)
    y_coords = np.linspace(min_y, max_y, rows)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    wind_field_df = pd.DataFrame({
        'pixel_x':    np.tile(np.arange(cols), rows),
        'pixel_y':    np.repeat(np.arange(rows), cols),
        'real_x':     x_grid.ravel(),
        'real_y':     y_grid.ravel(),
        'wind_value': VMAG.ravel(),
        'U':          U.ravel(),
        'V':          V.ravel(),
        'dir':        VDIR.ravel(),
    })

    return wind_field_df, rows, cols


if __name__ == '__main__':
    args = get_args()
    wind_angle = args.wind_direction
    angle_str  = str(int(wind_angle)) if wind_angle == int(wind_angle) else str(wind_angle)

    dataset_dir = args.dataset_path + f'output{angle_str}-{args.basename}/'
    gcp_file    = args.gcp_path     + f'output{angle_str}-{args.basename}/GCP.csv'
    out_dir     = args.output_path  + f'output{angle_str}-{args.basename}/'

    wind_field_df, rows, cols = GEOLOCATE(
        dataset_dir + 'U.csv',
        dataset_dir + 'V.csv',
        gcp_file,
    )

    plt.figure(figsize=(12, 12))
    plt.scatter(
        wind_field_df['real_x'], wind_field_df['real_y'],
        c=wind_field_df['wind_value'], cmap='magma',
    )
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'wind_field-{angle_str}.png')
    plt.close()
    print(f"Saved wind_field-{angle_str}.png")
