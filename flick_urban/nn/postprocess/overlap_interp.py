"""
overlap_interp.py — Alternative interpolation-based tile stitching.

Provides the same tile-stitching functionality as overlap.py but uses
a simpler pre-computed step/overlap approach without argparse.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from flick_urban.nn.postprocess.overlap import (
    extract_upc_number, read_output_files, interpolate,
    overlap_matrix, vel_magNdir, save_matrix_as_image,
)

# Default parameters (override at runtime as needed)
p_overlap = 0.5
N_points = 256
y_frames = 5
x_frames = 5
DATASET_PATH = './output/'
output_dir = './final_output/output/'
basename = 'grid_of_cubes'
x_factor = 1.5
y_factor = 1.5

if __name__ == '__main__':
    step    = int(p_overlap * N_points)
    overlap = N_points - step
    y_dir   = y_frames * N_points
    x_dir   = x_frames * N_points

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    matrix_U = read_output_files(DATASET_PATH, 'UGT')
    overlap_matrix_U = overlap_matrix(matrix_U, N_points, step, overlap, y_dir, x_frames, x_factor, y_factor)
    matrix_V = read_output_files(DATASET_PATH, 'VGT')
    overlap_matrix_V = overlap_matrix(matrix_V, N_points, step, overlap, y_dir, x_frames, x_factor, y_factor)
    VMAG, VDIR = vel_magNdir(overlap_matrix_U, overlap_matrix_V)
    matrix_mask = read_output_files(DATASET_PATH, 'MASK')
    mask = overlap_matrix(matrix_mask, N_points, step, overlap, y_dir, x_frames, x_factor, y_factor)

    save_matrix_as_image(VMAG, output_dir + 'VMAG.png')
