import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from PP_functions import extract_upc_number, read_output_files, interpolate, overlap_matrix,vel_magNdir,save_matrix_as_image

#### PARAMETERS FROM WIND-NN ####
p_overlap = 0.5
N_points = 256
step = int(p_overlap * N_points)
overlap = N_points - step
y_frames = 5
x_frames = 5
y_dir = y_frames * N_points
x_dir = x_frames * N_points
DATASET_PATH='../Wind-NN'
output_dir = './final_output/output/'
basename = 'grid_of_cubes'
x_factor = 1.5
y_factor = 1.5
##################################

if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    matrix_U = read_output_files(DATASET_PATH, 'UGT')
    overlap_matrix_U = overlap_matrix(matrix_U, N_points, step, overlap, y_dir, x_frames,x_factor, y_factor)
    matrix_V = read_output_files(DATASET_PATH, 'VGT')
    overlap_matrix_V = overlap_matrix(matrix_V, N_points, step, overlap, y_dir, x_frames,x_factor,y_factor)
    VMAG,VDIR = vel_magNdir(overlap_matrix_U, overlap_matrix_V)
    matrix_mask= read_output_files(DATASET_PATH, 'MASK')
    mask = overlap_matrix(matrix_mask, N_points, step, overlap, y_dir, x_frames,x_factor,y_factor)
    
    # cut_U = overlap_matrix_U[overlap*y_frames-overlap:, 65:768]
    # cut_V = overlap_matrix_V[overlap*y_frames-overlap:, 65:768]
    # mask_cut = mask[overlap*y_frames-overlap:, 65:768]
    # cutcut_U = cut_U[0:600,:]
    # cutcut_V = cut_V[0:600,:]
    # mask_cutcut = mask_cut[0:600,:]
    # VMAG,VDIR = vel_magNdir(cutcut_U, cutcut_V)
    # mask_multiply = mask_cutcut * VMAG
    
    plt.plot(matrix_U)
    plt.show()
    plt.savefig(f'image.png')
    
    save_matrix_as_image(VMAG, output_dir + 'VMAG.png')
    # save_matrix_as_image(mask_multiply, output_dir + 'VMAG_mask.png')
    
    
    

    