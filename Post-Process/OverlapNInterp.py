import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from PP_functions import extract_upc_number, read_output_files, interpolate, overlap_matrix

#### PARAMETERS FROM WIND-NN ####
p_overlap = 0.5
step = 256
y_frames = 5    
x_frames = 7
overlap = int(p_overlap*step)
output_dir = './BIM/'
##################################

if __name__ == '__main__':
    matrix_U,_ = read_output_files(output_dir, 'UGT') # In the inference scripts the output writes 'UGT' or 'VGT' for U and V wind fields
    combined_matrix_U = overlap_matrix(matrix_U, step, overlap, y_dir, x_frames)
    matrix_V,_ = read_output_files(output_dir, 'VGT')
    combined_matrix_V = overlap_matrix(matrix_V, step, overlap, y_dir, x_frames)
    print('Combined matrix shape:', combined_matrix_U.shape)
    print(len(matrix))
    # Adjust the matrix to remove vide regions
    combined_matrix_U = combined_matrix_U[y_frames*overlap:, :]
    combined_matrix_U = combined_matrix_U[:, :-overlap]
    combined_matrix_V = combined_matrix_V[y_frames*overlap:, :]
    combined_matrix_V = combined_matrix_V[:, :-overlap]
    VMAG,VDIR2D = vel_magNdir(combined_matrix_U, combined_matrix_V)
    # Create the plot
    fig, ax = plt.subplots()
    cax = ax.imshow(VMAG, cmap='magma')
    # Add the colorbar
    cbar = fig.colorbar(cax, ax=ax)
    plt.show()
    