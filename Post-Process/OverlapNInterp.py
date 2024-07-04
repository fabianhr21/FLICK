import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

#### PARAMETERS FROM WIND-NN ####
p_overlap = 0.5
step = 256
y_frames = 5    
x_frames = 7
overlap = int(p_overlap*step)
output_dir = './BIM/'
##################################

def extract_upc_number(filename):
    match = re.search(r'UPC-(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # If no number is found, place it at the end

def read_output_files(output_dir, keyword):
    sorted_list = sorted(os.listdir(output_dir),key=extract_upc_number)
    # Read the output files in one array
    matrix = []
    for filename in sorted_list:
        if filename.endswith('UGT_matrix.csv'):
            with open(output_dir+filename) as f:
                # print('Reading file:', filename)
                data = np.genfromtxt(f, delimiter=',')
                matrix.append(data)
            f.close()
    x_dir = x_frames*step
    y_dir = y_frames*step
    complete_matrix = np.zeros((y_dir, x_dir))
    n = 0
    for i in range(y_dir - step, -1, -step):
        for j in range(0, x_dir, step):
            complete_matrix[i:i+step, j:j+step] = matrix[n]
            n += 1
            # Define the extent for the image
    plt.imshow(complete_matrix, cmap='magma')
    plt.show()
#################################################### INTERPOLATION ####################################################
def interpolate_overlap(matrix1, matrix2, overlap, axis):
    # Generate the interpolation weights
    alpha = np.linspace(0, 1, overlap)
    
    if axis == 'horizontal':
        for i in range(overlap):
            matrix1[:, -overlap + i] = matrix1[:, -overlap + i] * (1 - alpha[i]) + matrix2[:, i] * alpha[i]
        return matrix1
    elif axis == 'vertical':
        for i in range(overlap):
            matrix1[-overlap + i, :] = matrix1[-overlap + i, :] * (1 - alpha[i]) + matrix2[i, :] * alpha[i]
        return matrix1

single_matrix_height, single_matrix_width = matrix[0].shape
combined_height = int(y_frames * (single_matrix_height - overlap) + overlap)
combined_width = int(x_frames * (single_matrix_width - overlap) + overlap)
# combined_matrix = np.zeros((combined_height, combined_width))
combined_matrix = np.zeros((y_dir, overlap*x_frames+256))
print('Combined matrix shape:', combined_matrix.shape)
print(len(matrix))
n = 0
for i in range(y_dir - step, -1, -overlap):
    row = 0
    for j in range(0, overlap*x_frames, overlap):
        if n >= len(matrix):
            continue
        combined_matrix[i:i+step, j:j+step] = matrix[n]
        n += 1
        if j > 0 and n % x_frames != 0:
            combined_matrix[i:i+step, j:j+step] = interpolate_overlap(combined_matrix[i:i+step, j:j+step], matrix[n-1], overlap, 'horizontal')
        
        if row > 0:
            combined_matrix[i:i+step, j:j+step] = interpolate_overlap(combined_matrix[i:i+256, j:j+256], matrix[n-x_frames], overlap, 'vertical')
    row += 1
#Erase the first rows of the matrix
combined_matrix_U = combined_matrix[5*overlap:, :]
#Erase the last columns of the matrix
combined_matrix_U = combined_matrix_U[:, :-overlap]
# Create the plot
fig, ax = plt.subplots()
cax = ax.imshow(combined_matrix_U, cmap='magma')
# Add the colorbar
cbar = fig.colorbar(cax, ax=ax)
plt.show()