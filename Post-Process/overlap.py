#### GOOOD ONE ######
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from PIL import Image

#### PARAMETERS FROM WIND-NN ####
p_overlap = 0.5
N_points = 256
step = int(p_overlap * N_points)
overlap = N_points - step
y_frames = 5
x_frames = 5
y_dir = y_frames * N_points
x_dir = x_frames * N_points
DATASET_PATH='../Wind-NN/output/'
output_dir = './final_output/'
basename = 'grid_of_cubes'
##################################
def save_matrix_as_image(matrix, output_file,colormap='magma'):
    # Normalize matrix values to 0-1
    normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

    # Apply colormap
    colormap_function = plt.get_cmap(colormap)
    colored_matrix = colormap_function(normalized_matrix)

    # Convert to 8-bit per channel (0-255) and remove alpha channel
    image_data = (colored_matrix[:, :, :3] * 255).astype(np.uint8)
    # Save image
    image = Image.fromarray(image_data)
    image.save(output_file)
    print(f"Image saved as {output_file}")

def extract_upc_number(filename):
    match = re.search(r'-(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # If no number is found, place it at the end

def read_output_files(output_dir, keyword):
    sorted_list = sorted(os.listdir(output_dir), key=extract_upc_number)
    # Read the output files in one array
    matrix = []
    for filename in sorted_list:
        if keyword in filename:
            with open(output_dir + filename) as f:
                data = np.genfromtxt(f, delimiter=',')
                matrix.append(data)
            f.close()
    complete_matrix = np.zeros((y_dir, x_dir))
    n = 0
    for i in range(y_dir - N_points, -1, -N_points):
        for j in range(0, x_dir, N_points):
            complete_matrix[i:i + N_points, j:j + N_points] = matrix[n]
            n += 1
    return matrix, complete_matrix  # Complete matrix is the matrix with all the frames, only visualization purposes

def interpolate(matrix1, matrix2, overlap, axis, factor=1):
    # Generate the interpolation weights using an exponential decay function
    alpha = np.linspace(0, 1, overlap)
    weights = np.exp(-alpha * factor)  # Adjust the factor to control the decay rate
    if axis == 'horizontal':
        for i in range(overlap):
            weight = weights[i]
            matrix1[:, i] = matrix1[:, i] * (1 - weight) + matrix2[:, -overlap + i] * weight
        return matrix1
    if axis == 'vertical':
        for i in range(overlap):
            weight = weights[i]
            matrix1[-overlap + i, :] = matrix1[-overlap + i, :] * weight + matrix2[i, :] * (1 - weight)
        return matrix1

def overlap_matrix(matrix, N_points, step, overlap, y_dir, x_frames,x_factor,y_factor):
    combined_matrix = np.zeros((y_dir, step * x_frames + N_points))
    memory = []
    n = 0
    row = 0
    for i in range(y_dir - N_points, -1, -step):
        for j in range(0, step * x_frames, step):
            if n >= len(matrix):
                continue
            temp_matrix = matrix[n].copy()
            # print(n)
            if j > 0:
                temp_matrix = interpolate(temp_matrix, matrix[n - 1], overlap, 'horizontal',x_factor)
            if row > 0:
                temp_matrix = interpolate(temp_matrix, memory[n - x_frames], overlap, 'vertical',y_factor)
            combined_matrix[i:i + N_points, j:j + N_points] = temp_matrix
            memory.append(temp_matrix)
            n += 1
        row += 1
    return combined_matrix

def vel_magNdir(U, V,W=0):
    VMAG = np.sqrt(U**2 + V**2 + W**2)
    VDIR2D = np.arctan2(V,U)
    
    return VMAG, VDIR2D

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
x_factor = 1.5
y_factor = 1.5
matrix_U, matrix = read_output_files(DATASET_PATH, 'UGT') # In the inference scripts the output writes 'UGT' or 'VGT' for U and V wind fields
overlap_matrix_U = overlap_matrix(matrix_U, N_points, step, overlap, y_dir, x_frames,x_factor, y_factor)
matrix_V, matrix = read_output_files(DATASET_PATH, 'VGT')
overlap_matrix_V = overlap_matrix(matrix_V, N_points, step, overlap, y_dir, x_frames,x_factor,y_factor)
VMAG,VDIR = vel_magNdir(overlap_matrix_U, overlap_matrix_V)
matrix_mask,_= read_output_files(DATASET_PATH, 'MASK')
mask = overlap_matrix(matrix_mask, N_points, step, overlap, y_dir, x_frames,x_factor,y_factor)

cut_U = overlap_matrix_U[overlap*y_frames-overlap:, 65:768]
cut_V = overlap_matrix_V[overlap*y_frames-overlap:, 65:768]
mask_cut = mask[overlap*y_frames-overlap:, 65:768]
cutcut_U = cut_U[0:600,:]
cutcut_V = cut_V[0:600,:]
mask_cutcut = mask_cut[0:600,:]
VMAG,VDIR = vel_magNdir(cutcut_U, cutcut_V)

mask_multiply = mask_cutcut * VMAG
# Save matrix as image
save_matrix_as_image(VMAG, output_dir + 'VMAG.png')
save_matrix_as_image(mask_multiply, output_dir + 'VMAG_mask.png')

