import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from PIL import Image
import sys
sys.path.append('../../Pre-Process/')
import STL2GeoTool_loop

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
    # complete_matrix = np.zeros((y_dir, x_dir))
    # n = 0
    # for i in range(y_dir - N_points, -1, -N_points):
    #     for j in range(0, x_dir, N_points):
    #         complete_matrix[i:i + N_points, j:j + N_points] = matrix[n]
    #         n += 1
    return matrix #, complete_matrix  # Complete matrix is the matrix with all the frames, only visualization purposes

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
