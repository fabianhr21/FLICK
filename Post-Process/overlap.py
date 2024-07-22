#### GOOOD ONE ######
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from PIL import Image
import argparse
import sys
sys.path.append('../Pre-Process/')
import STL2GeoTool_loop

#### PARAMETERS FROM WIND-NN ####
p_overlap = 0.5                     # Overlap percentage
N_points = 256                      # Number of points in the output matrix
y_frames = 5                        # Number of frames in the y direction
x_frames = 5                        # Number of frames in the x direction
DATASET_PATH='../Wind-NN/output/'   # Path to the output files
output_dir = './final_output/'      # Path to save the final output
basename = 'grid_of_cubes'          # Basename of the output files
x_factor = 1.5                      # Window weight in the x direction
y_factor = 1.5                      # Window weight in the y direction
##################################

def get_args():
    parser = argparse.ArgumentParser(description='args for 2D H5 data samples training')
    parser.add_argument('-overlap', type=float, default=p_overlap, help='overlap percentage')
    parser.add_argument('-N_points', type=int, default=N_points, help='number of points in the output matrix')
    parser.add_argument('-y_frames', type=int, default=y_frames, help='number of frames in the y direction')
    parser.add_argument('-x_frames', type=int, default=x_frames, help='number of frames in the x direction')
    parser.add_argument('-dataset_path', default=DATASET_PATH, help='dataset folder name.')
    parser.add_argument('-output_dir', default=output_dir, help='output folder name')
    parser.add_argument('-basename', default=basename, help='output files basename')
    parser.add_argument('-x_factor', type=float, default=x_factor, help='window weight in the x direction')
    parser.add_argument('-y_factor', type=float, default=y_factor, help='window weight in the y direction')
    args, _ = parser.parse_known_args()
    return args

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

def remove_empty_lines(matrix):
    if not np.any(matrix):
        return np.array([[]])

    first_non_zero_row = np.argmax(np.any(matrix != 0, axis=1))
    matrix = matrix[first_non_zero_row:]
    last_non_zero_col = np.max(np.nonzero(np.any(matrix != 0, axis=0))) + 1
    matrix = matrix[:, :last_non_zero_col]

    return matrix


def read_global_vars():
    global_vars = {}
    with open('../Pre-Process/global_vars.txt', 'r') as f:
        for line in f:
            name, value = line.strip().split('=')
            global_vars[name] = int(value)
    return global_vars


if __name__ == '__main__':
    x_frames, y_frames = read_global_vars().values()
    for wind_angle in STL2GeoTool_loop.WIND_DIRECTION:
        args = get_args()
        N_points = args.N_points
        DATASET_PATH = args.dataset_path
        output_dir = args.output_dir
        basename = args.basename
        x_factor = args.x_factor
        y_factor = args.y_factor

        p_overlap = STL2GeoTool_loop.p_overlap
        print(f'x_Frames: {x_frames}, y_Frames: {y_frames}')
        print(f'Processing output{wind_angle}-{basename}...')

        
        step = int(p_overlap * N_points)    # Number of overlapping points
        overlap = N_points - step           # Number of non-overlapping points
        y_dir = y_frames * N_points         # Number of points in the y direction
        x_dir = x_frames * N_points         # Number of points in the x direction
        
        output_dir = output_dir + f'output{wind_angle}-{basename}/'
        DATASET_PATH = DATASET_PATH + f'output{wind_angle}-{basename}/' 
        
        # Count files in the DATASET_PATH directory
        # files = os.listdir(DATASET_PATH)
        # h5_files = [file for file in files if file.endswith('MASK_matrix.csv')]
        # N_files = len(h5_files)
        # x_frames = np.sqrt(N_files).astype(int)
        # y_frames = np.sqrt(N_files).astype(int)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        matrix_U, matrix = read_output_files(DATASET_PATH, 'UGT') # In the inference scripts the output writes 'UGT' or 'VGT' for U and V wind fields
        overlap_matrix_U = overlap_matrix(matrix_U, N_points, step, overlap, y_dir, x_frames,x_factor, y_factor)
        matrix_V, matrix = read_output_files(DATASET_PATH, 'VGT')
        overlap_matrix_V = overlap_matrix(matrix_V, N_points, step, overlap, y_dir, x_frames,x_factor,y_factor)
        VMAG,VDIR = vel_magNdir(overlap_matrix_U, overlap_matrix_V)
        matrix_mask,_= read_output_files(DATASET_PATH, 'MASK')
        mask = overlap_matrix(matrix_mask, N_points, step, overlap, y_dir, x_frames,x_factor,y_factor)

        # Clean matrices
        overlap_matrix_U = remove_empty_lines(overlap_matrix_U)
        overlap_matrix_V = remove_empty_lines(overlap_matrix_V)
        mask = remove_empty_lines(mask)
        
        VMAG,VDIR = vel_magNdir(overlap_matrix_U, overlap_matrix_V)

        masked_matrix = mask * VMAG
        
        # SAVE THE FINAL OUTPUT
        np.savetxt(output_dir + 'VMAG.csv', VMAG, delimiter=',')
        np.savetxt(output_dir + 'Mask.csv', mask, delimiter=',')
        np.savetxt(output_dir + 'U.csv', overlap_matrix_U, delimiter=',')
        np.savetxt(output_dir + 'V.csv', overlap_matrix_V, delimiter=',')
        save_matrix_as_image(VMAG, output_dir + 'VMAG.png')
        save_matrix_as_image(masked_matrix, output_dir + 'VMAG_mask.png')