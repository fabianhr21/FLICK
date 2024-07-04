import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


def extract_upc_number(filename):
    match = re.search(r'UPC-(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # If no number is found, place it at the end

def read_output_files(output_dir, keyword,x_frames,y_frames,step=256):
    sorted_list = sorted(os.listdir(output_dir),key=extract_upc_number)
    # Read the output files in one array
    matrix = []
    for filename in sorted_list:
        if keyword in filename:
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
    return matrix, complete_matrix #Complete matrix is the matrix with all the frames, only visualization purposes
#################################################### INTERPOLATION ####################################################
def interpolate(matrix1, matrix2, overlap, axis):
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

def overlap_matrix(matrix, step, overlap, y_dir, x_frames):
    combined_matrix = np.zeros((y_dir, overlap*x_frames+256))
    n = 0
    for i in range(y_dir - step, -1, -overlap):
        row = 0
        for j in range(0, overlap*x_frames, overlap):
            if n >= len(matrix):
                continue
            combined_matrix[i:i+step, j:j+step] = matrix[n]
            n += 1
            if j > 0 and n % x_frames != 0:
                combined_matrix[i:i+step, j:j+step] = interpolate(combined_matrix[i:i+step, j:j+step], matrix[n-1], overlap, 'horizontal')
            
            if row > 0:
                combined_matrix[i:i+step, j:j+step] = interpolate(combined_matrix[i:i+step, j:j+step], matrix[n-x_frames], overlap, 'vertical')
        row += 1
    return combined_matrix

def vel_magNdir(U, V,W=0):
    VMAG = np.sqrt(U**2 + V**2 + W**2)
    VDIR2D = np.arctan2(V,U)
    return VMAG, VDIR2D