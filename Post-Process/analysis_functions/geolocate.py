import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../../Pre-Process/')
import STL2GeoTool_loop
sys.path.append('../')

basename = 'UPC_small_origin'
GCP_path = '../../Pre-Process/output/'
DATASET_PATH = '../final_output/'
OUTPUT_PATH = './output/'

def vel_magNdir(U, V,W=0):
    VMAG = np.sqrt(U**2 + V**2 + W**2)
    VDIR2D = np.arctan2(V,U)
    
    return VMAG, VDIR2D

def GEOLOCATE(U,V, GCP_PATH):
    GCP = pd.read_csv(GCP_PATH)
    U = np.loadtxt(U, delimiter=',')
    V = np.loadtxt(V, delimiter=',')
    min_x = GCP['min_x']
    max_x = GCP['max_x']
    min_y = GCP['min_y']
    max_y = GCP['max_y']
    
    VMAG, VDIR = vel_magNdir(U,V)
    
    rows, cols = VMAG.shape
    
    x_coords = np.linspace(min_x, max_x, cols)
    y_coords = np.linspace(min_y, max_y, rows)
    
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    
    wind_field_df = pd.DataFrame({
    'pixel_x': np.tile(np.arange(cols), rows),
    'pixel_y': np.repeat(np.arange(rows), cols),
    'real_x': x_grid.ravel(),
    'real_y': y_grid.ravel(),
    'wind_value': VMAG.ravel(),
    'U': U.ravel(),
    'V': V.ravel(),
    'dir': VDIR.ravel()
    
    })
    
    return wind_field_df, rows, cols # GEOLOCATED WIND FIELD DATAFRAME

if __name__ == '__main__':
    for wind_angle in STL2GeoTool_loop.WIND_DIRECTION:
        DATASET_PATH = DATASET_PATH + f'output{wind_angle}-{basename}/'
        GCP_PATH = GCP_path + f'output{wind_angle}-{basename}/GCP.csv'
        OUTPUT_PATH = OUTPUT_PATH + f'output{wind_angle}-{basename}/'
        
        U = DATASET_PATH + 'U.csv'
        V = DATASET_PATH + 'V.csv'
        
        wind_field_df,rows, cols = GEOLOCATE(U,V, GCP_PATH)
        
        # Plot wind value
        # set the axis of the imshow to the real x and y coordinates in an imshow
        
        plt.figure(figsize=(12, 12))
        plt.scatter(wind_field_df['real_x'], wind_field_df['real_y'], c=wind_field_df['wind_value'], cmap='magma')
        # plt.imshow(wind_field_df['wind_value'].values.reshape(rows, cols), cmap='magma')
        # plt.gca().set_xlim([wind_field_df['real_x'].min(), wind_field_df['real_x'].max()])
        # plt.gca().set_ylim([wind_field_df['real_y'].min(), wind_field_df['real_y'].max()])
        plt.colorbar()
        plt.show()
        plt.savefig(f'image-{wind_angle}.png')
        plt.close()
        
        
    

