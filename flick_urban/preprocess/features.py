# ADD_FEAT.py

import STL2GeoTool_loop
from gmtry_utils import append_UV_features
import os
import argparse

# Add arguments when calling the function
STL_BASENAME = 'campusnord_512'

def get_args():
    parser = argparse.ArgumentParser(description='args for 2D H5 data samples training')
    parser.add_argument('-stl_basename', default=STL_BASENAME, help='input dataset files base name')
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = get_args()
    STL_BASENAME = args.stl_basename
    print(f"STL_BASENAME: {STL_BASENAME}")
    # Loop over each output path
    for wind_angle in STL2GeoTool_loop.WIND_DIRECTION:
        OUTPUT_DIR = STL2GeoTool_loop.POST_DIR_MAIN + f'output{wind_angle}-{STL_BASENAME}/'
        # Loop for every file in the output directory
        #data sample indices to load examples from the dataset
        files = os.listdir(OUTPUT_DIR)
        h5_files = [file for file in files if file.endswith('.h5')]
        sample_indices = [i for i in range(0,len(h5_files))]
        # if wind_angle != 0:
        #     STL_BASENAME = f'{STL_BASENAME}{str(wind_angle)}'
        for file in range(0,len(h5_files)): 
            print(f"Adding U and V features to {OUTPUT_DIR}{STL_BASENAME}-{file}-geodata.h5")     
            append_UV_features(f"{OUTPUT_DIR}{STL_BASENAME}-{file}")
            print(f"U and V features added to {OUTPUT_DIR}{STL_BASENAME}-{file}-geodata.h5")   
    
    
    