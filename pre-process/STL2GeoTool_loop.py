# STL2GeoTool_loop.py
from __future__ import print_function, division
##
import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI
import csv
import os, re, glob, subprocess, numpy as np
from gmtry_utils import geometrical_magnitudes, save_scalarfield, plane_generation, calculate_bounding_box, append_UV_features, move_stl_to_origin, rotate_geometry
import pyQvarsi
from stl import mesh
import shutil
import argparse

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# Folders & files
STL_DIR = './'
STL_BASENAME = 'UPCNord_geometry'
POST_DIR_MAIN = './output/'

STL_SCALE = 1.0
DIST_RESOLUTION = 1.0

# Parameters
WIND_DIRECTION =  [0] #, 187.04, 177.39, 194.31] #, 192.43, 204.27, 184.09, 174.25, 185.76, 181.64, 181.8, 177.15, 172.11, 177.66, 174.09, 221.76, 187.22, 198.28, 181.22, 195.38, 136.49, 187.77] # Rotates geometry to align with wind direction (degrees)
# WIND_DIRECTION = [0,20,40,60,80]
STL_ROT_ANGLE = [0.0, 0.0, 0.0]
STL_DISPLACEMENT = [0, 0, 0.0]
STEP_SIZE = 128
N_POINTS = 256
p_overlap = 0.50
overlap = int(2 * STEP_SIZE * p_overlap)

def get_args():
    parser = argparse.ArgumentParser(description='args for 2D H5 data samples training')
    parser.add_argument('-dataset_path', default=STL_DIR, help='dataset folder name.')
    parser.add_argument('-stl_basename', default=STL_BASENAME, help='input dataset files base name')
    parser.add_argument('-output_path', default=POST_DIR_MAIN, help='output folder name')
    parser.add_argument('-step_size', type=int, default=STEP_SIZE, help='step size')
    parser.add_argument('-n_points', type=int, default=N_POINTS, help='number of points')
    parser.add_argument('-p_overlap', type=int, default=p_overlap, help='overlap')
    parser.add_argument('-wind_direction', type=int, default=WIND_DIRECTION, help='wind direction')
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = get_args()
    STL_DIR = args.dataset_path
    STL_BASENAME = args.stl_basename
    POST_DIR_MAIN = args.output_path
    STEP_SIZE = args.step_size
    N_POINTS = args.n_points
    WIND_DIRECTION = args.wind_direction
    p_overlap = args.p_overlap

    overlap = int(2 * STEP_SIZE * p_overlap)

    for wind_angle in WIND_DIRECTION:
        POST_DIR = POST_DIR_MAIN + f'output{wind_angle}-{STL_BASENAME}/'
        rotated_stl_basename = STL_BASENAME  # To avoid changing the original base name
        
        if mpi_rank == 0:
            if not os.path.exists(POST_DIR):
                os.makedirs(POST_DIR)
            shutil.copy(STL_DIR + STL_BASENAME + '.stl', POST_DIR + STL_BASENAME + '_geo.stl')
            shutil.copy(STL_DIR + STL_BASENAME + '.stl', POST_DIR + STL_BASENAME + '.stl')
            STL_GEOREF = STL_BASENAME + '_geo'
            
            rotate_geometry(POST_DIR + STL_BASENAME + '.stl', POST_DIR + STL_BASENAME, wind_angle)
            rotate_geometry(POST_DIR + STL_GEOREF + '.stl', POST_DIR + STL_GEOREF, wind_angle)
            rotated_stl_basename = STL_BASENAME
            rotated_stl_basename_geo = STL_GEOREF
            print(rotated_stl_basename)
            print(f'Rotated geometry to align with wind direction: {wind_angle} in file {rotated_stl_basename_geo}.stl')
    
            args = get_args()
            # Save rotated geometry coordinates
            GCP = {}
            min_coords, max_coords = calculate_bounding_box(POST_DIR + rotated_stl_basename_geo + '.stl')
            min_x, min_y = min_coords[:2]
            max_x, max_y = max_coords[:2]
            GCP['min_x'] = min_x
            GCP['min_y'] = min_y
            GCP['max_x'] = max_x
            GCP['max_y'] = max_y
            
            with open(POST_DIR + 'GCP.csv', 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(GCP.keys())
                writer.writerow(GCP.values())
                     
            move_stl_to_origin(POST_DIR + rotated_stl_basename + '.stl', POST_DIR + rotated_stl_basename + '.stl')
            min_coords, max_coords = calculate_bounding_box(POST_DIR + rotated_stl_basename + '.stl')
            min_x, min_y = min_coords[:2]
            max_x, max_y = max_coords[:2]
            x_length = int(np.ceil(max_x - min_x))
            y_length = int(np.ceil(max_y - min_y))
        
        x_length = mpi_comm.bcast(x_length if mpi_rank == 0 else None, root=0)
        y_length = mpi_comm.bcast(y_length if mpi_rank == 0 else None, root=0)

        mpi_comm.Barrier()

        print(f'Domain size in x: {x_length} and y: {y_length}')
        n = 0
        x_frames = 0
        y_frames = 0
        for i in range(0, -y_length, -overlap):
            y_frames += 1
            for j in range(0, -x_length, -overlap):
                if i == 0:
                    x_frames += 1
                STL_DISPLACEMENT = [j, i, 0]
                print(STL_DISPLACEMENT)
                int_mesh = plane_generation(STEP_SIZE, N_POINTS, N_POINTS)
                pyQvarsi.pprint(0, 'plane mesh Generated', flush=True)

                int_xyz = int_mesh.xyz.copy()

                pyQvarsi.pprint(0, 'STL DIR: ', POST_DIR, flush=True)
                pyQvarsi.pprint(0, 'POST DIR: ', POST_DIR, flush=True)

                if mpi_rank == 0 and not os.path.exists(POST_DIR):
                    os.makedirs(POST_DIR)

                pyQvarsi.pprint(0, "reading ", POST_DIR + rotated_stl_basename + '.stl')
                output_fields = geometrical_magnitudes(POST_DIR + rotated_stl_basename + '.stl', int_xyz, stl_angle=STL_ROT_ANGLE, stl_displ=STL_DISPLACEMENT, stl_scale=STL_SCALE, dist_resolution=DIST_RESOLUTION)

                if pyQvarsi.utils.is_rank_or_serial(0):
                    int_mesh.save(POST_DIR + rotated_stl_basename + f'-{n}-geodata.h5', mpio=False)

                mpi_comm.Barrier()

                metadata = {'STL name:': rotated_stl_basename + '.stl'}
                if pyQvarsi.utils.is_rank_or_serial(1):
                    output_fields.save(POST_DIR + rotated_stl_basename + f'-{n}-geodata.h5', mpio=False)

                pyQvarsi.pprint(0, 'Done.', flush=True)
                pyQvarsi.cr_info()
                n += 1
        mpi_comm.Barrier()

        if mpi_rank == 0:
            with open(POST_DIR + 'global_vars.txt', 'w') as f:
                f.write(f"x_frames={x_frames}\n")
                f.write(f"y_frames={y_frames}\n")
        mpi_comm.Barrier()
