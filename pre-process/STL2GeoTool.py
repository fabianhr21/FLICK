# STL2GeoTool_loop.py
from __future__ import print_function, division
##
import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI
import csv
import os, re, glob, subprocess, numpy as np
from gmtry_utils import geometrical_magnitudes, save_scalarfield, plane_generation, calculate_bounding_box, append_UV_features, move_stl_to_origin, rotate_geometry
from gpu_gmtry_utils import geometrical_data_extractor_gpu,geometrical_magnitudes_gpu
import pyQvarsi
from stl import mesh
import shutil
import argparse

from time import perf_counter
start = perf_counter()

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# Folders & files
STL_DIR = './'
STL_BASENAME = 'campusnord_256'
POST_DIR_MAIN = './output/'

STL_SCALE = 1.0
DIST_RESOLUTION = 1.0

# Parameters
# Wind direction each 22.5 degrees
WIND_DIRECTION =  [0,22.5,45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5] 
STL_ROT_ANGLE = [0.0, 0.0, 0.0]
STL_DISPLACEMENT = [640, 640, 0.0]
N_POINTS = 1280
STEP_SIZE = N_POINTS // 2 # For 1 meter resolution
p_overlap = 1
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
    parser.add_argument('-use_gpu', default=True, help='Use GPU for computations')
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    j_global, i_global, k_global = STL_DISPLACEMENT
    args = get_args()

    STL_BASENAME = args.stl_basename
    POST_DIR_MAIN = args.output_path
    STEP_SIZE = args.step_size
    N_POINTS = args.n_points
    WIND_DIRECTION = args.wind_direction
    p_overlap = args.p_overlap
    use_gpu = args.use_gpu

    overlap = int(2 * STEP_SIZE * p_overlap)

    for wind_angle in WIND_DIRECTION:
        POST_DIR = POST_DIR_MAIN + f'output{wind_angle}-{STL_BASENAME}/'
        rotated_stl_basename = STL_BASENAME  # To avoid changing the original base name
        
        if mpi_rank == 0:
            print(f"N_POINTS: {N_POINTS}, STEP_SIZE: {STEP_SIZE}, WIND_DIRECTION: {wind_angle}")
            if not os.path.exists(POST_DIR):
                os.makedirs(POST_DIR)
            shutil.copy(STL_DIR + STL_BASENAME + '.stl', POST_DIR + STL_BASENAME + '_geo.stl')
            shutil.copy(STL_DIR + STL_BASENAME + '.stl', POST_DIR + STL_BASENAME + '.stl')
            STL_GEOREF = STL_BASENAME + '_geo'
            
            rotate_geometry(POST_DIR + STL_BASENAME + '.stl', POST_DIR + STL_BASENAME, wind_angle)
            rotate_geometry(POST_DIR + STL_GEOREF + '.stl', POST_DIR + STL_GEOREF, wind_angle)
            rotated_stl_basename = STL_BASENAME
            rotated_stl_basename_geo = STL_GEOREF
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

            # mpi_comm.Barrier()
            MPI.COMM_WORLD.Barrier()
        else:
            STL_GEOREF = STL_BASENAME + '_geo'
            rotated_stl_basename = STL_BASENAME
            rotated_stl_basename_geo = STL_GEOREF
            x_length = mpi_comm.bcast(x_length if mpi_rank == 0 else None, root=0)
            y_length = mpi_comm.bcast(y_length if mpi_rank == 0 else None, root=0)


        print(f'Domain size in x: {x_length} and y: {y_length}')
        n = 0
        x_frames = 0
        y_frames = 0
        for i in range(0, -y_length, -overlap):
            print(f"[Rank {mpi_rank}] Starting processing at wind angle {wind_angle}")

            y_frames += 1
            for j in range(0, -x_length, -overlap):
                if i == 0:
                    x_frames += 1
                STL_DISPLACEMENT = [j-j_global, i-i_global, 0]
                print(STL_DISPLACEMENT)
                # Measure plane generation
                t0 = perf_counter()
                int_mesh = plane_generation(STEP_SIZE, N_POINTS, N_POINTS)
                t1 = perf_counter()
                if mpi_rank == 0:
                    print(f"[Timing] plane_generation: {t1 - t0:.3f} seconds")

                # Copy mesh points
                int_xyz = int_mesh.xyz.copy()

                # Measure geometry processing
                t2 = perf_counter()
                if use_gpu:
                    output_fields = geometrical_magnitudes_gpu(
                        STL_FILE=POST_DIR + rotated_stl_basename + '.stl',
                        target_mesh=int_xyz,
                        stl_angle=STL_ROT_ANGLE,
                        stl_displ=STL_DISPLACEMENT,
                        stl_scale=STL_SCALE,
                        dist_resolution=DIST_RESOLUTION,
                        z_tol=1e-2
                    )
                else:
                    output_fields = geometrical_magnitudes(
                        STL_FILE=POST_DIR + rotated_stl_basename + '.stl',
                        target_mesh=int_xyz,
                        stl_angle=STL_ROT_ANGLE,
                        stl_displ=STL_DISPLACEMENT,
                        stl_scale=STL_SCALE,
                        dist_resolution=DIST_RESOLUTION,
                        z_tol=1e-2
                    )
                t3 = perf_counter()
                if mpi_rank == 0:
                    print(f"[Timing] geometrical_magnitudes{'_gpu' if use_gpu else ''}: {t3 - t2:.3f} seconds")

                # Save H5
                t4 = perf_counter()
                if pyQvarsi.utils.is_rank_or_serial(0):
                    int_mesh.save(POST_DIR + rotated_stl_basename + f'-{n}-geodata.h5', mpio=False)
                if pyQvarsi.utils.is_rank_or_serial(1):
                    output_fields.save(POST_DIR + rotated_stl_basename + f'-{n}-geodata.h5', mpio=False)
                t5 = perf_counter()
                if mpi_rank == 0:
                    print(f"[Timing] H5 save: {t5 - t4:.3f} seconds")

                print(f"[Rank {mpi_rank}] Step {n} total time: {t5 - t0:.3f} seconds\n")


                pyQvarsi.pprint(0, 'Done.', flush=True)
                pyQvarsi.cr_info()
                n += 1
                print(f"[Rank {mpi_rank}] Step {n} took {perf_counter() - start:.2f} seconds")
        mpi_comm.Barrier()

        if mpi_rank == 0:
            with open(POST_DIR + 'global_vars.txt', 'w') as f:
                f.write(f"x_frames={x_frames}\n")
                f.write(f"y_frames={y_frames}\n")
        mpi_comm.Barrier()

