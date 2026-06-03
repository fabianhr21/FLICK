"""
stl2geo.py — Main MPI+GPU preprocessing script.

Converts an STL city model into HDF5 feature maps (MASK, HEGT, WDST, U, V)
tiled across a configurable domain with optional wind-direction rotation.

Usage
-----
    mpirun -n 4 python -m flick_urban.preprocess.stl2geo \\
        -stl_dir ./data/ -stl_basename grid_of_cubes \\
        -output_path ./output/ -wind_direction 0
"""
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI
import csv
import os, re, glob, subprocess, numpy as np
from flick_urban.preprocess.geometry import (
    geometrical_magnitudes, save_scalarfield, plane_generation,
    calculate_bounding_box, append_UV_features, move_stl_to_origin,
    rotate_geometry,
)
from flick_urban.preprocess.gpu_geometry_opt import (
    geometrical_data_extractor_gpu, geometrical_magnitudes_gpu,
)
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
STL_BASENAME = 'grid_of_cubes'
POST_DIR_MAIN = './output/'

STL_SCALE = 1.0
DIST_RESOLUTION = 1

# Parameters
# Wind direction in degrees (0 = wind from south, i.e. geometry faces north)
WIND_DIRECTION = 0.0
STL_ROT_ANGLE = [0.0, 0.0, 0.0]
STEP_SIZE = 640 # Size of the domain in meters
PX_RESOLUTION = 0.25 # Resolution in meters
p_overlap = 1
CENTER_DOMAIN = True # Center the domain around the origin
cx = 0.0  # Static center X coordinate
cy = 0.0  # Static center Y coordinate
cz = 0.0

def get_args(argv=None):
    """Parse CLI arguments for the stl2geo preprocessing script."""
    parser = argparse.ArgumentParser(description='args for 2D H5 data samples training')
    parser.add_argument('-stl_dir', default=STL_DIR, help='dataset folder name.')
    parser.add_argument('-stl_basename', default=STL_BASENAME, help='input dataset files base name')
    parser.add_argument('-output_path', default=POST_DIR_MAIN, help='output folder name')
    # parser.add_argument('-step_size', type=int, default=STEP_SIZE, help='step size')
    parser.add_argument('-step_size', type=int, default=STEP_SIZE, help='step size in meters')
    parser.add_argument('-px_resolution', type=float, default=PX_RESOLUTION, help='pixel resolution in meters')
    parser.add_argument('-p_overlap', type=int, default=p_overlap, help='p-overlap percentage, 1 is 0 overlap')
    parser.add_argument('-wind_direction', default=WIND_DIRECTION, help='wind direction list')
    parser.add_argument('-use_gpu', default=True, help='Use GPU for computations')
    parser.add_argument('-center_domain', type=bool, default=CENTER_DOMAIN, help='Center the domain around the origin')
    parser.add_argument('-batch_size', type=int, default=4092, help='batch size for data loading')
    args, _ = parser.parse_known_args()
    return args

def _angle_str(a):
    """Format angle as int string when a whole number, else as float string."""
    return str(int(a)) if float(a) == int(float(a)) else str(a)


if __name__ == "__main__":
    args = get_args()
    PX_RESOLUTION = args.px_resolution
    STL_DIR = args.stl_dir
    STL_BASENAME = args.stl_basename
    POST_DIR_MAIN = args.output_path
    # STEP_SIZE = args.step_size
    # N_POINTS = args.n_points
    STEP_SIZE = args.step_size # Resolution of 0.25 m
    # WIND_DIRECTION = [float(angle) for angle in args.wind_direction]
    WIND_DIRECTION = [float(args.wind_direction)]
    p_overlap = args.p_overlap
    use_gpu = args.use_gpu

    print(WIND_DIRECTION)
    print(f"Pixel resolution: {PX_RESOLUTION} meters")

    # N_POINTS = int(STEP_SIZE / PX_RESOLUTION) # Resolution of 0.25 m
    # N_POINTS = 1421
    N_POINTS = int(STEP_SIZE / PX_RESOLUTION)  # To ensure center point
    STL_DISPLACEMENT = [STEP_SIZE, STEP_SIZE, 0.0] # Displacement to center to the N_POINTS center
    
    overlap = int(2 * STEP_SIZE * p_overlap)

    j_global, i_global, k_global = STL_DISPLACEMENT
    
    # CLEAR OR CREATE OUTPUT MAIN DIRECTORY FIRST
    if mpi_rank == 0:
        # ensure POST_DIR_MAIN exists before creating subdirs or calling rmtree
        if not os.path.exists(POST_DIR_MAIN):
            os.makedirs(POST_DIR_MAIN, exist_ok=True)

        for wind_angle in WIND_DIRECTION:
            POST_DIR = os.path.join(POST_DIR_MAIN, f'output{_angle_str(wind_angle)}-{STL_BASENAME}/')
            # rmtree sólo si existe
            if os.path.exists(POST_DIR):
                shutil.rmtree(POST_DIR)
                pyQvarsi.pprint(0, f"Cleared: {POST_DIR}", flush=True)
            # crear la carpeta destino vacía para los archivos que vas a copiar
            os.makedirs(POST_DIR, exist_ok=True)
            pyQvarsi.pprint(0, f"Created: {POST_DIR}", flush=True)
            
    mpi_comm.Barrier()

    for wind_angle in WIND_DIRECTION:
        pyQvarsi.pprint(0, f'Starting processing for wind direction: {wind_angle}', flush=True)
        POST_DIR = os.path.join(POST_DIR_MAIN, f'output{_angle_str(wind_angle)}-{STL_BASENAME}/')
        rotated_stl_basename = STL_BASENAME  # To avoid changing the original base name

        if mpi_rank == 0:
            # print(f"N_POINTS: {N_POINTS}, STEP_SIZE: {STEP_SIZE}, WIND_DIRECTION: {wind_angle}")
            # verify STL_DIR is set and the source file exists
            src = os.path.join(POST_DIR, STL_BASENAME + '.stl')
            # if not os.path.exists(src):
            #     raise FileNotFoundError(f"Source STL not found: {src}")
            # POST_DIR ya fue creado arriba, así que copiar debe funcionar
            shutil.copy(STL_DIR + STL_BASENAME + '.stl', POST_DIR + STL_BASENAME + '_geo.stl')
            shutil.copy(STL_DIR + STL_BASENAME + '.stl', POST_DIR + STL_BASENAME + '.stl')
            STL_GEOREF = STL_BASENAME + '_geo'
            
            # Rotate around the static center
            rotate_geometry(
                POST_DIR + STL_BASENAME + '.stl',
                POST_DIR + STL_BASENAME,
                wind_angle,
                center_world=(cx, cy, cz)
            )
            rotate_geometry(
                POST_DIR + STL_GEOREF + '.stl',
                POST_DIR + STL_GEOREF,
                wind_angle,
                center_world=(cx, cy, cz)
            )
            rotated_stl_basename = STL_BASENAME
            rotated_stl_basename_geo = STL_GEOREF
            # pyQvarsi.pprint(0,f'Rotated geometry to align with wind direction: {wind_angle} in file {rotated_stl_basename_geo}.stl',flush=True)
    
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
            GCP["Center_X"] = (min_x + max_x) / 2
            GCP["Center_Y"] = (min_y + max_y) / 2
            center_x = GCP["Center_X"]
            center_y = GCP["Center_Y"]

            # print(f"Bounding box: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
            # print(f"Center coordinates: Center_X={center_x}, Center_Y={center_y}")

                     
            move_stl_to_origin(POST_DIR + rotated_stl_basename + '.stl', POST_DIR + rotated_stl_basename + '.stl')
            min_coords, max_coords = calculate_bounding_box(POST_DIR + rotated_stl_basename + '.stl')
            min_x, min_y = min_coords[:2]
            max_x, max_y = max_coords[:2]
            x_length = int(np.ceil(max_x - min_x))
            y_length = int(np.ceil(max_y - min_y))

            x_length = mpi_comm.bcast(x_length if mpi_rank == 0 else None, root=0)
            y_length = mpi_comm.bcast(y_length if mpi_rank == 0 else None, root=0)
            
            if CENTER_DOMAIN == True:
                # Center the domain around the origin
                j_global = x_length // 2
                i_global = y_length // 2
                STL_DISPLACEMENT = [j_global, i_global, 0]
                # pyQvarsi.pprint(0, f"Centering domain at: {STL_DISPLACEMENT}", flush=True)

                # New domain extent
                min_x = center_x - STEP_SIZE
                min_y = center_y - STEP_SIZE
                max_x = center_x + STEP_SIZE
                max_y = center_y + STEP_SIZE
                # print(f"New domain extent: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
                # Update GCP with new coordinates
                GCP['new_min_x'] = min_x
                GCP['new_min_y'] = min_y
                GCP['new_max_x'] = max_x
                GCP['new_max_y'] = max_y

            else:
                # Use the original displacement
                STL_DISPLACEMENT = [j_global, i_global, 0]
                # print(f"Using original displacement: {STL_DISPLACEMENT}")
            
            # Save GCP to a file
            with open(POST_DIR + 'GCP.csv', 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(GCP.keys())
                writer.writerow(GCP.values())
            
            MPI.COMM_WORLD.Barrier()
        else:
            STL_GEOREF = STL_BASENAME + '_geo'
            rotated_stl_basename = STL_BASENAME
            rotated_stl_basename_geo = STL_GEOREF
            x_length = mpi_comm.bcast(x_length if mpi_rank == 0 else None, root=0)
            y_length = mpi_comm.bcast(y_length if mpi_rank == 0 else None, root=0)


        # pyQvarsi.pprint(0, f'Domain size in x: {x_length} and y: {y_length}', flush=True)
        n = 0
        x_frames = 0
        y_frames = 0
        for i in range(0, 1):#-y_length, -overlap):
            # pyQvarsi.pprint(0, f"[Rank {mpi_rank}] Starting processing at wind angle {wind_angle}", flush=True)

            y_frames += 1
            for j in range(0, 1):#-x_length, -overlap):
                if i == 0:
                    x_frames += 1
                STL_DISPLACEMENT = [j-j_global, i-i_global, 0]
                pyQvarsi.pprint(0, STL_DISPLACEMENT, flush=True)
                # Measure plane generation
                t0 = perf_counter()
                int_mesh = plane_generation(STEP_SIZE, N_POINTS, N_POINTS)
                t1 = perf_counter()
                if mpi_rank == 0:
                    pyQvarsi.pprint(0, f"[Timing] plane_generation: {t1 - t0:.3f} seconds", flush=True)

                # Copy mesh points
                int_xyz = int_mesh.xyz.copy()
            
                pyQvarsi.pprint(0, 'STL DIR: ', POST_DIR, flush=True)
                pyQvarsi.pprint(0, 'POST DIR: ', POST_DIR, flush=True)
                pyQvarsi.pprint(0, f"Stl file to process: {POST_DIR + rotated_stl_basename + '.stl'}", flush=True)

                # Measure geometry processing
                t2 = perf_counter()
                if use_gpu:
                    output_fields = geometrical_magnitudes_gpu(
                        STL_FILE=POST_DIR + rotated_stl_basename + '.stl',
                        target_mesh=int_xyz,
                        stl_angle=STL_ROT_ANGLE,
                        stl_displ=STL_DISPLACEMENT,
                        stl_scale=STL_SCALE,
                        batch_size=args.batch_size,
                        grid_dims_Nx=N_POINTS,
                        grid_dims_Ny=N_POINTS,
                        dist_resolution=DIST_RESOLUTION,
                        z_tol=1e-1
                    )
                else:
                    output_fields = geometrical_magnitudes(
                        STL_FILE=POST_DIR + rotated_stl_basename + '.stl',
                        target_mesh=int_xyz,
                        stl_angle=STL_ROT_ANGLE,
                        stl_displ=STL_DISPLACEMENT,
                        stl_scale=STL_SCALE,
                        grid_dims_Nx=N_POINTS,
                        grid_dims_Ny=N_POINTS,
                        dist_resolution=DIST_RESOLUTION,
                        z_tol=1e-4
                    )
                t3 = perf_counter()
                if mpi_rank == 0:
                    pyQvarsi.pprint(0, f"[Timing] geometrical_magnitudes{'_gpu' if use_gpu else ''}: {t3 - t2:.3f} seconds", flush=True)

                # Save H5
                t4 = perf_counter()
                if pyQvarsi.utils.is_rank_or_serial(0):
                    int_mesh.save(POST_DIR + rotated_stl_basename + f'-{n}-geodata.h5', mpio=False)
                if pyQvarsi.utils.is_rank_or_serial(1):
                    output_fields.save(POST_DIR + rotated_stl_basename + f'-{n}-geodata.h5', mpio=False)
                t5 = perf_counter()
                if mpi_rank == 0:
                    pyQvarsi.pprint(0, f"[Timing] H5 save: {t5 - t4:.3f} seconds", flush=True)
                            # Append U and V features
                    append_UV_features(POST_DIR + rotated_stl_basename + f'-{n}')
                    pyQvarsi.pprint(0, f"U and V features added to {POST_DIR + rotated_stl_basename + f'-{n}-geodata.h5'}", flush=True)

                pyQvarsi.pprint(0, f"[Rank {mpi_rank}] Step {n} total time: {t5 - t0:.3f} seconds\n", flush=True)


                pyQvarsi.pprint(0, 'Done.', flush=True)
                pyQvarsi.cr_info()

                n += 1
                pyQvarsi.pprint(0, f"[Rank {mpi_rank}] Step {n} took {perf_counter() - start:.2f} seconds", flush=True)
        mpi_comm.Barrier()

        if mpi_rank == 0:
            with open(POST_DIR + 'global_vars.txt', 'w') as f:
                f.write(f"x_frames={x_frames}\n")
                f.write(f"y_frames={y_frames}\n")
        mpi_comm.Barrier()

