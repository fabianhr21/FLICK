from __future__ import print_function, division

import mpi4py 
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI
import csv
import os, re, glob, subprocess, numpy as np
from geometry_utils import geometrical_magnitudes,save_scalarfield,plane_generation,calculate_bounding_box,append_UV_features,move_stl_to_origin,rotate_geometry
import pyAlya
from stl import mesh

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

#### CHANGE HERE ####
# Folders & files
## Input a georeferenced STL file to obtain ground control points at the end
STL_DIR = '/gpfs/scratch/bsc21/bsc084826/WRF-NN/'
STL_GEOREF = 'output_SHIFT_simp'
STL_BASENAME = 'output_simp_origin'
POST_DIR = '/gpfs/scratch/bsc21/bsc084826/WRF-NN/output_UPC_rot/'

STL_SCALE=1.0
DIST_RESOLUTION=100000.0

# Parameters
WIND_DIRECTION = [45] # Rotates geometry to align with wind direction, if 0 no rotation is applied
STL_ROT_ANGLE=[0.0,0.0,0.0]
STL_DISPLACEMENT=[0.0,0.0,0.0]
STEP_SIZE=128  #this is L/2 where L is the side of the square
N_POINTS=256 #the point at the corner must be taken into account.
D_LENGTH= 300
p_overlap = 0.5
overlap = int(N_POINTS*p_overlap)
#####################

#Generate plane mesh

if __name__ == '__main__':
    for WIND_DIRECTION in WIND_DIRECTION:
        if WIND_DIRECTION != 0:
            rotate_geometry(STL_DIR+STL_GEOREF+'.stl', STL_DIR+STL_GEOREF+str(WIND_DIRECTION)+'.stl', 'z', WIND_DIRECTION)
            mpi_comm.Barrier() 
            STL_GEOREF = STL_GEOREF + str(WIND_DIRECTION)
            print(f'Rotated geometry to align with wind direction: {WIND_DIRECTION} in file {STL_GEOREF}.stl')
            mpi_comm.Barrier() 
        GCP = {}
        min_coords, max_coords = calculate_bounding_box(STL_DIR+STL_GEOREF+'.stl')
        min_x = min_coords[0]
        min_y = min_coords[1]
        max_x = max_coords[0]
        max_y = max_coords[1]
        # Create GCP dictionary
        GCP['min_x'] = min_x
        GCP['min_y'] = min_y
        GCP['max_x'] = max_x
        GCP['max_y'] = max_y
        
        # Save GCP dictionary TO A CSV FILE
        with open(POST_DIR+'GCP.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write the headers (keys)
            writer.writerow(GCP.keys())
            # Write the values
            writer.writerow(GCP.values())           
        # Generate a non-georeferenced STL file in the origin
        move_stl_to_origin(STL_DIR+STL_GEOREF+'.stl', STL_DIR+STL_BASENAME+'.stl')
        mpi_comm.Barrier() 
        
        n = 0
        for i in range(-overlap,-D_LENGTH,-overlap):
            for j in range(overlap,-D_LENGTH,-overlap):
                STL_DISPLACEMENT=[j,i,0]
                print(STL_DISPLACEMENT)
                STL_BASENAME = re.sub(r'\.stl$', '', STL_BASENAME)
                int_mesh = plane_generation(STEP_SIZE,N_POINTS,N_POINTS)
                pyAlya.pprint(0,'plane mesh Generated',flush=True)

                int_xyz = int_mesh.xyz.copy() # ensure we have a deep copy

                pyAlya.pprint(0,'STL DIR: ',STL_DIR,flush=True)
                pyAlya.pprint(0,'POST DIR: ',POST_DIR,flush=True)

                if mpi_rank == 0 and not os.path.exists(POST_DIR):
                    os.makedirs(POST_DIR)

                # Read source mesh
                pyAlya.pprint(0,"reading ",STL_DIR+STL_BASENAME+'.stl')
                output_fields = geometrical_magnitudes(STL_DIR+STL_BASENAME+'.stl',int_xyz,stl_angle=STL_ROT_ANGLE,stl_displ=STL_DISPLACEMENT,stl_scale=STL_SCALE,dist_resolution=DIST_RESOLUTION)

                # Save mesh

                if pyAlya.utils.is_rank_or_serial(0):
                    int_mesh.save(POST_DIR + STL_BASENAME+f'-{n}-geodata.h5',mpio=False)

                mpi_comm.Barrier()

                metadata={'STL name:': STL_BASENAME+'.stl'}
                if pyAlya.utils.is_rank_or_serial(1):
                    output_fields.save(POST_DIR + STL_BASENAME+f'-{n}-geodata.h5',mpio=False)#,metadata=metadata)
            

                pyAlya.pprint(0,'Done.',flush=True)
                pyAlya.cr_info()
                n += 1
        mpi_comm.Barrier()
        for i in range(0,n):
            append_UV_features(f"{POST_DIR}{STL_BASENAME}-{i}")
            