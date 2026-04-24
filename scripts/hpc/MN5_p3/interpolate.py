#!/usr/bin/env python
import numpy as np
# import mpi4py
from mpi4py import MPI
import h5py

import os, copy, sys
import traceback

import argparse

# import bsp
from scipy.spatial import cKDTree


comm = MPI.COMM_WORLD
id_proc = comm.Get_rank()
n_proc = comm.Get_size()


valueGroups2exclude = ['time','istep']

# generate some random 3D unstructured CFD data
# n_points = 1000
# points = np.random.rand(n_points, 3)
# values = np.random.rand(n_points)

# basepath = "/gpfs/res_scratch/upc64/"
# src_mesh = basepath+'00_Grid/07_malla06_menorDZ/mesh07-8.h5'
# trg_mesh = basepath+'00_Grid/07_malla06_menorDZ/mesh07-128.h5'

mesh_groupGrid = "Coords"
meshHDF_groupGrid = "/VTKHDF/Points"

# src_field = basepath+'01_simulacions/202304-Cylinder_Malla07_cont3-implicit/results_mesh-8_51.h5'
# trg_field = './results_mesh-128_987654321.h5'

src_values_group_h5 = "/"
src_values_group_hdf = "/VTKHDF/PointData"
trg_values_group = "/"

def parserSetup():
    parser = argparse.ArgumentParser(description='Interpolate SOD2D results from one mesh to another. Creates a "restart h5" file as output.')
    parser.add_argument('src_mesh', type=str, help='Path to \'source mesh\' in hdf or h5 format.')
    parser.add_argument('trg_mesh', type=str, help='Path to \'target mesh\' in hdf format.')
    parser.add_argument('src_field', type=str, help='Path to \'results\' to read in hdf or h5 format.')
    parser.add_argument('trg_field', type=str, help='Full filename of output file. Typically, "restart_[MESHNAME]-[NUMBEROFPARTITIONS]_1.h5" ')
    parser.add_argument('--srcMeshLzSize', type=float, help="Meshes differ in 3D span size. Z-coord of the target mesh are taken the modulo with this value.")

    return parser

parser = parserSetup()

try:
    args = None
    if id_proc == 0:
#        print("Proc ",id_proc, "parsing")
       args = parser.parse_args()
finally:
    args = comm.bcast(args, root=0)

if args is None:
#     print("Proc ",id_proc, "leaving")
    exit(-1)


src_mesh = args.src_mesh
trg_mesh = args.trg_mesh
src_field = args.src_field
trg_field = args.trg_field

meshesDifferInZ = False

if args.srcMeshLzSize is not None:
    srcMeshLzSize = args.srcMeshLzSize
    meshesDifferInZ = True


# ----------------
# Check file formats
# ----------------
srcMeshFormatIsh5 = os.path.splitext(src_mesh)[1] == ".h5"

# Target meshes only in hdf format
assert(os.path.splitext(trg_mesh)[1] == ".hdf")

srcFieldFormatIsh5 = os.path.splitext(src_field)[1] == ".h5"

src_values_group = src_values_group_h5 if srcFieldFormatIsh5 else src_values_group_hdf


# ----------------
# Read source mesh
# ----------------

with h5py.File(src_mesh,"r") as src_h5g:
    if srcMeshFormatIsh5:
        coordinates_src = np.moveaxis( np.array([src_h5g[mesh_groupGrid+"/X"][()], src_h5g[mesh_groupGrid+"/Y"][()], src_h5g[mesh_groupGrid+"/Z"][()]]), 0, 1)
    else:
        coordinates_src = src_h5g[meshHDF_groupGrid][()]

# import code
# code.interact(local=locals())

if id_proc == 0:
    print("Source grid read")
    sys.stdout.flush()

# ----------------
# Create BSP
# ----------------
# mybsp = bsp.BSP(coordinates_src, values_src, max_depth=10, min_points=10)
# # mybsp.split() # Does not seem to work
mybspTree = cKDTree(coordinates_src)

if id_proc == 0:
    print("BSP created")
    sys.stdout.flush()


# ------------------------------------------------
# Read target mesh to find number of points
# ------------------------------------------------

with h5py.File(trg_mesh,"r") as trg_h5g:
    trg_npoint = trg_h5g[meshHDF_groupGrid][()].shape[0]

if id_proc == 0:
    print("Total number of points in target mesh: ", trg_npoint)
    sys.stdout.flush()

# ------------------------------------------------
# Split the workload into several processors
# ------------------------------------------------
nrow_global = trg_npoint
if (id_proc != n_proc-1) :
  nrow_proc = int(np.floor(nrow_global/n_proc) )
  inirow_proc = int(nrow_proc * id_proc )
  endrow_proc = int(nrow_proc + inirow_proc )
else:
  nrow_proc = int(np.floor(nrow_global/n_proc) )
  inirow_proc = int(nrow_proc * id_proc )
  nrow_proc = int(nrow_proc + (nrow_global-nrow_proc*n_proc) )
  endrow_proc = int(nrow_proc + inirow_proc  )

print("Rank ", id_proc, inirow_proc, endrow_proc, nrow_proc)
sys.stdout.flush()


if id_proc == 0:
    print("Work split done")
    sys.stdout.flush()

# ------------------------------------------------
# Interpolate and write to file (a file per rank)
# ------------------------------------------------

percent2sing = 1
percent2singincrement = copy.copy(percent2sing)

newOldFieldPairs = dict()
valuesdtype = None
with h5py.File(trg_mesh,"r") as trg_h5g, h5py.File(src_field,"r") as srcValuesh5:

    if n_proc == 1:
        ofh5 = h5py.File( trg_field, "w")
    else:
        ofh5 = h5py.File( trg_field, "w", driver='mpio', comm=comm )
    
    srch5_grp = srcValuesh5[src_values_group]
    ofh5_grp = ofh5.require_group(trg_values_group)

    for dsetName in srch5_grp:
        if dsetName in valueGroups2exclude:
            continue
        
        newDsetName = copy.copy(dsetName)
        if dsetName == "mu_e":
            newDsetName = "mue"
        elif dsetName == "mu_sgs": 
            newDsetName = "mut"

        if len(srcValuesh5[src_values_group+"/"+dsetName].shape) > 1 and newDsetName != "u":
            # Skip any field which has components, excepting u
            continue



        if newDsetName == "u":
            # For hdf input cases, as h5 input cases have already u as three separate entities
            # Restart needs the velocity as three separate vectors
           ofh5_grp.create_dataset("u_x", (trg_npoint,)  , dtype=srcValuesh5[src_values_group+"/u"].dtype)
           ofh5_grp.create_dataset("u_y", (trg_npoint,)  , dtype=srcValuesh5[src_values_group+"/u"].dtype)
           ofh5_grp.create_dataset("u_z", (trg_npoint,)  , dtype=srcValuesh5[src_values_group+"/u"].dtype)

           newOldFieldPairs["u_x"] = "u"
           newOldFieldPairs["u_y"] = "u"
           newOldFieldPairs["u_z"] = "u"

        else: 

           newOldFieldPairs[newDsetName] = dsetName

           ofh5_grp.create_dataset(newDsetName, (trg_npoint,)  , dtype=srcValuesh5[src_values_group+"/"+dsetName].dtype)
        

        if valuesdtype is None:
            valuesdtype = srcValuesh5[src_values_group+"/"+dsetName].dtype

#     coordinates_trg = np.moveaxis( np.array([trg_h5g_coords["X"][inirow_proc:endrow_proc], trg_h5g_coords["Y"][inirow_proc:endrow_proc], trg_h5g_coords["Z"][inirow_proc:endrow_proc]]), 0, 1)
    coordinates_trg = trg_h5g[meshHDF_groupGrid][inirow_proc:endrow_proc]

    if meshesDifferInZ:
        coordinates_trg[:,2] =  np.mod(coordinates_trg[:,2], srcMeshLzSize)

    print("Rank ", id_proc, " starting tree query" )
    sys.stdout.flush()
    dist, id_on_src = mybspTree.query(coordinates_trg)
    print("Rank ", id_proc, " finished tree query" )
    sys.stdout.flush()

#     import code
#     code.interact(local=locals())
        
    for dsetName in ofh5_grp:
#         try:
        
        olddsetname = newOldFieldPairs[dsetName]

        if not srcFieldFormatIsh5 and dsetName[:2] == "u_":
            if dsetName == "u_x":
                ofh5_grp[dsetName][inirow_proc:endrow_proc] = srch5_grp[olddsetname][()][id_on_src,0]
            elif dsetName == "u_y":
                ofh5_grp[dsetName][inirow_proc:endrow_proc] = srch5_grp[olddsetname][()][id_on_src,1]
            elif dsetName == "u_z":
                ofh5_grp[dsetName][inirow_proc:endrow_proc] = srch5_grp[olddsetname][()][id_on_src,2]
        else:

            ofh5_grp[dsetName][inirow_proc:endrow_proc] = srch5_grp[olddsetname][()][id_on_src]
#         except Exception as e:
#             import code
#             code.interact(local=locals())

#     ofh5_grp.create_dataset("time", data=srcValuesh5[src_values_group+"/time"])
    ofh5_grp.create_dataset("time", data=srcValuesh5["/time"])
    try:
        ofh5_grp.create_dataset("istep", data=srcValuesh5[src_values_group+"/istep"])
    except Exception as e:
        print("istep not found in source file. Setting it to zero.")
        ofh5_grp.create_dataset("istep", data=[0])


    if "mue" not in ofh5_grp:
        if id_proc == 0:
            print("mue not found in source field. Creating a zero array")
            sys.stdout.flush()
#         ofh5_grp.create_dataset("mue", data=np.zeros_like(srcValuesh5[src_values_group+"/rho"]))
        ofh5_grp.create_dataset("mue", data=np.zeros( (trg_npoint,), valuesdtype ) )

    ofh5.close()


if id_proc == 0:
    print("Interpolation done")
    sys.stdout.flush()


