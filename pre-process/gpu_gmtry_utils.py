# geometrical_gpu.py
import cupy as cp
import numpy as np
from mpi4py import MPI
import pyQvarsi
from stl import mesh
import numpy as np
from gmtry_utils import rotate_stl, move_stl


mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def wall_distance_gpu(point, perimeter_gpu):
    point_vec = cp.tile(cp.array(point,dtype=cp.float32), (perimeter_gpu.shape[0], 1))
    dist = cp.linalg.norm(perimeter_gpu - point_vec, axis=1)
    return cp.min(dist).item()

def isIn_gpu(point, triangles_gpu):
    point_vec = cp.tile(cp.array(point,dtype=cp.float32), (triangles_gpu.shape[0], 1))
    v0 = triangles_gpu[:, 0, :]
    v1 = triangles_gpu[:, 1, :]
    v2 = triangles_gpu[:, 2, :]

    S = 0.5 * cp.linalg.norm(cp.cross(v1 - v0, v2 - v0), axis=1)
    S1 = 0.5 * cp.linalg.norm(cp.cross(v0 - point_vec, v1 - point_vec), axis=1)
    S2 = 0.5 * cp.linalg.norm(cp.cross(v1 - point_vec, v2 - point_vec), axis=1)
    S3 = 0.5 * cp.linalg.norm(cp.cross(v2 - point_vec, v0 - point_vec), axis=1)

    isIn = cp.abs((S1 + S2 + S3) - S) < 0.001
    idx = cp.where(isIn)[0]
    return int(idx[0].item()) if len(idx) > 0 else -1

def geometrical_data_extractor_gpu(target_mesh, horizontal_triangles, vertical_triangles, dist_resolution):
    h_triangles_gpu = cp.array(horizontal_triangles, dtype=cp.float32)
    h_triangles_gpu[:, :, 2] = 0.0  # ya en GPU

    # Generar perímetro (en CPU)
    perimeter = []
    for tri in vertical_triangles:
        v1, v2 = np.zeros(3), np.zeros(3)
        if tri[0][2] != 0: v1, v2 = tri[1], tri[2]
        elif tri[1][2] != 0: v1, v2 = tri[0], tri[2]
        elif tri[2][2] != 0: v1, v2 = tri[0], tri[1]
        nsteps = int(np.linalg.norm(v2 - v1) / dist_resolution)
        if nsteps > 1:
            dLam = 1.0 / nsteps
            Lam = 0.0
            perimeter.append(v1)
            for _ in range(nsteps):
                Lam += dLam
                x = v1 + Lam * (v2 - v1)
                perimeter.append(x)
            perimeter.append(v2)
        else:
            perimeter.append(v1)
            perimeter.append(v2)
    perimeter_gpu = cp.array(perimeter,dtype=cp.float32)

    # Plano de puntos
    points = np.copy(target_mesh)
    points[:, 2] = 0.0

    size_G = points.shape[0]
    size_L = int(size_G / mpi_size)
    ini_idx = size_L * mpi_rank
    final_idx = size_L * (mpi_rank + 1) - 1
    if mpi_rank == (mpi_size - 1):
        final_idx = size_G - 1

    subset = points[ini_idx:final_idx + 1]
    mask_L = np.zeros(subset.shape[0])
    height_L = np.zeros(subset.shape[0])

    # h_triangles_gpu = cp.array(h_triangles,dtype=cp.float32)
    horizontal_triangles_gpu = cp.array(horizontal_triangles,dtype=cp.float32)

    for idx, p in enumerate(subset):
        tri_idx = isIn_gpu(p, h_triangles_gpu)
        if tri_idx < 0:
            mask_L[idx] = 1
            height_L[idx] = 0
        else:
            mask_L[idx] = 0
            height_L[idx] = horizontal_triangles[tri_idx][0][2]


    # Reducir entre procesos
    recv_mask = mpi_comm.allgather(mask_L)
    recv_height = mpi_comm.allgather(height_L)

    mask_G = np.concatenate(recv_mask)
    height_G = np.concatenate(recv_height)

    fields = pyQvarsi.Field(xyz=points, ptable=pyQvarsi.PartitionTable.new(1, 1, 0))
    fields['MASK'] = mask_G
    fields['HEGT'] = height_G
    return fields

def geometrical_magnitudes_gpu(STL_FILE, target_mesh, stl_angle=[0.0, 0.0, 0.0],
                                stl_displ=[0.0, 0.0, 0.0], stl_scale=1.0,
                                dist_resolution=1.0, z_tol=1e-2):
    # Load and process STL file
    my_mesh = mesh.Mesh(np.concatenate([m.data for m in mesh.Mesh.from_multi_file(STL_FILE)]))

    # Apply transformations
    triangles = stl_scale * my_mesh.vectors
    triangles = rotate_stl(triangles, stl_angle, stl_displ)
    triangles = move_stl(triangles, stl_displ)

    # Extract Z coordinates of each vertex
    z0 = triangles[:, 0, 2]
    z1 = triangles[:, 1, 2]
    z2 = triangles[:, 2, 2]

    # === Horizontal triangles: all 3 z values are approximately equal and not at z=0
    horizontal_mask = (
        (np.abs(z0 - z1) < z_tol) &
        (np.abs(z0 - z2) < z_tol) &
        (np.abs(z0) > z_tol)
    )
    horizontal_triangles = triangles[horizontal_mask]

    # === Vertical triangles: at least 2 vertices are near z=0, but not all 3
    z = triangles[:, :, 2]
    z_near_zero = np.abs(z) < z_tol
    count_z_near_zero = np.sum(z_near_zero, axis=1)
    vertical_mask = (count_z_near_zero >= 2) & (count_z_near_zero < 3)
    vertical_triangles = triangles[vertical_mask]

    # Run GPU extractor
    return geometrical_data_extractor_gpu(
        target_mesh,
        horizontal_triangles,
        vertical_triangles,
        dist_resolution
    )