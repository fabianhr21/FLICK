"""
gpu_geometry_opt.py — Optimised GPU geometry extractor (production).

Uses vectorised CuPy kernels and robust graph-based perimeter tracing to
compute MASK, HEGT, and WDST feature maps from STL triangle meshes.
Supersedes the legacy ``gpu_geometry.py``.
"""
import cupy as cp
import numpy as np
from mpi4py import MPI
import pyQvarsi
from stl import mesh
from flick_urban.preprocess.geometry import rotate_stl, move_stl
from scipy.spatial import cKDTree
from scipy.ndimage import label
from collections import defaultdict
from matplotlib.path import Path

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# --- Robust Perimeter Tracing ---
HASH_PRECISION = 8  # 8 decimal places for vertex hashing

def hash_vertex(v):
    """Creates a hashable (tuple) representation of a 2D vertex."""
    return (round(v[0], HASH_PRECISION), round(v[1], HASH_PRECISION))

def extract_and_trace_perimeters(vertical_triangles, z_tol):
    """
    Finds all 'ground edges' from vertical triangles, builds an adjacency graph,
    and traces it to find all continuous, ordered perimeter loops.
    
    Returns:
      A list of [ (N, 3) np.array ], where each array is an ordered
      list of vertices (X,Y,Z) forming a closed perimeter loop.
    """
    # if mpi_rank == 0:
    #     print(f"[Graph] Building perimeter graph from {len(vertical_triangles)} vertical triangles...")
        
    adj = defaultdict(list)
    vertex_map = {} # Map from hashable 2D tuple -> original 3D vertex
    
    # 1. Build Adjacency Graph from Ground Edges
    for tri in vertical_triangles:
        ground_verts = []
        for v in tri:
            if abs(v[2]) < z_tol:
                ground_verts.append(v)
        
        # We need exactly 2 ground vertices to form a ground edge
        if len(ground_verts) == 2:
            v1, v2 = ground_verts
            h1 = hash_vertex(v1)
            h2 = hash_vertex(v2)
            
            if h1 != h2:
                adj[h1].append(h2)
                adj[h2].append(h1)
                
                # Store the 3D vertex, preferring the one with z=0 if possible
                if h1 not in vertex_map or abs(v1[2]) < abs(vertex_map[h1][2]):
                    vertex_map[h1] = v1
                if h2 not in vertex_map or abs(v2[2]) < abs(vertex_map[h2][2]):
                    vertex_map[h2] = v2

    # 2. Trace Paths
    visited = set()
    all_perimeters = []

    for start_node in adj:
        if start_node not in visited:
            # Start tracing a new perimeter
            path = []
            current = start_node
            prev = None
            
            while current is not None:
                visited.add(current)
                if current not in vertex_map:
                    # if mpi_rank == 0:
                        # print(f"Warning: Node {current} in graph but not in vertex_map. Skipping path.")
                    path = [] # Invalidate path
                    break
                    
                path.append(vertex_map[current])
                
                next_node = None
                for neighbor in adj[current]:
                    if neighbor != prev:
                        next_node = neighbor
                        break 
                
                if next_node is None:
                    break
                elif next_node == start_node:
                    path.append(vertex_map[start_node]) 
                    break
                elif next_node in visited:
                    # if mpi_rank == 0:
                    #     print(f"Warning: Path hit visited node {next_node} before closing. Stopping trace.")
                    break
                
                prev = current
                current = next_node

            if len(path) > 2:
                all_perimeters.append(np.array(path, dtype=np.float64))

    # if mpi_rank == 0:
    #     print(f"[Graph] Tracing complete. Found {len(all_perimeters)} perimeter components.")
    return all_perimeters
# --- END: Robust Perimeter Tracing ---


# --- OPTIMIZED: Vectorized Utility Functions ---

def wall_distance_gpu_vectorized(points_gpu, perimeter_gpu):
    """
    Calculates distance to nearest perimeter point for a *batch* of points.
    
    points_gpu: (N, 3) array of query points
    perimeter_gpu: (P, 3) array of perimeter points
    Returns: (N,) array of minimum distances
    """
    # Use broadcasting to compute all N_local -> P distances at once
    # (N, 1, 3) - (1, P, 3) = (N, P, 3)
    points_b = points_gpu[:, None, :]
    perimeter_b = perimeter_gpu[None, :, :]
    
    # Calculate norms over the last axis (axis=2)
    dist_all = cp.linalg.norm(points_b - perimeter_b, axis=2) # Shape (N, P)
    
    # Find the minimum distance for each point (axis=1)
    min_dist = cp.min(dist_all, axis=1) # Shape (N,)
    return min_dist

def isIn_gpu_vectorized(points_gpu, triangles_gpu):
    """
    Checks if a *batch* of 2D points are inside any 2D triangles.
    
    points_gpu: (N, 3) array of query points (we use XY)
    triangles_gpu: (T, 3, 3) array of triangles (we use XY)
    Returns: (N,) array of triangle indices (-1 if no hit)
    """
    N = points_gpu.shape[0]
    T = triangles_gpu.shape[0]
    if N == 0:
        return cp.array([], dtype=cp.int32)
    if T == 0:
        return cp.full(N, -1, dtype=cp.int32)

    # Pre-calculate triangle areas (S) - shape (T,)
    v0 = triangles_gpu[:, 0, :2]
    v1 = triangles_gpu[:, 1, :2]
    v2 = triangles_gpu[:, 2, :2]
    # Use 2D cross product formula: 0.5 * |(x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)|
    # More stable: 0.5 * |x0(y1-y2) + x1(y2-y0) + x2(y0-y1)|
    S = 0.5 * cp.abs(v0[:,0]*(v1[:,1]-v2[:,1]) + v1[:,0]*(v2[:,1]-v0[:,1]) + v2[:,0]*(v0[:,1]-v1[:,1]))

    # Broadcast points and triangles
    points_b = points_gpu[:, None, :2] # Shape (N, 1, 2)
    v0_b = v0[None, :, :]              # Shape (1, T, 2)
    v1_b = v1[None, :, :]              # Shape (1, T, 2)
    v2_b = v2[None, :, :]              # Shape (1, T, 2)

    # Calculate areas of sub-triangles (S1, S2, S3) using broadcasting
    # S1 = 0.5 * |x_p(y0-y1) + x0(y1-y_p) + x1(y_p-y0)|
    p_x, p_y = points_b[:,:,0], points_b[:,:,1]
    v0_x, v0_y = v0_b[:,:,0], v0_b[:,:,1]
    v1_x, v1_y = v1_b[:,:,0], v1_b[:,:,1]
    v2_x, v2_y = v2_b[:,:,0], v2_b[:,:,1]

    S1 = 0.5 * cp.abs(p_x*(v0_y-v1_y) + v0_x*(v1_y-p_y) + v1_x*(p_y-v0_y)) # (N, T)
    S2 = 0.5 * cp.abs(p_x*(v1_y-v2_y) + v1_x*(v2_y-p_y) + v2_x*(p_y-v1_y)) # (N, T)
    S3 = 0.5 * cp.abs(p_x*(v2_y-v0_y) + v2_x*(v0_y-p_y) + v0_x*(p_y-v2_y)) # (N, T)
    
    # Check if (S1 + S2 + S3) is close to S
    # S_b is shape (1, T)
    S_b = S[None, :]
    isIn_all = cp.abs((S1 + S2 + S3) - S_b) < 0.001 # Shape (N, T)
    
    # Find the *first* triangle index that matches for each point
    # We can use argmax on the boolean array
    any_hit = cp.any(isIn_all, axis=1) # Shape (N,)
    first_hit_idx = cp.argmax(isIn_all, axis=1) # Shape (N,)
    
    # Only use the argmax index if there was actually a hit
    tri_idx = cp.where(any_hit, first_hit_idx, -1) # Shape (N,)
    
    return tri_idx

# --- Data Extractor ---
def geometrical_data_extractor_gpu(target_mesh, horizontal_triangles, 
                                   all_perimeter_vertices, 
                                   dist_resolution, batch_size, grid_dims_Nx=None, grid_dims_Ny=None):
    """
    Processes robust perimeter data (from tracing) and assigns orientations
    using vectorized GPU and CPU calls.
    """
    # GPU horizontal triangles (for isIn_gpu tests)
    h_triangles_gpu = cp.array(horizontal_triangles, dtype=cp.float32)
    h_triangles_gpu[:, :, 2] = 0.0 # Project to 2D

    # --- Build segment properties (CPU, but fast) ---
    all_sampled_points = []
    all_seg_midpoints = []
    all_seg_normals = []
    all_seg_angles = []
    all_seg_bins = []
    all_seg_lengths = [] 
    all_seg_component_ids = [] # To link segments to components
    perimeter_paths = [] 

    if len(all_perimeter_vertices) > 0:
        for component_id, perimeter_verts in enumerate(all_perimeter_vertices):
            M = len(perimeter_verts) - 1
            if M < 2:
                perimeter_paths.append(None) 
                continue
            
            perimeter_paths.append(Path(perimeter_verts[:, :2]))
            A = perimeter_verts[:-1, :2] # (M, 2)
            B = perimeter_verts[1:, :2]  # (M, 2)
            
            seg_vec = B - A
            seg_len = np.linalg.norm(seg_vec, axis=1, keepdims=True)
            seg_len_flat = seg_len.flatten()
            
            zero_mask = seg_len[:,0] < 1e-12
            seg_len[zero_mask, :] = 1.0 
            seg_dir = seg_vec / seg_len
            
            normals = np.column_stack((-seg_dir[:,1], seg_dir[:,0]))
            midpoints = (A + B) * 0.5
            
            centroid = A.mean(axis=0) 
            v_mid = midpoints - centroid
            dots = (normals * v_mid).sum(axis=1)
            normals[dots < 0] *= -1.0
            
            angles = (np.degrees(np.arctan2(-normals[:,1], -normals[:,0]))) % 360.0
            safe_angles = np.minimum(angles, 359.999)
            bins = (np.floor(safe_angles / 10.0).astype(np.int32)) % 36
            
            component_id_array = np.full(M, component_id, dtype=np.int32)
            
            # --- OPTIMIZED Perimeter Sampling (Unchanged) ---
            component_points_list = []
            for i in range(M):
                v1 = perimeter_verts[i]
                v2 = perimeter_verts[i+1]
                component_points_list.append(v1[None, :])
                seg_len_3d = np.linalg.norm(v2 - v1)
                nsteps = int(seg_len_3d / dist_resolution)
                if nsteps > 1:
                    lam = np.linspace(0.0, 1.0, num=nsteps + 1, dtype=np.float64)[1:-1]
                    points = v1[None, :] + lam[:, None] * (v2[None, :] - v1[None, :])
                    component_points_list.append(points)
            all_sampled_points.append(np.concatenate(component_points_list))
            # --- END Optimized Sampling ---

            all_seg_midpoints.append(midpoints)
            all_seg_normals.append(normals)
            all_seg_angles.append(angles)
            all_seg_bins.append(bins)
            all_seg_lengths.append(seg_len_flat) 
            all_seg_component_ids.append(component_id_array) 

    # --- Shannon Entropy Calculation (Unchanged) ---
    num_components = len(all_seg_bins)
    shannon_lookup_table = np.zeros(num_components, dtype=np.float64) 
    for i in range(num_components):
        component_bins = all_seg_bins[i]
        component_lengths = all_seg_lengths[i]
        total_length = np.sum(component_lengths)
        if total_length < 1e-9:
            shannon_lookup_table[i] = 0.0
            continue
        lengths_per_bin = np.zeros(36, dtype=np.float64)
        np.add.at(lengths_per_bin, component_bins, component_lengths)
        P_wf = lengths_per_bin / total_length
        P_wf_nonzero = P_wf[P_wf > 0]
        H0 = -np.sum(P_wf_nonzero * np.log(P_wf_nonzero))
        shannon_lookup_table[i] = H0
    # --- END ENTROPY CALCULATION ---

    # --- Concatenate all component data into single arrays (Unchanged) ---
    if len(all_sampled_points) > 0:
        perimeter_np = np.concatenate(all_sampled_points, dtype=np.float64)
        perimeter_gpu = cp.array(perimeter_np, dtype=cp.float32)
        all_midpoints_np = np.concatenate(all_seg_midpoints, dtype=np.float64)
        all_normals_np = np.concatenate(all_seg_normals, dtype=np.float64)
        all_seg_angles_np = np.concatenate(all_seg_angles, dtype=np.float64)
        all_seg_bins_np = np.concatenate(all_seg_bins, dtype=np.int32)
        all_seg_lengths_np = np.concatenate(all_seg_lengths, dtype=np.float64)
        all_seg_component_ids_np = np.concatenate(all_seg_component_ids, dtype=np.int32) 
    else:
        perimeter_np = np.empty((0,3), dtype=np.float64)
        perimeter_gpu = cp.empty((0,3), dtype=cp.float32)
        all_midpoints_np = np.empty((0,2), dtype=np.float64)
        all_normals_np = np.empty((0,2), dtype=np.float64) # Ensure 2D
        all_seg_angles_np = np.empty((0,), dtype=np.float64)
        all_seg_bins_np = np.empty((0,), dtype=np.int32)
        all_seg_lengths_np = np.empty((0,), dtype=np.float64)
        all_seg_component_ids_np = np.empty((0,), dtype=np.int32) 

    # --- Calculate Frontal Area per Component (Unchanged) ---
    avg_height_lookup_table = np.zeros(num_components, dtype=np.float64)
    if horizontal_triangles.shape[0] > 0 and num_components > 0:
        roof_centroids = np.mean(horizontal_triangles[:, :, :2], axis=1) # (T, 2)
        roof_heights = horizontal_triangles[:, 0, 2] # (T,)
        for i in range(num_components):
            path = perimeter_paths[i]
            if path is None: continue
            inside_mask = path.contains_points(roof_centroids)
            heights_inside = roof_heights[inside_mask]
            if heights_inside.shape[0] > 0:
                avg_height_lookup_table[i] = np.mean(heights_inside)
            
    frontal_length_lookup_table = np.zeros(num_components, dtype=np.float64)
    if all_seg_lengths_np.shape[0] > 0:
        projected_lengths = all_seg_lengths_np * np.maximum(0.0, -all_normals_np[:, 0])
        np.add.at(frontal_length_lookup_table, all_seg_component_ids_np, projected_lengths)
    
    frontal_area_lookup_table = frontal_length_lookup_table * avg_height_lookup_table
    # --- END FRONTAL AREA SECTION ---

    # --- <<< NEW: Build Frontal Perimeter Lookup >>> ---
    # This table maps a segment index to its ANGLE if it's frontal,
    # or to NaN if it's not.
    if all_normals_np.shape[0] > 0:
        # Mask for segments with normal_x < -tolerance (facing -X)
        is_frontal_seg_mask = (all_normals_np[:, 0] < -1e-6)
        # Create a lookup: has angle if frontal, NaN otherwise
        frontal_perimeter_lookup = np.where(is_frontal_seg_mask, all_seg_angles_np, np.nan)
    else:
        frontal_perimeter_lookup = np.empty((0,), dtype=np.float64)
    # --- <<< END NEW SECTION >>> ---

    # Build KD-tree on *true segment midpoints* (Unchanged)
    if all_midpoints_np.shape[0] > 0:
        seg_tree = cKDTree(all_midpoints_np)
    else:
        seg_tree = None

    # --- Project target mesh to plane and MPI-split (Unchanged) ---
    points = np.copy(target_mesh)
    points[:, 2] = 0.0
    size_G = points.shape[0]
    idx_splits = np.array_split(np.arange(size_G), mpi_size)
    subset_idx = idx_splits[mpi_rank]
    subset = points[subset_idx] if subset_idx.size > 0 else np.empty((0,3), dtype=points.dtype)

    N_local = subset.shape[0]
    if N_local == 0:
        # This rank has no work, create empty arrays for allgather
        mask_L = np.empty((0,), dtype=np.float64)
        height_L = np.empty((0,), dtype=np.float64)
        distance_L = np.empty((0,), dtype=np.float64)
        prm_bins_L = np.empty((0,), dtype=np.int32)
        prm_angles_L = np.empty((0,), dtype=np.float64)
        prm_seglen_L = np.empty((0,), dtype=np.float64)
        prm_shannon_L = np.empty((0,), dtype=np.float64)
        frontal_area_L = np.empty((0,), dtype=np.float64) # Renamed from prm_frontal_L
        prm_frontal_L = np.empty((0,), dtype=np.float64)  # <<< NEW
    else:
        # --- BATCHED VECTORIZED MAIN LOOP ---
        GPU_BATCH_SIZE = batch_size 
        subset_gpu = cp.array(subset, dtype=cp.float32)

        # Pre-allocate full-size result arrays on GPU
        tri_idx_L_gpu = cp.empty(N_local, dtype=cp.int32)
        mask_L_gpu = cp.empty(N_local, dtype=cp.float64)
        height_L_gpu = cp.empty(N_local, dtype=cp.float32)
        distance_L_gpu = cp.empty(N_local, dtype=cp.float32)

        if horizontal_triangles.shape[0] > 0:
            h_vals_gpu = cp.asarray(horizontal_triangles[:, 0, 2], dtype=cp.float32)
        
        for i in range(0, N_local, GPU_BATCH_SIZE):
            # ... (Unchanged batch processing for MASK, HEGT, WDST) ...
            batch_start = i
            batch_end = min(i + GPU_BATCH_SIZE, N_local)
            batch_subset_gpu = subset_gpu[batch_start:batch_end]
            # 1. MASK
            batch_tri_idx_gpu = isIn_gpu_vectorized(batch_subset_gpu, h_triangles_gpu)
            batch_mask_gpu = cp.where(batch_tri_idx_gpu < 0, 1.0, 0.0)
            tri_idx_L_gpu[batch_start:batch_end] = batch_tri_idx_gpu
            mask_L_gpu[batch_start:batch_end] = batch_mask_gpu
            # 2. HEGT
            batch_height_gpu = cp.zeros(batch_subset_gpu.shape[0], dtype=cp.float32)
            if horizontal_triangles.shape[0] > 0:
                safe_indices = cp.maximum(batch_tri_idx_gpu, 0)
                all_heights = h_vals_gpu[safe_indices]
                batch_height_gpu = cp.where(batch_mask_gpu == 0.0, all_heights, 0.0)
            height_L_gpu[batch_start:batch_end] = batch_height_gpu
            # 3. WDST
            batch_distance_gpu = cp.zeros(batch_subset_gpu.shape[0], dtype=cp.float32)
            if perimeter_gpu.size > 0:
                all_dists = wall_distance_gpu_vectorized(batch_subset_gpu, perimeter_gpu)
                batch_distance_gpu = cp.where(batch_mask_gpu > 0, all_dists, 0.0)
            distance_L_gpu[batch_start:batch_end] = batch_distance_gpu
        # --- END BATCHED LOOP ---
        
        # 4. CPU: Get Perimeter Angles/Bins (for inside points)
        prm_bins_L = np.full(N_local, -1, dtype=np.int32)
        prm_angles_L = np.full(N_local, np.nan, dtype=np.float64)
        prm_seglen_L = np.full(N_local, np.nan, dtype=np.float64)
        prm_shannon_L = np.full(N_local, np.nan, dtype=np.float64)
        frontal_area_L = np.full(N_local, np.nan, dtype=np.float64) # Renamed
        prm_frontal_L = np.full(N_local, np.nan, dtype=np.float64)  # <<< NEW
        
        if seg_tree is not None:
            dist_s_L, seg_idx_L = seg_tree.query(subset[:, :2], k=1)
            
            # Get properties for the *nearest segment*
            all_bins = all_seg_bins_np[seg_idx_L]
            all_angles = all_seg_angles_np[seg_idx_L]
            all_seg_lengths = all_seg_lengths_np[seg_idx_L]
            
            # Get component-wide properties
            component_ids_L = all_seg_component_ids_np[seg_idx_L]
            all_shannon_vals = shannon_lookup_table[component_ids_L]
            all_frontal_areas = frontal_area_lookup_table[component_ids_L]
            
            # <<< NEW: Get frontal perimeter property
            all_prm_frontal_vals = frontal_perimeter_lookup[seg_idx_L]
            
            # Transfer mask from GPU and apply
            mask_L_cpu = cp.asnumpy(mask_L_gpu)
            inside_mask_cpu = (mask_L_cpu == 0.0)
            
            prm_bins_L[inside_mask_cpu] = all_bins[inside_mask_cpu]
            prm_angles_L[inside_mask_cpu] = all_angles[inside_mask_cpu]
            prm_seglen_L[inside_mask_cpu] = all_seg_lengths[inside_mask_cpu]
            prm_shannon_L[inside_mask_cpu] = all_shannon_vals[inside_mask_cpu]
            frontal_area_L[inside_mask_cpu] = all_frontal_areas[inside_mask_cpu] # Renamed
            prm_frontal_L[inside_mask_cpu] = all_prm_frontal_vals[inside_mask_cpu] # <<< NEW

        # 5. Download final results from GPU
        mask_L = cp.asnumpy(mask_L_gpu)
        height_L = cp.asnumpy(height_L_gpu)
        distance_L = cp.asnumpy(distance_L_gpu)
        # --- END VECTORIZED LOOP ---

    # --- Reduce/gather across MPI ranks (MODIFIED) ---
    recv_mask = mpi_comm.allgather(mask_L)
    recv_height = mpi_comm.allgather(height_L)
    recv_buff_distance = mpi_comm.allgather(distance_L)
    recv_prm_bins = mpi_comm.allgather(prm_bins_L)
    recv_prm_angles = mpi_comm.allgather(prm_angles_L)
    recv_prm_seglen = mpi_comm.allgather(prm_seglen_L)
    recv_prm_shannon = mpi_comm.allgather(prm_shannon_L)
    recv_frontal_area = mpi_comm.allgather(frontal_area_L) # Renamed
    recv_prm_frontal = mpi_comm.allgather(prm_frontal_L) # <<< NEW

    mask_G = np.concatenate(recv_mask) if len(recv_mask) > 0 else np.empty((0,), dtype=np.float64)
    height_G = np.concatenate(recv_height) if len(recv_height) > 0 else np.empty((0,), dtype=np.float64)
    distance_G = np.concatenate(recv_buff_distance) if len(recv_buff_distance) > 0 else np.empty((0,), dtype=np.float64)
    prm_bins_G = np.concatenate(recv_prm_bins) if len(recv_prm_bins) > 0 else np.empty((0,), dtype=np.int32)
    prm_angles_G = np.concatenate(recv_prm_angles) if len(recv_prm_angles) > 0 else np.empty((0,), dtype=np.float64)
    prm_seglen_G = np.concatenate(recv_prm_seglen) if len(recv_prm_seglen) > 0 else np.empty((0,), dtype=np.float64)
    prm_shannon_G = np.concatenate(recv_prm_shannon) if len(recv_prm_shannon) > 0 else np.empty((0,), dtype=np.float64)
    frontal_area_G = np.concatenate(recv_frontal_area) if len(recv_frontal_area) > 0 else np.empty((0,), dtype=np.float64) # Renamed
    prm_frontal_G = np.concatenate(recv_prm_frontal) if len(recv_prm_frontal) > 0 else np.empty((0,), dtype=np.float64) # <<< NEW

    # --- Build the Field and attach arrays (MODIFIED) ---
    fields = pyQvarsi.Field(xyz=points, ptable=pyQvarsi.PartitionTable.new(1, 1, 0))
    fields['MASK'] = mask_G
    fields['HEGT'] = height_G
    fields['WDST'] = distance_G
    fields['PRMBIN'] = prm_bins_G
    fields['PRMANG'] = prm_angles_G
    fields['PRMSEGLEN'] = prm_seglen_G
    fields['PRMSHAN'] = prm_shannon_G
    fields['FRONTAL'] = frontal_area_G    # Renamed
    fields['PRMFRONTAL'] = prm_frontal_G  # <<< NEW

    # --- Alignment logic (Unchanged) ---
    if grid_dims_Nx is not None and grid_dims_Ny is not None:
        if mask_G.shape[0] != (grid_dims_Nx * grid_dims_Ny):
            if mpi_rank == 0:
                print(f"ADVERTENCIA: La forma de la malla ({mask_G.shape[0]}) no coincide con las dimensiones {grid_dims_Nx}*{grid_dims_Ny}")
            align_G = np.zeros_like(mask_G, dtype=np.int32)
        else:
            mask_2d = mask_G.reshape((grid_dims_Ny, grid_dims_Nx))
            align_2d = np.zeros_like(mask_2d, dtype=np.int32)
            for y in range(grid_dims_Ny):
                labeled_row, num_segments = label(mask_2d[y, :])
                align_2d[y, :] = labeled_row
            align_G = align_2d.flatten()
        fields['ALIGN'] = align_G
    else:
        fields['ALIGN'] = np.zeros_like(mask_G, dtype=np.int32)

    return fields

# --- CPU Fallback (Unchanged) ---
def wall_distance(point,perimeter):
        point_vec=np.tile(point,(perimeter.shape[0],1))
        dist=np.linalg.norm(perimeter-point_vec,axis=1)
        return np.amin(dist)
    
# --- Main Function (Unchanged from robust version) ---
def geometrical_magnitudes_gpu(STL_FILE, target_mesh, stl_angle=[0.0, 0.0, 0.0],
                                stl_displ=[0.0, 0.0, 0.0], stl_scale=1.0, batch_size=4192, grid_dims_Nx=None, grid_dims_Ny=None,
                                dist_resolution=1.0, z_tol=1e-2):
    
    # Load and process STL file (Only rank 0)
    if mpi_rank == 0:
        print(f"Loading STL: {STL_FILE}...")
        my_mesh = mesh.Mesh(np.concatenate([m.data for m in mesh.Mesh.from_multi_file(STL_FILE)]))

        triangles = stl_scale * my_mesh.vectors
        triangles = rotate_stl(triangles, stl_angle, stl_displ)
        triangles = move_stl(triangles, stl_displ)

        z0 = triangles[:, 0, 2]
        z1 = triangles[:, 1, 2]
        z2 = triangles[:, 2, 2]

        # Horizontal triangles
        horizontal_mask = (
            (np.abs(z0 - z1) < z_tol) &
            (np.abs(z0 - z2) < z_tol) &
            (np.abs(z0) > z_tol) # Not on the ground
        )
        horizontal_triangles = triangles[horizontal_mask]

        # Vertical triangles
        z = triangles[:, :, 2]
        z_near_zero = np.abs(z) < z_tol
        count_z_near_zero = np.sum(z_near_zero, axis=1)
        vertical_mask = (count_z_near_zero == 2)
        vertical_triangles = triangles[vertical_mask]

        # Robust Perimeter Extraction
        all_perimeter_vertices = extract_and_trace_perimeters(vertical_triangles, z_tol)
    else:
        # Other ranks get None
        horizontal_triangles = None
        all_perimeter_vertices = None
        dist_resolution = None

    # Broadcast geometry to all ranks
    horizontal_triangles = mpi_comm.bcast(horizontal_triangles, root=0)
    all_perimeter_vertices = mpi_comm.bcast(all_perimeter_vertices, root=0)
    dist_resolution = mpi_comm.bcast(dist_resolution, root=0)

    # All ranks run the extractor on their subset of target_mesh
    return geometrical_data_extractor_gpu(
        target_mesh,
        horizontal_triangles,
        all_perimeter_vertices, 
        dist_resolution,
        batch_size,
        grid_dims_Nx=grid_dims_Nx, grid_dims_Ny=grid_dims_Ny
    )