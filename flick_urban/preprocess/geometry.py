"""
flick_urban.preprocess.geometry — Geometric operations on STL meshes.

Provides rotation, translation, bounding-box calculation, wall-distance
computation, and plane-mesh generation used throughout the FLICK pipeline.

Example
-------
>>> from flick_urban.preprocess.geometry import calculate_bounding_box
>>> mn, mx = calculate_bounding_box('Examples/preprocess/data/grid_of_cubes.stl')
>>> print(mn, mx)   # [0. 0. 0.] [390. 390.  30.]
"""
from __future__ import print_function, division

import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

from stl import mesh
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pyQvarsi
import h5py
import trimesh

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def plane_generation(Length,nx,ny):

	# Generate partition table
	ptable = pyQvarsi.PartitionTable.new(1,nelems=(nx-1)*(ny-1),npoints=nx*ny)

	# Generate points
	points = np.array([
		[-Length,-Length,0.0],
		[ Length,-Length,0.0],
		[ Length, Length,0.0],
		[-Length, Length,0.0]
		],dtype='double')

	# Generate plane mesh
	return pyQvarsi.MeshAlya.plane(points[0],points[1],points[3],nx,ny,ngauss=1,ptable=ptable,compute_massMatrix=False)

def solid_perimeter_generation(triangles,dist_resolution) -> None:

        perimeter=[]
        for tri in triangles:
                v1=np.zeros(3)
                v2=np.zeros(3)
                if tri[0][2]!=0: 
                        v1=tri[1]
                        v2=tri[2]
                elif tri[1][2]!=0:
                        v1=tri[0]
                        v2=tri[2]
                elif tri[2][2]!=0:
                        v1=tri[0]
                        v2=tri[1]
                
                nsteps=int(np.linalg.norm(v2-v1)/dist_resolution)
                if nsteps>1: 
                        dLam=1.0/nsteps
                        Lam=0.0
                        perimeter.append(v1)
                        for i in range(nsteps):
                                Lam+=dLam
                                x=v1+Lam*(v2-v1)
                                perimeter.append(x)
                        perimeter.append(v2)
                else: 
                        perimeter.append(v1)
                        perimeter.append(v2)

        return np.array(perimeter)

def geometrical_data_extractor(target_mesh,horizontal_triangles,vertical_triangles,dist_resolution):

        h_triangles=np.copy(horizontal_triangles)
        h_triangles[:,:,2]=0
        
        perimeter=solid_perimeter_generation(vertical_triangles,dist_resolution)

        points=np.copy(target_mesh)
        points[:,2]=0.0

        size_G=points.shape[0]
        size_L=int(size_G/mpi_size)
        ini_idx=size_L*mpi_rank
        final_idx=size_L*(mpi_rank+1)-1
        if mpi_rank==(mpi_size-1): final_idx=size_G-1

        subset=points[ini_idx:final_idx+1]
        
        mask_L=np.zeros(subset.shape[0])
        height_L=np.zeros(subset.shape[0])
        distance_L=np.zeros(subset.shape[0])

        for idx,p in enumerate(subset):
                tri_idx=isIn(p,h_triangles)
                if tri_idx<0:
                        mask_L[idx]=1
                        height_L[idx]=0
                else:
                        mask_L[idx]=0
                        height_L[idx]=horizontal_triangles[tri_idx][0][2]

                if mask_L[idx]==1:
                        distance_L[idx]=wall_distance(p,perimeter)

        recv_buff_mask = mpi_comm.allgather(mask_L)
        recv_buff_height = mpi_comm.allgather(height_L)
        recv_buff_distance = mpi_comm.allgather(distance_L)
        
        mask_G=recv_buff_mask[0]
        height_G=recv_buff_height[0]
        distance_G=recv_buff_distance[0]
        for i in range(mpi_size-1):
                mask_G=np.concatenate((mask_G,recv_buff_mask[i+1]),axis=0)
                height_G=np.concatenate((height_G,recv_buff_height[i+1]),axis=0)
                distance_G=np.concatenate((distance_G,recv_buff_distance[i+1]),axis=0)

        fields = pyQvarsi.Field(xyz = points, ptable=pyQvarsi.PartitionTable.new(1,1,0))

        fields['MASK'] = mask_G
        fields['HEGT'] = height_G
        fields['WDST'] = distance_G

        return fields

def wall_distance(point,perimeter):

        point_vec=np.tile(point,(perimeter.shape[0],1))

        dist=np.linalg.norm(perimeter-point_vec,axis=1)
        return np.amin(dist)

def isIn(point,triangles):

        point_vec=np.tile(point,(triangles.shape[0],1))

        v0 =triangles[:,0,:] 
        v1 =triangles[:,1,:] 
        v2 =triangles[:,2,:] 

        S=0.5*np.linalg.norm(np.cross(v1-v0,v2-v0,axis=1),axis=1)
        S1=0.5*np.linalg.norm(np.cross(v0-point_vec,v1-point_vec,axis=1),axis=1)
        S2=0.5*np.linalg.norm(np.cross(v1-point_vec,v2-point_vec,axis=1),axis=1)
        S3=0.5*np.linalg.norm(np.cross(v2-point_vec,v0-point_vec,axis=1),axis=1)

        isIn=abs((S1+S2+S3)-S)<0.001
        
        output=np.where(isIn==True)[0]

        return output[0] if len(output) > 0 else -1

def display_scalarfield(plane):

        plt.imshow(plane, cmap='plasma')
        plt.show()

def save_scalarfield(plane,filename):

        plt.imsave(filename, plane)



def display_points(points):
        x=points[:,0]
        y=points[:,1]
        z=points[:,2]

        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x,y,z,s=1.0)
        plt.show()

def display_points_plane(points,plane):
        x=points[:,0]
        y=points[:,1]
        z=points[:,2]

        x1=plane[0].flatten()
        y1=plane[1].flatten()
        z1=np.zeros(len(x1))

        fig=plt.figure()
        ax = fig.add_subplot()
        ax.scatter(x,y,s=0.1)
        ax.scatter(x1,y1,s=1.0,c='#ff7f0e')
        plt.show()

def display_stl(mesh):
        # Create a new plot

        figure = plt.figure()
        axes = figure.add_subplot(projection='3d')
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))

        # Auto scale to the mesh size
        scale = mesh.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)
        # Show the plot to the screen
        plt.show()

def display_triangles(triangles):
        points=triangles.reshape(triangles.shape[0]*3,-1)
        display_points(points)


        
def rotate_stl(stl,angles,center=np.array([0, 0, 0])):
        
		alpha = math.pi*angles[2]/180.0
		beta  = math.pi*angles[1]/180.0
		gamma = math.pi*angles[0]/180.0
        
		R = np.ndarray(shape=(3,3))
        
		R[0][0] = math.cos(alpha)*math.cos(beta)
		R[1][0] = math.cos(alpha)*math.sin(beta)*math.sin(gamma)-math.sin(alpha)*math.cos(gamma)
		R[2][0] = math.cos(alpha)*math.sin(beta)*math.cos(gamma)+math.sin(alpha)*math.sin(gamma)
		R[0][1] = math.sin(alpha)*math.cos(beta)
		R[1][1] = math.sin(alpha)*math.sin(beta)*math.sin(gamma)+math.cos(alpha)*math.cos(gamma)
		R[2][1] = math.sin(alpha)*math.sin(beta)*math.cos(gamma)-math.cos(alpha)*math.sin(gamma)
		R[0][2] = -math.sin(beta)
		R[1][2] = math.cos(beta)*math.sin(gamma)
		R[2][2] = math.cos(beta)*math.cos(gamma)

		for tri, triangle in enumerate(stl):
			centers=np.tile(center,(3,1))
			stl[tri] = np.dot(triangle-centers,R)+centers
		return stl

def move_stl(stl,displacement=np.array([0, 0, 0])):
		for tri, triangle in enumerate(stl):
			displacements=np.tile(displacement,(3,1))
			stl[tri] = triangle+displacements
		
		return stl

def display_stl(mesh):
        # Create a new plot

        figure = plt.figure()
        axes = figure.add_subplot(projection='3d')
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))

        # Auto scale to the mesh size
        scale = mesh.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)
        # Show the plot to the screen
        plt.show()

def geometrical_magnitudes(STL_FILE,target_mesh,stl_angle=[0.0,0.0,0.0],stl_displ=[0.0,0.0,0.0],stl_scale=1.0,dist_resolution=1.0):
                
                my_mesh = mesh.Mesh(np.concatenate([m.data for m in mesh.Mesh.from_multi_file(STL_FILE)]))
                
                triangles = stl_scale*my_mesh.vectors
                triangles = rotate_stl(triangles,stl_angle,stl_displ)
                triangles = move_stl(triangles,stl_displ)

                horizontal_triangles=triangles[(triangles[:,0,2]==triangles[:,1,2]) & (triangles[:,0,2]==triangles[:,2,2]) & (triangles[:,0,2]!=0)]
                vertical_triangles=triangles[((triangles[:,0,2]==0) & (triangles[:,1,2]==0) & (triangles[:,2,2]!=0)) | \
                                        ((triangles[:,0,2]==0) & (triangles[:,1,2]!=0) & (triangles[:,2,2]==0)) | \
                                        ((triangles[:,0,2]!=0) & (triangles[:,1,2]==0) & (triangles[:,2,2]==0))]
                fields=geometrical_data_extractor(target_mesh,horizontal_triangles,vertical_triangles,dist_resolution)
                
                return fields

#Find the bounding box of the STL file
def calculate_bounding_box(input_file):
    """
    Calculate the bounding box of the vertices.
    """
        # Load the STL file
    try:
        stl_mesh = mesh.Mesh.from_file(input_file)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return
    
    # Extract all vertices from the mesh
    all_vertices = np.concatenate([stl_mesh.v0, stl_mesh.v1, stl_mesh.v2]).astype(np.float64)
    
    min_coords = np.min(all_vertices, axis=0).astype(np.float64)
    max_coords = np.max(all_vertices, axis=0).astype(np.float64)

    return min_coords, max_coords

def append_UV_features(file_path, N_POINTS=256):
    with h5py.File(f"{file_path}-geodata.h5", 'r+') as file:
        input_xdim = N_POINTS
        input_ydim = N_POINTS
        U = np.ones((input_xdim, input_ydim))

        # List of dataset names to overwrite
        datasets = [
            '/FIELD/VARIABLES/GRDUX',
            '/FIELD/VARIABLES/GRDVY',
            '/FIELD/VARIABLES/GRDWZ',
            '/FIELD/VARIABLES/U',
            '/FIELD/VARIABLES/V'
        ]

        for dset in datasets:
            # If dataset exists, delete it
            if dset in file:
                del file[dset]
            # Now safely create the dataset
            file.create_dataset(dset, data=U)

def move_stl_to_origin(stl_file, output_file):
    # Load the STL file
    original_mesh = mesh.Mesh.from_file(stl_file)

    # Get the minimum coordinates of the mesh
    min_bounds = np.min(original_mesh.vectors, axis=(0, 1))

    # Translate the mesh to the origin
    original_mesh.translate(-min_bounds)

    # Save the translated mesh to a new file
    original_mesh.save(output_file)
    # print(f'Saved translated STL file to: {output_file}')

def move_stl_to_origin_trimesh(input_file, output_file):
    # Load the STL mesh
    mesh = trimesh.load_mesh(input_file, force='mesh')

    # Check for valid mesh
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded object is not a Trimesh mesh.")

    # Translate mesh so its minimum bounding box corner is at the origin
    translation_vector = -mesh.bounds[0]
    mesh.apply_translation(translation_vector)

    # Export the mesh to a new file
    mesh.export(output_file)
    # print(f"Saved translated STL file to: {output_file}")

def compute_bounding_box_center(stl_mesh):
    """
    Compute the center of the bounding box for the given mesh.
    """
    min_bound = np.min(stl_mesh.vectors, axis=(0, 1))
    max_bound = np.max(stl_mesh.vectors, axis=(0, 1))
    center = (min_bound + max_bound) / 2
    return center

def rotation_matrix_around_z(theta_deg, convention="dataset"):
    """
    Return 3x3 rotation matrix.
    - convention="dataset": dataset angles where + is clockwise and 0 = up (+Y).
      Implemented by converting to math CCW angle internally.
    - convention="math": standard math angles (CCW, 0 = +X).
    """
    if convention == "dataset":
        # convert dataset angle (0=+Y, +CW) to math CCW angle
        # math_theta = 90 - theta_deg  is equivalent to negation + offset;
        # either is fine; here we use negation to treat +theta as clockwise.
        theta_rad = np.radians(theta_deg)
    else:
        theta_rad = np.radians(theta_deg)

    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return R


def rotate_stl(triangles, angle_xyz, displ=None):
    """Rotate a (N, 3, 3) triangle array around Z by angle_xyz[2] degrees.

    Used by GPU geometry extractors to orient STL triangles in memory before
    computing MASK/HEGT/WDST features.

    Parameters
    ----------
    triangles : np.ndarray, shape (N, 3, 3)
    angle_xyz : sequence of 3 floats — rotation angles [θx, θy, θz] in degrees.
                Only θz (index 2) is applied.
    displ     : unused, kept for signature compatibility.

    Returns
    -------
    np.ndarray, shape (N, 3, 3) — rotated triangles.
    """
    R = rotation_matrix_around_z(angle_xyz[2], convention="math")
    return np.einsum('ij,...j->...i', R, triangles)


def move_stl(triangles, displ):
    """Translate a (N, 3, 3) triangle array by a displacement vector.

    Parameters
    ----------
    triangles : np.ndarray, shape (N, 3, 3)
    displ     : sequence of 3 floats — [dx, dy, dz]

    Returns
    -------
    np.ndarray, shape (N, 3, 3) — translated triangles.
    """
    return triangles + np.array(displ, dtype=np.float64)


def rotate_geometry(stl_path, output_path, angle_deg, *,
                    center_world=None,
                    convention="dataset",
                    verbose=True,
                    rounding_decimals=8):
    """
    Rotate an STL by `angle_deg` around Z about `center_world` (cx,cy,cz).
    If center_world is None, compute the mesh bbox center and use that.
    Ensures the provided pivot remains at the exact coordinates (within FP tolerance).

    Returns diagnostics dict.
    """
    # load mesh
    m = mesh.Mesh.from_file(stl_path)

    # Flatten points and force float64 computation
    pts = m.vectors.reshape(-1, 3).astype(np.float64)

    # If no pivot provided, compute bbox center from the mesh points
    if center_world is None:
        min_coords = pts.min(axis=0)
        max_coords = pts.max(axis=0)
        cx, cy, cz = ((min_coords + max_coords) / 2.0).tolist()
        if verbose:
            print("Computed bbox center (used as pivot):", (cx, cy, cz))
    else:
        # ensure center_world is float64 and length 3
        cw = np.asarray(center_world, dtype=np.float64).ravel()
        if cw.size == 2:
            # if user provided (x,y), set z to mesh bbox center z
            min_z, max_z = pts[:, 2].min(), pts[:, 2].max()
            cz = (min_z + max_z) / 2.0
            cx, cy = cw[0], cw[1]
            # if verbose:
            #     print("Provided pivot (x,y) used; computed pivot z from mesh bbox:", cz)
        elif cw.size == 3:
            cx, cy, cz = cw.tolist()
        else:
            raise ValueError("center_world must be length 2 or 3 (x,y[,z])")
        # if verbose:
        #     print("Using provided center_world (pivot):", (cx, cy, cz))

    # ensure pivot as float64 array
    pivot = np.array([cx, cy, cz], dtype=np.float64)

    # rotation matrix (float64)
    R = rotation_matrix_around_z(angle_deg, convention=convention)

    # Translate points so pivot -> origin (float64)
    translated = pts - pivot  # float64

    # Apply rotation
    rotated = translated.dot(R.T)  # float64

    # Translate back
    rotated += pivot  # float64

    # OPTIONAL: round to avoid tiny floating point noise before casting to float32
    if rounding_decimals is not None:
        rotated_rounded = np.round(rotated, rounding_decimals)
    else:
        rotated_rounded = rotated

    # Write back into mesh with float32 storage
    m.vectors = rotated_rounded.reshape(m.vectors.shape).astype(np.float32)

    # Save rotated mesh
    out_file = output_path if str(output_path).endswith('.stl') else str(output_path) + '.stl'
    m.save(out_file)
    # if verbose:
    #     print(f"Saved rotated mesh to: {out_file}")

    # Diagnostics
    min_before = pts.min(axis=0); max_before = pts.max(axis=0)
    center_before = (min_before + max_before) / 2.0

    min_after = rotated.min(axis=0); max_after = rotated.max(axis=0)
    center_after = (min_after + max_after) / 2.0

    # Verify pivot is preserved (should be unchanged)
    # Because pivot is not necessarily a mesh vertex, check by re-applying transform:
    # rotating the pivot around itself should produce the same pivot (exactly).
    # Numeric check: compute transformed pivot by our steps and compare.
    pivot_translated = pivot - pivot  # zeros
    pivot_rotated = pivot_translated.dot(R.T) + pivot  # should equal pivot
    pivot_error = np.max(np.abs(pivot_rotated - pivot))

    # if verbose:
    #     print("Center before (bbox center) (x,y,z):", center_before)
    #     print("Center after  (bbox center) (x,y,z):", center_after)
    #     print("Chosen rotation pivot (world):", tuple(pivot.tolist()))
    #     print("Pivot reconstruction error (should be ~0):", pivot_error)

    # If pivot_error is tiny (<= ~1e-8) it's fine.
    # As an extra sanity check, compute centroid of geometry and see how far it moved relative to pivot:
    centroid_before = pts.mean(axis=0)
    centroid_after = rotated.mean(axis=0)
    centroid_shift = np.linalg.norm(centroid_after - centroid_before)


    return {
        "out_file": out_file,
        "center_before_bbox": center_before,
        "center_after_bbox": center_after,
        "chosen_pivot": tuple(pivot.tolist()),
        "pivot_error": float(pivot_error),
        "centroid_before": centroid_before,
        "centroid_after": centroid_after,
        "centroid_shift": float(centroid_shift),
        "R": R
    }
