import trimesh
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def get_dominant_horizontal_direction(mesh):
    """
    Finds the dominant horizontal direction of the whole mesh bounding box
    by looking at the longest horizontal edge of the axis-aligned bounding box.
    Returns a unit vector in XY plane representing the longest axis.
    """
    bounds = mesh.bounds  # shape (2, 3): [min, max]
    extents = bounds[1] - bounds[0]  # [dx, dy, dz]

    # Only consider horizontal axes (X and Y)
    if extents[0] >= extents[1]:
        # X is the longest horizontal axis
        dominant = np.array([1.0, 0.0, 0.0])
    else:
        # Y is the longest horizontal axis
        dominant = np.array([0.0, 1.0, 0.0])

    return dominant


def align_mesh_to_wind(mesh):
    """
    Rotates the entire mesh around Z axis so that the longest horizontal
    axis of the bounding box aligns with the -X direction (wind inlet).
    Returns the rotated mesh and the angle used.
    """
    dominant = get_dominant_horizontal_direction(mesh)

    # Target direction: wind comes from -X, so longest axis should point along X
    target = np.array([1.0, 0.0, 0.0])

    # Calculate angle between dominant direction and target in XY plane
    angle = np.arctan2(dominant[1], dominant[0]) - np.arctan2(target[1], target[0])

    print(f"Dominant bounding box direction: {dominant}")
    print(f"Rotating mesh by {np.degrees(angle):.2f} degrees to align with wind (-X direction)")

    # Build rotation matrix around Z axis
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=-angle,  # negative to rotate dominant onto target
        direction=[0, 0, 1],
        point=mesh.centroid
    )

    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(rotation_matrix)

    return rotated_mesh, np.degrees(angle)


def analyze_buildings(stl_file_path, path):
    print(f"Loading {stl_file_path}...")

    # Load file
    full_mesh = trimesh.load(stl_file_path, force='mesh')

    # Align the whole mesh to wind direction before splitting
    # full_mesh, rotation_angle = align_mesh_to_wind(full_mesh)

    # # Save the rotated mesh next to the original
    # rotated_path = os.path.join(stl_file_path)
    # full_mesh.export(rotated_path)
    # print(f"Rotated mesh saved to: {rotated_path}")

    buildings = full_mesh.split(only_watertight=False)

    # print(f"--- Analysis Result ---")
    # print(f"Total separate structures found: {len(buildings)}")
    # print(f"Applied rotation: {rotation_angle:.2f} degrees")

    results = []
    height_smallest = 0
    for i, building in enumerate(buildings):
        min_coords = building.bounds[0]
        max_coords = building.bounds[1]

        height = max_coords[2] - min_coords[2]
        if height_smallest == 0 or height < height_smallest:
            height_smallest = height

        if height < 0.1:
            continue
        results.append({
            'id': i,
            'height': height,
            'bounds': building.bounds
        })

    maxh = max([r['height'] for r in results])
    avg_h = np.mean([r['height'] for r in results]).round(2)
    with open(f'{path}/domain_dimensions.txt', 'w') as f:
        f.write(f"max_h={max([r['height'] for r in results])}\n")
        f.write(f"top_hmax={maxh*20}\n")
        f.write(f"sides_hmax={maxh*40}\n")
        f.write(f"building_count={len(results)}\n")
        f.write(f"avg_h={np.mean([r['height'] for r in results]).round(2)}\n")
        f.write(f"top_avgh={avg_h*40}\n")
        f.write(f"sides_avgh={avg_h*65}\n")
        f.write(f"std_h={np.std([r['height'] for r in results]).round(2)}\n")
        f.write(f"var_h={np.var([r['height'] for r in results]).round(2)}\n")
        f.write(f"h_smallest={height_smallest}\n")
    return results

def generate_witness_points(stl_file_path, path):
    pedestrian_level = 1.5
    witness_total = 50
    witness_pedestrian = 10

    full_mesh = trimesh.load(stl_file_path, force='mesh')
    buildings = full_mesh.split(only_watertight=False)

    # Identify the ground plane
    ground_mesh = None
    building_meshes = []

    for component in buildings:
        face_normals = component.face_normals
        # A face is "horizontal ground" if its normal points mostly upward
        horizontal_mask = face_normals[:, 2] > 0.9
        horizontal_ratio = horizontal_mask.sum() / len(face_normals)

        # Heuristic: if most faces are horizontal and it's relatively flat,
        # treat it as ground; otherwise it's a building
        z_range = component.vertices[:, 2].max() - component.vertices[:, 2].min()
        if horizontal_ratio > 0.5 and z_range < 1.0:
            if ground_mesh is None or component.area > ground_mesh.area:
                ground_mesh = component
        else:
            building_meshes.append(component)

    # Combine all building meshes to define exclusion zones
    if building_meshes:
        combined_buildings = trimesh.util.concatenate(building_meshes)
    else:
        combined_buildings = None

    # --- Pedestrian-level witness points ---
    bounds = full_mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    x_min, y_min = bounds[0][0], bounds[0][1]
    x_max, y_max = bounds[1][0], bounds[1][1]

    pedestrian_points = []
    max_attempts = witness_pedestrian * 100

    while len(pedestrian_points) < witness_pedestrian:
        if len(pedestrian_points) >= witness_pedestrian:
            break
        # Random x, y within the bounding box
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = pedestrian_level

        x = round(x, 2)
        y = round(y, 2)

        point = np.array([x, y, z])

        # Reject if the point is inside any building
        if combined_buildings is not None:
            if combined_buildings.contains([point])[0]:
                continue

        pedestrian_points.append(point)
    
    # --- Additional witness points at various heights ---
    additional_points = []
    # for _ in range(witness_total - witness_pedestrian):
    while len(additional_points) < (witness_total - witness_pedestrian):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(bounds[0][2], bounds[1][2])  # Random height within the mesh bounds

        # Round to two decimals for cleaner output
        x = round(x, 2)
        y = round(y, 2)
        z = round(z, 2)

        point = np.array([x, y, z])

        # Reject if the point is inside any building
        if combined_buildings is not None:
            if combined_buildings.contains([point])[0]:
                continue

        additional_points.append(point)
    
    # Save witness points to file
    with open(f'{path}/witness.txt', 'w') as f:
        for pt in pedestrian_points:
            f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
        
        for pt in additional_points:
            f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
        
    return pedestrian_points, additional_points
    


def plot_witness_points(pedestrian_points, additional_points, path, stl_file_path):
    ped = np.array(pedestrian_points)
    add = np.array(additional_points) if additional_points else np.empty((0, 3))

    # Project mesh triangles onto each 2D plane for geometry overlay
    mesh = trimesh.load(stl_file_path, force='mesh')
    tris = mesh.triangles  # (N, 3, 3)
    tris_xy = tris[:, :, :2]       # drop Z → XY
    tris_yz = tris[:, :, 1:]       # drop X → YZ

    def add_geometry_xy(ax):
        col = PolyCollection(tris_xy, facecolor='lightgray', edgecolor='none', alpha=0.4, zorder=1)
        ax.add_collection(col)

    def add_geometry_yz(ax):
        col = PolyCollection(tris_yz, facecolor='lightgray', edgecolor='none', alpha=0.4, zorder=1)
        ax.add_collection(col)

    # --- Plot 1: XY plane at pedestrian level (1.5 m) ---
    fig, ax = plt.subplots(figsize=(8, 8))
    add_geometry_xy(ax)
    ax.scatter(ped[:, 0], ped[:, 1], c='tab:blue', s=40, zorder=3, label='Pedestrian (z=1.5 m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Witness points – XY plane at pedestrian level (z = 1.5 m)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.autoscale()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.savefig(f'{path}/debug_witness_xy_pedestrian.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 2: XY plane with all points ---
    fig, ax = plt.subplots(figsize=(8, 8))
    add_geometry_xy(ax)
    if len(add):
        ax.scatter(add[:, 0], add[:, 1], c='tab:orange', s=20, alpha=0.6, zorder=2, label='Additional')
    ax.scatter(ped[:, 0], ped[:, 1], c='tab:blue', s=40, zorder=3, label='Pedestrian (z=1.5 m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Witness points – XY plane (all heights)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.autoscale()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.savefig(f'{path}/debug_witness_xy_all.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 3: YZ plane (vertical distribution) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    add_geometry_yz(ax)
    if len(add):
        ax.scatter(add[:, 1], add[:, 2], c='tab:orange', s=20, alpha=0.6, zorder=2, label='Additional')
    ax.scatter(ped[:, 1], ped[:, 2], c='tab:blue', s=40, zorder=3, label='Pedestrian (z=1.5 m)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Witness points – YZ plane (vertical distribution)')
    ax.legend(loc='upper right')
    ax.autoscale()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.savefig(f'{path}/debug_witness_yz.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Debug plots saved to {path}/debug_witness_xy_pedestrian.png, "
          f"debug_witness_xy_all.png, debug_witness_yz.png")


def arg_parse():
    parser = argparse.ArgumentParser(description="Analyze building heights from STL file.")
    parser.add_argument('-stl', '--stl_file', type=str, required=True, help="Path to the STL file.")
    parser.add_argument('-p', '--path', type=str, required=True, help="Output path for domain_dimensions.txt.")
    parser.add_argument('-o', '--stl_output', type=str, required=True, help="Base output path; witness.txt and plots are saved to <stl_output>/MN5/.")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    output_dir = os.path.join(args.stl_output, 'MN5')
    os.makedirs(output_dir, exist_ok=True)
    analyze_buildings(args.stl_file, args.path)
    pedestrian_points, additional_points = generate_witness_points(args.stl_file, output_dir)
    plot_witness_points(pedestrian_points, additional_points, output_dir, args.stl_file)
    print("Analysis complete. Results saved to domain_dimensions.txt and witness_points.txt.")