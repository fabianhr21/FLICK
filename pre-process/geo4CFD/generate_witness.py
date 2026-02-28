import trimesh
import numpy as np
import os
import argparse


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
    


def arg_parse():
    parser = argparse.ArgumentParser(description="Analyze building heights from STL file.")
    parser.add_argument('-stl', '--stl_file', type=str, required=True, help="Path to the STL file.")
    parser.add_argument('-p', '--path', type=str, required=True, help="Output path for domain_dimensions.txt.")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    analyze_buildings(args.stl_file, args.path)
    generate_witness_points(args.stl_file,args.path)
    print("Analysis complete. Results saved to domain_dimensions.txt and witness_points.txt.")