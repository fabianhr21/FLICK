import trimesh
import numpy as np
import os
import argparse

def analyze_buildings(stl_file_path,path):
    print(f"Loading {stl_file_path}...")


    # Load file
    full_mesh = trimesh.load(stl_file_path, force='mesh')
    buildings = full_mesh.split(only_watertight=False)

    print(f"--- Analysis Result ---")
    print(f"Total separate structures found: {len(buildings)}")

    results = []

    for i, building in enumerate(buildings):
        # trimesh calculates the bounding box (axis aligned) automatically
        # bounds[0] is the min (x,y,z), bounds[1] is the max (x,y,z)
        min_coords = building.bounds[0]
        max_coords = building.bounds[1]
        
        # Calculate Height (Z-axis difference)
        height = max_coords[2] - min_coords[2]
        
        if height < 0.1: # Adjust threshold based on your unit scale
            continue
        results.append({
            'id': i,
            'height': height,
            'bounds': building.bounds
        })

    # Write results to domain_dimensions.txt
    with open(f'{path}/domain_dimensions.txt', 'w') as f:
        f.write(f"max_h={max([r['height'] for r in results])}\n")
        f.write(f"building_count={len(results)}\n")
        f.write(f"avg_h={np.mean([r['height'] for r in results]).round(2)}\n")
    return results

def arg_parse():
    parser = argparse.ArgumentParser(description="Analyze building heights from STL file.")
    parser.add_argument('-stl', '--stl_file', type=str, required=True, help="Path to the STL file.")
    parser.add_argument('-p', '--path', type=str, required=True, help="Output path for domain_dimensions.txt.")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    analyze_buildings(args.stl_file,args.path)
