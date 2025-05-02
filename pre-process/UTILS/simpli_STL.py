# IF YOUR MESH HAS TO MANY TRAIANGLES, USE THIS SCRIPT TO SIMPLIFY IT
import pymeshlab

def simplify_stl(input_filename, output_filename, target_faces):
    try:
        # Create a new MeshSet
        ms = pymeshlab.MeshSet()

        # Load the mesh from the input STL file
        ms.load_new_mesh(input_filename)

        # Simplify the mesh to the target number of faces
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=int(target_faces))

        # Save the simplified mesh to the output STL file
        ms.save_current_mesh(output_filename)

        print(f'Simplified mesh saved to {output_filename}')
    except Exception as e:
        print(f'An error occurred: {e}')

# Example usage
input_stl = '24-07-05_San_Jeronimo_domain_buildings_and_vegetaion_only.stl'
output_stl = 'simplified_SanJeronimo.stl'
target_faces = 1e8  # Adjust this value based on the desired level of simplification

simplify_stl(input_stl, output_stl, target_faces)
