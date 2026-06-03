"""
simpli_stl.py — STL mesh simplification via pymeshlab.

Use this script when an STL file has too many triangles for efficient
preprocessing. Applies quadric edge-collapse decimation to reduce the face
count to a target value.

Requires: pymeshlab  (``pip install pymeshlab``)
"""
import pymeshlab


def simplify_stl(input_filename, output_filename, target_faces):
    """Simplify an STL mesh to *target_faces* using quadric edge collapse.

    Parameters
    ----------
    input_filename  : str — path to the input STL file
    output_filename : str — path for the simplified output STL
    target_faces    : int — desired number of faces after decimation
    """
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_filename)
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=int(target_faces))
        ms.save_current_mesh(output_filename)
        print(f'Simplified mesh saved to {output_filename}')
    except Exception as e:
        print(f'An error occurred: {e}')


if __name__ == '__main__':
    # Example: simplify the reference geometry
    input_stl = 'grid_of_cubes.stl'
    output_stl = 'grid_of_cubes_simplified.stl'
    target_faces = 500000  # adjust based on desired detail level
    simplify_stl(input_stl, output_stl, target_faces)
