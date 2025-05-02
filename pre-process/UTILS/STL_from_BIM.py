import ifcopenshell
import ifcopenshell.geom
import re
import trimesh
import numpy as np
import os

def export_specific_ifcbuildingelementproxy_to_stl_with_z0(ifc_file_paths, output_stl_path, target_fraction=0.5):
    # Initialize the settings for the geometry processing
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    # Compile the regex pattern
    pattern = re.compile(r'Construcció:[0-9]{9}')

    # List to store all vertices and faces for creating a combined mesh
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    # Loop through all IFC files
    for ifc_file_path in ifc_file_paths:
        # Load the IFC file
        ifc_file = ifcopenshell.open(ifc_file_path)

        # Loop through all IfcBuildingElementProxy entities
        for proxy in ifc_file.by_type("IfcBuildingElementProxy"):
            try:
                # Check if the name matches the pattern
                if proxy.Name and pattern.match(proxy.Name):
                    # Get the geometry of the proxy
                    shape = ifcopenshell.geom.create_shape(settings, proxy)
                    geometry = shape.geometry

                    # Extract the vertices and faces from the geometry
                    vertices = np.array(geometry.verts)  # Convert to numpy array
                    faces = np.array(geometry.faces)

                    if vertices.size == 0:
                        continue  # Skip if no vertices are found

                    # Find the minimum z value to adjust vertices to z=0
                    min_z = np.min(vertices[2::3])  # Get the minimum z value from the vertices

                    # Adjust vertices to set the geometry on a z=0 plane
                    vertices[2::3] -= min_z

                    # Append the vertices and faces to the combined lists
                    all_vertices.append(vertices.reshape(-1, 3))
                    all_faces.append(faces.reshape(-1, 3) + vertex_offset)
                    vertex_offset += vertices.size // 3
            except Exception as e:
                print(f"Failed to process proxy {proxy.GlobalId} in file {os.path.basename(ifc_file_path)}: {e}")
                continue

    # Combine all vertices and faces into a single mesh
    if all_vertices:
        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)
        mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)

        # Simplify the mesh
        simplified_mesh = mesh.simplify_quadratic_decimation(int(len(mesh.faces) * target_fraction))

        # Export the simplified mesh to STL
        simplified_mesh.export(output_stl_path)
    else:
        print("No matching IfcBuildingElementProxy entities found in the provided files.")

# Usage example
ifc_file_paths = ["/content/left.ifc", "/content/right.ifc"]
output_stl_path = '/content/combined_output_UPC.stl'
export_specific_ifcbuildingelementproxy_to_stl_with_z0(ifc_file_paths, output_stl_path, target_fraction=0.5)