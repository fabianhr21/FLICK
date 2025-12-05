import ansa
from ansa import base, mesh, constants,session,dm
from ansa import *
import CreateDomain 
from GetVertical import separate_faces_by_vector
import orXYZ
import os 
import sys
import math

deck = constants.OPENFOAM
params = "/home/fabianh/ANSA/Data/scripts_final/MESH_PARAMETERS_MANDATORY.ansa_mpar"
working_directory = "/home/fabianh/ANSA/Data/cedval/cedval_const/"
# input_file = "cityBlocks"
input_file = "cedval"
target_path = "/home/fabianh/ANSA/Data/cedval/cedval_const/"
h_max = 150

def GroundCreate(x_min,x_max,y_min,y_max,z_min,z_max,h_max):
	deck = constants.NASTRAN
	
	 ## Create Morph box
	min_coords = [x_min-h_max,y_min-h_max,z_min]
	max_coords = [x_max+h_max ,y_max+h_max ,z_min]
	morph.MorphMinMax(None, min_coords, max_coords)
	m1 = base.CollectEntities(deck, None, "MORPHEDGE")
	
	# COnverto morph box to curve
	new_faces = morph.MorphConvert("MorphEdgesToCurve", m1, {"delete_original": True})
	m = base.CollectEntities(deck, None, "MORPHBOX")
	morph.MorphBoxDel(m)

	# GET ground faces
	faces = base.GetEntity(deck, "FACE", 6)
	ground = base.GetEntity(deck, "PSHELL", 6)
	search_type = ("FACE",)
	ground_faces = base.CollectEntities(deck, ground, search_type,recursive=True)
	before_ids   = {f._id for f in ground_faces}

	# COllect curves
	curves = base.CollectEntities(deck,None,"CURVE")

	# Cons project
	faces_project = base.ConsProjectNormal(curves, ground_faces, 0.0,connect_with_faces=True)
	
	# Delete curves
	curves = base.CollectEntities(deck,None,"CURVE")
	base.DeleteEntity(curves, True)
	
	# Get new faces
	ground = base.GetEntity(deck, "PSHELL", 6)
	search_type = ("FACE",)
	new_ground_faces = base.CollectEntities(deck, ground, search_type,recursive=True)
	after_ids   = {f._id for f in new_ground_faces}
	new_ids = after_ids - before_ids
	new_faces    = [base.GetEntity(deck, "FACE", fid) for fid in new_ids]
	
	faces_list = GetFacesInXYPlaneRegion([x_min,x_max,y_min,y_max,z_min,z_min])
	for face in faces_list:
		new_faces.append(face)
	
	# Create groundBuildings PID
	for face in new_faces:
		base.SetEntityCardValues(deck, face, {'PID':10})
	print(f"Created groundBuildings with PID 10.")

def GetFacesInXYPlaneRegion(box):
    deck = constants.NASTRAN
    """
    Return all FACE entities whose COG is inside the given box
    AND whose normal is (±)Z-aligned (i.e. face lies in the XY plane).

    :param box:       [x_min, x_max, y_min, y_max, z_min, z_max]
    :param deck_type: e.g. constants.OPENFOAM or constants.NASTRAN
    :return:          list of FACE entities
    """
    x_min, x_max, y_min, y_max, z_min, z_max = box

    # gather every face on the deck
    faces = base.CollectEntities(deck, None, "FACE")

    selected = []
    tol = 1e-3  # allow small numerical noise

    for face in faces:
        # 1) bounding-box test
        x, y, z = base.Cog(face)
        if not (x_min <= x <= x_max and
                y_min <= y <= y_max and
                z_min <= z <= z_max):
            continue

        # 2) orientation test: GetFaceOrientation → normalize
        vec = base.GetFaceOrientation(face)
        try:
            dx, dy, dz = vec
            mag = math.sqrt(dx*dx + dy*dy + dz*dz)
            nx, ny, nz = dx/mag, dy/mag, dz/mag
        except Exception:
            # if orientation failed, skip
            continue

        # keep only if |nx|,|ny| near zero and |nz| near 1
        if abs(nx) < tol and abs(ny) < tol and abs(abs(nz) - 1.0) < tol:
            selected.append(face)

    return selected

	

def separate_faces_by_vector(deck, x, y, z, angle, tol=0.1, pid=1,to_pid=11):
    """
    Selects all faces whose orientation is within (angle + tol) degrees
    of the reference vector (x, y, z), and applies base.Or() to them.
    
    Parameters
    ----------
    deck : int
        Solver deck identifier (e.g. ansa.constants.NASTRAN).
    x, y, z : float
        Components of the reference vector.
    angle : float
        Target angle in degrees.
    tol : float, optional
        Additional tolerance in degrees (default 0.1°).
    filter_visible : bool, optional
        Whether to only consider visible entities (default True).

    Returns
    -------
    list[Entity]
        The list of faces whose normals lie within (angle + tol) degrees
        of the reference vector.
    """
    # Normalize and compute max allowed angle
    ref_vec = (x, y, z)
    try:
        ref_u = calc.Normalize(ref_vec)
    except Exception:
        raise ValueError(f"Cannot normalize reference vector {ref_vec}")
    max_angle = angle + tol
    
    buildings = base.GetEntity(deck, "PSHELL", pid)
    base.Or(buildings)
    search_type = ("FACE",)
    faces = base.CollectEntities(deck, buildings, search_type,recursive=False,filter_visible=True)
    if not faces:
        print("No face elements exist in database")
        return []

    matched = []
    for face in faces:
        try:
            # Get face orientation and normalize
            vx, vy, vz = base.GetFaceOrientation(face)
            vec_u = calc.Normalize((vx, vy, vz))
            # Compute angle between vectors
            ang = math.degrees(calc.CalcAngleOfVectors(ref_u, vec_u))
            if ang <= max_angle:
                matched.append(face)
                base.SetEntityCardValues(deck, face, {"PID": to_pid})
        except TypeError:
            # skip faces without a valid orientation
            continue
        #base.SetEntityCardValues(deck, face, {"PID": to_pid})

    if matched:
        base.Or(matched)
    return matched

def main():
	# Input StereoLithography from City4CFD
    session.New("discard")
    mesh.ReadMeshParams(params)
    input = base.InputStereoLithography(
        working_directory + input_file + ".stl", elements_id="offset-freeid"
    )
	# Select working parts and recognize FM perimeters
    working_parts = base.CollectEntities(deck, None, "ANSAPART", filter_visible=True)
    fm = base.FeatureHandler(working_parts)
    fm.clear(False)
    fm.recognize(True)
    fe_perimeters = base.CollectEntities(deck, None, "FE PERIMETER")
    fe_perimeter_shells = mesh.GetFEPerimeterShells(fe_perimeters, expand_to_macro=True)
    
    # Obtain the height of the largest building
    ents = ("SHELL",)  # or "SOLID", "FACE", etc., depending on your model
    #shells = base.PickEntities(deck, ents,recursive=True,filter_visible=True)
    shells = base.CollectEntities(deck, None, "SHELL", filter_visible=True)
    nodes = base.CollectEntities(deck, shells, "NODE")
    z_values = [base.GetEntityCardValues(deck, node, ("Z",))["Z"] for node in nodes]
    x_values = [base.GetEntityCardValues(deck, node, ("X",))["X"] for node in nodes]
    y_values = [base.GetEntityCardValues(deck, node, ("Y",))["Y"] for node in nodes]
    x_min = min(x_values)
    x_max = max(x_values)
    x_length = x_max - x_min
    y_min = min(y_values)
    y_max = max(y_values)
    y_length = y_max - y_min
    z_min = min(z_values)
    z_max = max(z_values)
    height = z_max - z_min
    print("Max building height:", height)
    h_max = height

    # Creates domain and assign different PID to faces, core script modified (change in your directory)
    #xp,yp,zp,xn,yn,zn
    xp = 0.5*h_max
    yp = 0.5*h_max
    zp = 720
    xn = 0.5*h_max
    yn = 0.5*h_max
    zn = z_min
    x_length_domain = x_length + xp + xn
    y_length_domain = y_length + yp + yn
    z_length_domain = (z_max - z_min) + zp + (0.5*h_max)
    print("Domain lengths:", x_length_domain, y_length_domain, z_length_domain)
    CreateDomain._multibox(xp,yp,zp,xn,yn,zn,False)

	
	# Select working parts and recognize FM perimeters
    working_parts = base.CollectEntities(deck, None, "ANSAPART", filter_visible=True)
    fm = base.FeatureHandler(working_parts)
    fm.clear(False)
    fm.recognize(True)
    
    fe_perimeters = base.CollectEntities(deck, None, "FE PERIMETER")
    fe_perimeter_shells = mesh.GetFEPerimeterShells(fe_perimeters, expand_to_macro=False)
    
    # Separate PID
    base.PidToPart()
    


	# Create ABL	for sizing
    min_coords = [x_min - xn, y_min - yn, z_min]
    max_coords = [x_max + xp, y_max + yp, 48]    
    abl_box = morph.MorphMinMax(None, min_coords, max_coords)

    #Uses STL algorith to recover exact geometry
    #mesh.AspacingSTL('1%', 50, 0, 0.001)
    mesh.AspacingSTL("5%", 50.0, 0.0, 1)
    print("STL spacing\n")
    base.SetANSAdefaultsValues({'element_type':'quad'})
    mesh.CreateStlMesh()
    print("Buildings STL mesh generated\n")
    
    # Delete created mesh, keep geometry
    faces = base.CollectEntities(deck, None, 'FACE', recursive = True, filter_visible = True)
    mesh.ReleaseElements(faces)
    base.DeleteFaces(faces)
    print("Elements released from faces\n")
    
    # Describe the solid
    mesh.IntersectSolidDescription(0, fuse_distance = 0.1, improve_mesh_quality=False)
    print("Solid description of the buildings done\n")
    
    # Create surface geometry
    shells = base.CollectEntities(deck, None, 'SHELL', recursive = True)
    mesh.FEMToSurfArea(shells,delete = True, imprint = False)
    # base.DeleteEntity(shells, True)
    print("Buildings elements converted to faces\n")
    
    # # Quality Change
    options = ["CRACKS", "OVERLAPS", "NEEDLE FACES", "COLLAPSED CONS", "UNCHECKED FACES", "TRIPLE CONS"]
    fix = [1, 1, 1, 1, 1, 1]
    errors = base.CheckAndFixGeometry(0, options, fix, True, True)
    print(errors)
    if errors != None:
    	print('Total remaining errors: ', len(errors['failed']))
    	print('Type of remaining errors: ', len(errors['remaining_errors']))
    else:
    	print("Final geometry checked and fixed\n")
    	
    base.Topo()
    print("Topology created\n")
    
    # Convert Size Boxes to Size Field
    # size_field = mesh.ConvertSizeBoxesToSizeField(size_boxes=sbs)
    
    # Creates PID for later
    bot_out = base.CreateEntity(deck, "SHELL_PROPERTY", {"Name": "bot_out"})
    bot_south = base.CreateEntity(deck, "SHELL_PROPERTY", {"Name": "bot_south"})
    bot_in = base.CreateEntity(deck, "SHELL_PROPERTY", {"Name": "bot_in"})
    bot_north = base.CreateEntity(deck, "SHELL_PROPERTY", {"Name": "bot_north"})
    
    # Create groundBuildings
    # GroundCreate(x_min,x_max,y_min,y_max,z_min,z_max,h_max)
    
    # Simplify faces
    ret = mesh.SimplifyMacros(
        "ALL",
        fine_draft_slider=100,
        keep_perimeters_on_symmetry_plane=True,
        maintain_sharp_edges=True,
        minimum_side_length=3.5,
        minimum_perimeter_corner_angle=1,
        freeze_meshed_macros=False,
    )
    print(ret)
    
    # Separate Roofs from Walls
    # separate_faces_by_vector(constants.NASTRAN, 0, 0, 1, 30, tol=0.1, pid=1,to_pid=11)
    # # Delete north face
    north_face = base.GetEntity(deck, "SHELL_PROPERTY", 3)
    south_face = base.GetEntity(deck, "SHELL_PROPERTY", 5)
    base.DeleteEntity(north_face, True,compress=False)
    base.CreateEntity(deck, "SHELL_PROPERTY", {"PID": 3, "Name": "lateralDomainNorth"})
    print("North face deleted\n")

    # Create north face pid linked to south
    base.GeoTranslate("LINK",-2,"SAME PART","COPY",0,y_length_domain,0,south_face,keep_connectivity=True,draw_results=True)
    # Delete outlet face
    outlet_face = base.GetEntity(deck, "SHELL_PROPERTY", 7)
    base.DeleteEntity(outlet_face, True,compress=False)
    base.CreateEntity(deck, "SHELL_PROPERTY", {"PID": 7, "Name": "outlet"})
    print("Outlet face deleted\n")

    # Create outlet face pid linked to inlet
    inlet_face = base.GetEntity(deck, "SHELL_PROPERTY", 2)
    base.GeoTranslate("LINK",5,"SAME PART","COPY",x_length_domain,0,0,inlet_face,keep_connectivity=True,draw_results=True)
    
    base.Topo()
    base.Orient()
    # Save
    base.SaveAs(target_path+input_file+".ansa")
    print (input_file, "saved\n")    


if __name__ == '__main__':
	main()
