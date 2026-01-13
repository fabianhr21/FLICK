import ansa
from ansa import base, mesh, constants,session,dm
from ansa import *
import CreateDomain 
from GetVertical import separate_faces_by_vector
# import orXYZ
import os 
import sys
import math
import argparse
import sys

deck = constants.OPENFOAM
params = "/home/fabianh/ANSA/Data/scripts_final/MESH_PARAMETERS_MANDATORY.ansa_mpar"
working_directory = "/home/fabianh/GEO_CASES/BARCELONA/"
input_file = "bcn_Buildings"
target_path = "/home/fabianh/GEO_CASES/BARCELONA/"
h_max = 150

def generate_geo(geo_script, input_file,y_length):
    # Read template
    with open(geo_script, "r") as f:
        geo = f.read()

    # Replace parameter
    geo = geo.replace("{{y_length}}", str(y_length))
    geo = geo.replace("{{input_file}}", str(input_file))

    # Save new file
    with open(geo_script, "w") as f:
        f.write(geo)

    print(f"Generated: {geo_script}")

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
	
	# Create new identity groundBuildings
	# base.CreateEntity(deck, "SHELL_PROPERTY", {"Name": "groundBuildings","PID": 10})

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
    
    buildings = base.GetEntity(constants.NASTRAN, "PSHELL", pid)
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

    #if matched:
    #    base.Or(matched)
    return matched

def get_args(input_file, working_directory, target_path):
    input_file = input_file
    working_directory = working_directory
    target_path = target_path
    print(f"Input file: {input_file}")
    print(f"Working directory: {working_directory}")
    print(f"Target path: {target_path}")
    return input_file, working_directory, target_path

def orXYZ(xchecked, ychecked, zchecked, xstatus, ystatus, zstatus, xval, yval, zval):
	base.BlockRedraws(True)
	deck = constants.OPENFOAM
	base.SetCurrentMenu("MESH")

	entities = []
	type = ["SHELL", "SOLID"]
	faces = base.CollectEntities(deck, None, "FACE", filter_visible = True)
	shells = base.CollectEntities(deck, None, type, filter_visible = True)
	for face in faces:
		val = base.GetEntityCardValues(deck, face, ("Meshed With", ))
		if(val["Meshed With"] == "UNMESHED"):	
			entities.append(face)
	for shell in shells:
		entities.append(shell)
	if not entities:
		print("There are no visible Shells,Solids or Faces")
		print("Script execution stopped")
		base.BlockRedraws(False)		
		return True

#	print(str(xchecked)+" , "+str(ychecked)+","+str(zchecked)+", "+xstatus+", "+ystatus+", "+zstatus+", "+str(xval)+", "+str(yval)+", "+str(zval))
	xcollected = []
	ycollected = []
	zcollected = []
#If X is checked then the code below will be executed
	if(xchecked == 1):
		for ent in entities:
			(x, y, z) = base.Cog(ent)
			ret = _main_core(ent, xstatus, xval, x)
			if ret is not None:
				xcollected.append(ret)
	base.Not(xcollected)
	
#If Y is checked then the code below will be executed
	if(ychecked == 1):
		for ent in entities:
			(x, y, z) = base.Cog(ent)
			ret = _main_core(ent, ystatus, yval, y)
			if ret is not None:
				ycollected.append(ret)
	base.Not(ycollected)
	
#If Z is checked then the code below will be executed
	if(zchecked == 1):
		for ent in entities:
			(x, y, z) = base.Cog(ent)
			ret = _main_core(ent, zstatus, zval, z)
			if ret is not None:
				zcollected.append(ret)
	base.Not(zcollected)    
	base.BlockRedraws(False)	
	
def _main_core(iso_ent, status, user_val, cog_val):
#We perform exactly the opposite action
#Since we work on visible entities and we need to isolate first in X then in Y and at the end in Z
#We cannot run the OR function. For this reason we run NOT, but on the exactly opposite directions!
	if(status == "Less than"):
		if(cog_val > user_val):
			collected = iso_ent
			return collected
		else:
			return None
	else:
		if(cog_val < user_val):
			collected = iso_ent
			return collected
		else:
			return None

def main(input_file_dir, working_directory, target_path):

    # Take the last part of the path as input file name without extension
    input_file = os.path.splitext(os.path.basename(input_file_dir))[0]

    print(f"Input file: {input_file}")
    print(f"Working directory: {working_directory}")
    print(f"Target path: {target_path}")
	# Input StereoLithography from City4CFD
    session.New("discard")
    mesh.ReadMeshParams(params)
    input = base.InputStereoLithography(
        working_directory + input_file + ".stl" , elements_id="offset-freeid"
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
    y_min = min(y_values)
    y_max = max(y_values)
    z_min = min(z_values)
    z_max = max(z_values)
    x_length = x_max - x_min
    y_length = y_max - y_min
    z_length = z_max - z_min

    height = z_max - z_min
    print("Max building height:", height)
    h_max = height

    # Creates domain and assign different PID to faces, core script modified (change in your directory)
    #xp,yp,zp,xn,yn,zn
    xp =40*h_max
    yp = 30*h_max
    zp = 20*h_max
    xn = 20*h_max
    yn = 30*h_max
    zn = z_min
    x_length_domain = x_length + xp + xn
    y_length_domain = y_length + yp + yn
    z_length_domain = z_length + zp
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
    
    # Create Size Boxes
    ## Buildings size box
    buildings = base.GetEntity(deck, "PSHELL", 1)
    search_type = ("SHELL",)
    buildings_shells = base.CollectEntities(deck, buildings, search_type,recursive=True)
    arg2 = []
    arg2.append([1.0, 0.0, 0.0, ])
    arg2.append([0.0, 1.0, 0.0, ])
    buildings_sb = ansa.base.SizeBoxOrtho(buildings_shells, directions=arg2,  max_length_surface=10,max_length_volume=16)
    ## Campus
    min_coords = [x_min+(5*h_max),y_min+ (5*h_max),z_min]
    max_coords = [x_max-(5*h_max),y_max -(5*h_max),z_max]
    campus_sb = base.SizeBoxMinMax(None, min_coords, max_coords, 6, 10)   
    ## ABL
    min_coords = [x_min - (15*h_max), y_min,z_min]
    max_coords = [x_max,y_max,z_max]
    print(min_coords, max_coords)
    abl_sb = base.SizeBoxMinMax(None, min_coords, max_coords, 30, 40)
    ## Close ground
    min_coords = [x_min-(20*h_max),y_min- (30*h_max),z_min]
    max_coords = [x_max+(40*h_max),y_max +(30*h_max),z_max+h_max]
    close_ground_sb = base.SizeBoxMinMax(None, min_coords, max_coords, 85, 85)
    ## Wake 1 (10h)
    min_coords = [x_max - h_max,y_min,z_min]
    max_coords = [(x_max - h_max) + (10*h_max),y_max ,z_max]
    wake1_sb = base.SizeBoxMinMax(None, min_coords, max_coords, 40, 60)   
    ## Wake 2 (20h)
    min_coords = [(x_max - h_max)+ (9*h_max+20),y_min ,z_min]
    max_coords = [(x_max - h_max) +(29*h_max),y_max,z_max]
    wake2_sb = base.SizeBoxMinMax(None, min_coords, max_coords, 60, 80)
    
    # Save Size Boxes to a list
    sbs = [buildings_sb, campus_sb, abl_sb, close_ground_sb, wake1_sb, wake2_sb]
    
    #Uses STL algorith to recover exact geometry
    #mesh.AspacingSTL('1%', 50, 0, 0.001)
    mesh.AspacingSTL("5%", 50.0, 30.0, 0.2)
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
    mesh.IntersectSolidDescription(0, fuse_distance = 0.5, improve_mesh_quality=False)
    print("Solid description of the buildings done\n")
    
    # Create surface geometry
    shells = base.CollectEntities(deck, None, 'SHELL', recursive = True)
    mesh.FEMToSurfArea(shells, delete = True, imprint = False)
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
    size_field = mesh.ConvertSizeBoxesToSizeField(size_boxes=sbs)
    
    # Creates PID for later
    topPrecursor = base.CreateEntity(deck, "SHELL_PROPERTY", {"Name": "topPrecursor"})
    groundPrecursor = base.CreateEntity(deck, "SHELL_PROPERTY", {"Name": "groundPrecursor"})
    
    # Create groundBuildings
    GroundCreate(x_min,x_max,y_min,y_max,z_min,z_max,h_max)
    
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
    separate_faces_by_vector(constants.NASTRAN, 0, 0, 1, 30, tol=0.1, pid=1,to_pid=11)

    # Creates PID for later
    bot_out = base.CreateEntity(deck, "SHELL_PROPERTY", {"Name": "bot_out"})
    bot_south = base.CreateEntity(deck, "SHELL_PROPERTY", {"Name": "bot_south"})
    bot_in = base.CreateEntity(deck, "SHELL_PROPERTY", {"Name": "bot_in"})
    bot_north = base.CreateEntity(deck, "SHELL_PROPERTY", {"Name": "bot_north"})
    
    # Ensure link between laterls-pair and inlet-oulet pair
    # # Delete north face
    north_face = base.GetEntity(deck, "SHELL_PROPERTY", 3)
    south_face = base.GetEntity(deck, "SHELL_PROPERTY", 5)
    base.DeleteEntity(north_face, True,compress=False)
    base.CreateEntity(deck, "SHELL_PROPERTY", {"PID": 3, "Name": "lateralDomainNorth"})
    print("North face deleted\n")

    # Create north face pid linked to south
    base.GeoTranslate("LINK",-2,"SAME PART","COPY",0,y_length_domain,0,south_face,keep_connectivity=True,draw_results=True)
    print("North-South faces linked\n")
    
    base.Topo(cons='all')
    props = base.CollectEntities(deck, None, "PSHELL", False)
    base.AutoCalculateOrientation("Visible", False)
    
    base.All()
    # Project abl size box to sides
    ## Close ground
    min_coords = [x_min-(20*h_max),y_min- (30*h_max),z_min]
    max_coords = [x_max+(40*h_max),y_max +(30*h_max),48]
    # Create Morph box for project
    morph.MorphMinMax(None, min_coords, max_coords)
    m1 = base.CollectEntities(deck, None, "MORPHEDGE")
    # COnverto morph box to curve
    morph_perimeters = morph.MorphConvert("MorphEdgesToCurve", m1, {"delete_original": True})
    m = base.CollectEntities(deck, None, "MORPHBOX")
    morph.MorphBoxDel(m)
    # Get faces to project
    in_face = base.GetEntity(constants.NASTRAN, "PSHELL", 2)
    out_face = base.GetEntity(constants.NASTRAN, "PSHELL", 7)
    north_south = base.GetEntity(constants.NASTRAN, "PSHELL", 3)
    search_type = ("FACE",)
    in_out_faces = base.CollectEntities(constants.NASTRAN, [in_face,out_face,north_south], search_type,recursive=False)
    # Collect curves
    curves = base.CollectEntities(deck,None,"CURVE")
    # Cons project
    arg3 = {}
    arg3['Normal'] = (0.0, 0.0, 0.0, )
    cons = base.ConsProject(entities=morph_perimeters, faces_array=in_out_faces, project_type=arg3, min_length=20.0, split_original=True, paste_sides=True, paste=True)
    base.Or(cons[0])

    # Save domain dimensions
    domain_file = open(target_path+"domain_dimensions.txt", "w")
    domain_file.write(f"x_min: {x_min - xn}\n")
    domain_file.write(f"x_max: {x_max + xp}\n")
    domain_file.write(f"y_min: {y_min - yn}\n")
    domain_file.write(f"y_max: {y_max + yp}\n")
    domain_file.write(f"z_min: {z_min - 0.5*h_max}\n")
    domain_file.write(f"z_max: {z_max + zp}\n")
    domain_file.write(f"x_length: {x_length_domain}\n")
    domain_file.write(f"y_length: {y_length_domain}\n")
    domain_file.write(f"z_length: {z_length_domain}\n")
    domain_file.write(f"Number in x: {x_length_domain/12}\n")
    domain_file.write(f"Number in y: {y_length_domain/12}\n")

    domain_file.close()

    bottom_ents = []
    base.All()
    # Select bottom inlet
    orXYZ(1,0,1,"Less than","Less than","Less than",-xn+10,1,50)
    ent = base.CollectEntities(deck, None,"FACE", filter_visible=True)
    base.SetEntityCardValues(deck, ent[0], {'PID':14})
    cons = base.CollectEntities(deck, None,"CONS", filter_visible=True)
    mesh.NumberPerimeters([cons[0],cons[2]],f"{y_length_domain//12}")
    mesh.NumberPerimeters([cons[1],cons[3]], "4")
    bottom_ents.append(ent[0])
    
    base.All()
    # Select bottom outlet
    orXYZ(1,0,1,"Greater than","Greater than","Less than",xn-10,1,50)
    ent = base.CollectEntities(deck, None,"FACE", filter_visible=True)
    base.SetEntityCardValues(deck, ent[0], {'PID':12})
    cons = base.CollectEntities(deck, None,"CONS", filter_visible=True)
    mesh.NumberPerimeters([cons[0],cons[2]], f"{y_length_domain//12}")
    mesh.NumberPerimeters([cons[1],cons[3]], "4")
    bottom_ents.append(ent[0])
    
    base.All()
    # Select bottom north
    orXYZ(0,1,1,"","Greater than","Less than",0,yn-10,50)
    ent = base.CollectEntities(deck, None,"FACE", filter_visible=True)
    base.SetEntityCardValues(deck, ent[0], {'PID':15})
    cons = base.CollectEntities(deck, None,"CONS", filter_visible=True)
    mesh.NumberPerimeters([cons[0],cons[2]], f"{x_length_domain//12}")
    mesh.NumberPerimeters([cons[1],cons[3]], "4")
    bottom_ents.append(ent[0])
    
    base.All()
    # Select bottom south
    orXYZ(0,1,1,"","Less than","Less than",0,-yn+10,50)
    ent = base.CollectEntities(deck, None,"FACE", filter_visible=True)
    base.SetEntityCardValues(deck, ent[0], {'PID':13})
    cons = base.CollectEntities(deck, None,"CONS", filter_visible=True)
    mesh.NumberPerimeters([cons[0],cons[2]], f"{x_length_domain//12}")
    mesh.NumberPerimeters([cons[1],cons[3]], "4")
    bottom_ents.append(ent[0])
    base.Or(bottom_ents)
    
    mesh.ReadMeshParams(params)
    mesh.CreateFreeMesh()

    ansa.connections.ReadAssemblyScenario(working_directory + "Meshing_Scenario.ansa")
    ansa.connections.ReadAssemblyScenario(working_directory + "Volume_Scenario.ansa")

    # Change geo file with y_length
    if os.path.exists(target_path+input_file+".geo"):
        generate_geo(target_path+input_file+".geo", input_file, y_length_domain)
    else:
        print("Geo file not found, skipping modification. ### MANUALLY CHANGE y_length IN THE GEO FILE. ###")

    # Save
    base.SaveAs(target_path+input_file+".ansa")
    print (input_file, "saved\n")    


# if __name__ == '__main__':
	