import ansa
from ansa import base, mesh, constants,session,dm
from ansa import *
import CreateDomain 
import os 
import sys

deck = constants.OPENFOAM
working_directory = "/home/fabianh/ANSA/automation/"
input_file = "mesh_bcn_script-test_bcn_Buildings"
target_path = "/home/fabianh/ANSA/automation/"
h_max = 150
def main():
	# Input StereoLithography from City4CFD
    session.New("discard")
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
    shells = base.PickEntities(deck, ents)
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
    height = z_max - z_min
    print("Max building height:", height)
    h_max = height

    # Creates domain and assign different PID to faces, core script modified (change in your directory)
    #xp,yp,zp,xn,yn,zn
    CreateDomain._multibox(40*h_max,30*h_max,20*h_max,20*h_max,30*h_max,z_min,False)    
	
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
    abl_sb = base.SizeBoxMinMax(None, min_coords, max_coords, 30, 40)
    ## Close ground
    min_coords = [x_min-(20*h_max),y_min- (30*h_max),z_min]
    max_coords = [x_max+(40*h_max),y_max +(30*h_max),z_max+h_max]
    close_ground_sb = base.SizeBoxMinMax(None, min_coords, max_coords, 85, 85)
    ## Wake 1 (10h)
    min_coords = [x_min + (15*h_max),y_min,z_min]
    max_coords = [x_min+ (25*h_max),y_max ,z_max]
    wake1_sb = base.SizeBoxMinMax(None, min_coords, max_coords, 40, 60)   
    ## Wake 2 (20h)
    min_coords = [x_min+ (24*h_max+20),y_min ,z_min]
    max_coords = [x_min +(44*h_max),y_max,z_max]
    wake2_sb = base.SizeBoxMinMax(None, min_coords, max_coords, 60, 80)
    
    # Save Size Boxes to a list
    sbs = [buildings_sb, campus_sb, abl_sb, close_ground_sb, wake1_sb, wake2_sb]
    
    #Uses STL algorith to recover exact geometry
    mesh.AspacingSTL('1%', 50, 0, 0.1)
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
    
    base.SaveAs(target_path+input_file+".ansa")
    print (input_file, "saved\n")    


if __name__ == '__main__':
	main()
