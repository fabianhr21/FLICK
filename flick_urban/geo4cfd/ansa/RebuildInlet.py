# PYTHON script
import os
import ansa
from ansa import base, mesh, constants,session,dm
from ansa import *
import CreateDomain 
import IsolateNormalVector
import sys
import math

deck = constants.OPENFOAM
def DividePIDbyNormalVector(pid_id: int, new_pid_list=None, angle_tolerance=1.0):
    """
    Separates each face in a given PID into different PIDs based on their normal vector direction.

    :param pid_id: PID (int) of the SHELL_PROPERTY to split.
    :param angle_tolerance: Angle tolerance (in degrees) to group similar normals.
    """
    if new_pid_list is None:
         new_pid_list = [0, 2, 4, 5, 3, 6]
         print(new_pid_list)

    prop = base.GetEntity(deck, "SHELL_PROPERTY", pid_id)
    if not prop:
        print(f"No SHELL")
	# DividePIDbyNormalVector(11,[8,17,16,19,9,18])
	
	# # # Set all fluids to the same PID
	# prec_ents = base.CollectEntities(deck, None, "__ALL_ENTITIES__")
	# for ent in prec_ents:
	# 	base.SetEntityCardValues(deck,ent,{"PID": 10})

	# # new_inlet = base.GetEntity(deck, "SOLID_PROPERTY", 12)
	# #for vol in vol_ents:
	# #	print(vol)
	# #	base.SetEntityId(vol[0],10)
	# # base.SetEntityCardValues(deck, new_inlet, {"PID": 10})
	
	# base.Compress("")PROPERTY with PID {pid_id}")
        return

    shells = base.CollectEntities(deck, prop, "SHELL")
    if not shells:
        print(f"No shells found in PID {pid_id}")
        return

    groups = []
    normals = []

    for shell in shells:
        try:
            nvec = base.GetNormalVectorOfShell(shell)
            nvec = calc.Normalize(nvec)
        except TypeError:
            continue

        found_group = False
        for i, ref in enumerate(normals):
            angle = calc.CalcAngleOfVectors(nvec, ref)
            angle_deg = math.degrees(angle)
            if angle_deg < angle_tolerance:
            	n = base.GetEntity(deck,"SHELL",shell._id)
            	groups[i].append(n)
            	found_group = True
            	break

        if not found_group:
            normals.append(nvec)
            groups.append([shell])

    # Assign each group to a new PID
    print(base.GetEntityType(deck,groups[1][1]))
    print("PID will assign in order to:", new_pid_list)
    for i, group in enumerate(groups):
    	print(i)
    	for ent in group:
            #   base.SetEntityCardValues(deck, ent, {"PID": 13+i})
    		if new_pid_list[i] == 0:
    			base.DeleteEntity(ent,False,False)
    		else:
    			base.SetEntityCardValues(deck, ent, {"PID": new_pid_list[i]})

    print(f"Divided PID {pid_id} into {len(groups)} normal-based PIDs.")


def main():
	# Get inlet entity faces
	inlet = base.GetEntity(deck, "SHELL_PROPERTY", 2)
	source = base.CollectEntities(deck, inlet, "FACE")
	extrude = mesh.VolumesExtrude()
	
	# Extrude wth offset
	extrusion = extrude.offset(source=source, source_remove=source, steps=2, distance=-150.0)
	
	# Delete inlet contents
	base.DeleteEntity(inlet,True)
	base.CreateEntity(deck, "SHELL_PROPERTY", {"Name": "inlet", "PID":2})
	
	
	# Create the skin, divide and assign to their respective domain PIDs
	solid_ex = base.GetEntity(deck, "SOLID_PROPERTY", 12)
	base.CreateShellsFromSolidFacets("skin", 12)
	DividePIDbyNormalVector(11)
	
	# Create Domain Precursor
	inlet = base.GetEntity(deck, "SHELL_PROPERTY", 2)
	base.GeoTranslate("COPY","AUTO_OFFSET","SAME PART","COPY",-100,0,0,inlet,keep_connectivity=True,draw_results=False)
	
	# Precursor parameters
	distance = -5000
	element_length = 20
	
    # Save current entities
	#solids_before = base.CollectEntities(deck, None, "__ALL_ENTITIES_")

	# Create Precursor
	precursor_outlet = base.GetEntity(deck, "SHELL_PROPERTY", 13)
	source = base.CollectEntities(deck, precursor_outlet, "SHELL")
	extrude = mesh.VolumesExtrude()
	extrusion = extrude.offset(source=source, source_remove=source, steps=25, distance=distance)
	base.DeleteEntity(precursor_outlet,True,True)

    # New solids
	#solids_after = base.CollectEntities(deck, None, "__ALL_ENTITIES_")
    # Get the new solids
	#new_solids = [s for s in solids_after if s not in solids_before]

    # #Create the skin, divide and assign to their respective domain PIDs
	# solids = base.GetEntity(deck, "SOLID_PROPERTY", 15)
	# print(solids)
	# solids_ent = base.CollectEntities(deck, solids, "__ELEMENTS__")
	# print(solids_ent)
	#mesh.CreateShellsOnSolidsPidSkin(solids)
	
	# base.CreateShellsFromSolidFacets("skin", 11)
	# DividePIDbyNormalVector(11,[8,17,16,19,9,18])
	
	# # # Set all fluids to the same PID
	# prec_ents = base.CollectEntities(deck, None, "__ALL_ENTITIES__")
	# for ent in prec_ents:
	# 	base.SetEntityCardValues(deck,ent,{"PID": 10})

	# # new_inlet = base.GetEntity(deck, "SOLID_PROPERTY", 12)
	# #for vol in vol_ents:
	# #	print(vol)
	# #	base.SetEntityId(vol[0],10)
	# # base.SetEntityCardValues(deck, new_inlet, {"PID": 10})
	
	# base.Compress("")

if __name__ == '__main__':
	main()


