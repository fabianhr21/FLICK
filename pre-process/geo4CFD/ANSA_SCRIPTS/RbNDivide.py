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
output_path = "/home/fabianh/ANSA/Data/cedval/"
output_name = "cedval14"

def splitTOhexa():
	m =utils.Messenger()
	vols = base.CollectEntities(constants.OPENFOAM, None, 'VOLUME')
	sols = base.CollectEntities(constants.OPENFOAM, None, 'SOLID')
	
	vols = [vol for vol in vols if mesh.IsVolumeMeshed(vol)]
	polys = [vol for vol in vols if vol.get_entity_values(constants.OPENFOAM, ['Type'])['Type'] in ['Polyhedral', 'Hextreme']]
	vols = [vol for vol in vols if vol.get_entity_values(constants.OPENFOAM, ['Type'])['Type'] not in ['Polyhedral', 'Hextreme']]
	
	sols = [sol for sol in sols if sol.get_entity_values(constants.OPENFOAM, ['type'])['type'] != 'POLYHEDRON']
	polys = polys + [sol for sol in sols if sol.get_entity_values(constants.OPENFOAM, ['type'])['type'] == 'POLYHEDRON']
	
	if not vols + sols and polys:
		m.print('Polyhedral mesh found!', 'html')
		m.print('Cannot split polyhedral elements.', 'html')
		return True
	elif not vols + sols:
		m.print('No volume elements found.', 'html')
		return True
	
	ret_vols, ret_sols = mesh.SplitToHexa(vols + sols)
	
	sols_from_ret_vols = []
	light_sols = 0
	if ret_vols:
		for vol in ret_vols:
			ret = vol.get_entity_values(constants.OPENFOAM, ['Light volume representation', 'Tetras', 'Pentas', 'Pyramids', 'Hexas', 'Polyhedrals'])
			if ret['Light volume representation'] == 'YES':
				light_sols = light_sols + ret['Tetras'] + ret['Pentas'] + ret['Pyramids'] + ret['Hexas'] + ret['Polyhedrals']
			else:
				sols_from_ret_vols.extend(base.CollectEntities(constants.OPENFOAM, ret_vols, 'SOLID'))
	
	conv_sols = set(sols_from_ret_vols) | set(ret_sols) if ret_sols else sols_from_ret_vols
	
	m.print(str(len(conv_sols) + light_sols) + ' hexas generated.', 'html')
	
	del m
	return True

def DividePIDbyFaceOrientation(pid_id: int, new_pid_list=None):
    """
    Splits each face in the given PID into up to six new PIDs,
    based on its GetFaceOrientation vector (or normal if that fails).

    Orientation buckets (and new_pid_list order):
      1: west   (dx < 0)
      2: north  (dy > 0)
      3: top    (dz > 0)
      4: south  (dy < 0)
      5: ground (dz < 0)
      6: east   (dx > 0)

    :param pid_id: original SHELL_PROPERTY PID to split.
    :param new_pid_list: list of six ints [pid_west, pid_north, pid_top,
                         pid_south, pid_ground, pid_east].  Use 0 to delete.
    """
    # --- validate new_pid_list ---
    if new_pid_list is None:
        new_pid_list = [0, 3, 4, 5, 6, 2]
    if len(new_pid_list) != 6:
        raise ValueError("new_pid_list must have exactly six elements")

    # --- fetch the property and its shells/faces ---
    prop = base.GetEntity(deck, "SHELL_PROPERTY", pid_id)
    if not prop:
        print(f"No SHELL_PROPERTY with PID {pid_id}")
        return

    shells = base.CollectEntities(deck, prop, "SHELL")
    if not shells:
        print(f"No shells found under PID {pid_id}")
        return

    # prepare six buckets
    groups = {i: [] for i in range(6)}

    for shell in shells:
        face = base.GetEntity(deck, "SHELL", shell._id)

        # try GetFaceOrientation, fall back on normal if needed
        vec = base.GetFaceOrientation(face)
        try:
            dx, dy, dz = vec
        except (TypeError, ValueError):
            # fallback: use shell normal
            nvec = base.GetNormalVectorOfShell(shell)
            nvec = calc.Normalize(nvec)
            dx, dy, dz = nvec

        # pick the dominant axis
        adx, ady, adz = abs(dx), abs(dy), abs(dz)
        if adx >= ady and adx >= adz:
            idx = 0 if dx < 0 else 5      # west or east
        elif ady >= adx and ady >= adz:
            idx = 3 if dy > 0 else 1      # south or north
        else:
            idx = 4 if dz > 0 else 2      # ground or top

        groups[idx].append(face)
        
    # re-PID (or delete) each group
    for idx, faces in groups.items():
        target_pid = new_pid_list[idx]
        for face in faces:
            if target_pid == 0:
                base.DeleteEntity(face, False, False)
            else:
                base.SetEntityCardValues(deck, face, {"PID": target_pid})

    used_buckets = sum(bool(g) for g in groups.values())
    print(f"Divided PID {pid_id} into {used_buckets} orientation‐based PIDs.")

def main():
	base.Clear()
	# Get inlet entity faces
	inlet = base.GetEntity(deck, "SHELL_PROPERTY", 2)
	source = base.CollectEntities(deck, inlet, "FACE")
	extrude = mesh.VolumesExtrude()
	
	# Extrude wth offset
	extrusion = extrude.offset(source=source, source_remove=source, steps=4, distance=-150.0)
	
	# Delete inlet contents
	base.DeleteEntity(inlet,True)
	base.CreateEntity(deck, "SHELL_PROPERTY", {"Name": "inlet", "PID":2})
	
	
	# Create the skin, divide and assign to their respective domain PIDs
	solid_ex = base.GetEntity(deck, "SOLID_PROPERTY", 13)
	base.CreateShellsFromSolidFacets("skin", 13)
	#base.Orient()
	DividePIDbyFaceOrientation(12)
	
	# Create Domain Precursor
	inlet = base.GetEntity(deck, "SHELL_PROPERTY", 2)
	base.GeoTranslate("COPY","AUTO_OFFSET","SAME PART","COPY",-100,0,0,inlet,keep_connectivity=True,draw_results=False)
	
	# # Precursor parameters
	distance = -5000
	element_length = 20

	# Create Precursor
	precursor_outlet = base.GetEntity(deck, "SHELL_PROPERTY", 14)
	source = base.CollectEntities(deck, precursor_outlet, "SHELL")
	extrude = mesh.VolumesExtrude()
	extrusion = extrude.offset(source=source, source_remove=source, steps=25, distance=distance)
	base.DeleteEntity(precursor_outlet,True,True)
	
	#Orient result
	base.Orient()
	
	# SHow only precursor
	precursor = base.GetEntity(deck, "VOLUME", 3)
	base.Or(precursor)

	# Create skin
	base.CreateShellsFromSolidFacets("skin", 12)
	DividePIDbyFaceOrientation(12,[19,18,9,20,8,17])#[West, South, GRound, North, Top, East] bc it was inverted

	# # Set all fluids to the same PID
	prec_ents = base.CollectEntities(deck, None, "__ALL_ENTITIES__")
	for ent in prec_ents:
		base.SetEntityCardValues(deck,ent,{"PID": 11})

	# Auto-orientation outside the volume
	props = base.CollectEntities(deck, None, "__ALL_ENTITIES__", False)
	base.AutoCalculateOrientation(props, False)
	
    # Delete geometry, keep faces
	faces = base.CollectEntities(deck, None, 'FACE', recursive = True)
	mesh.ReleaseElements(faces)
	base.DeleteFaces(faces)
	print("Elements released from faces\n")

	# COmpress empty entities12
	base.Compress("")
	
	# Orient shells
	base.AutoCalculateOrientation("Visible", False)
	
	#Split to Hexa
	splitTOhexa()
	
	#Output CGNS
	base.OutputCGNS(
        f"{output_path}{output_name}.cgns",
        mode="all",
        filetype="HDF5",
        format="unstructured",
        unstructured_options="separated",
        write_families="yes",
        version="v3.2.0",
        bc_correspondence="yes"
    )


if __name__ == '__main__':
	main()


