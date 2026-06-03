# PYTHON script
import os
import ansa
from ansa import *

deck = constants.OPENFOAM
# USER: Set these for your case
output_path = "/path/to/case/output/"
output_name = "case_name"

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
	
def main():
	# Need some documentation? Run this with F5
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


