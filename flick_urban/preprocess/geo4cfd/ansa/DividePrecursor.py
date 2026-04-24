# PYTHON script
import os
import ansa
from RebuildInlet import DividePIDbyNormalVector
from ansa import *

deck = constants.OPENFOAM
def main():
	base.CreateShellsFromSolidFacets("skin", 11)
	DividePIDbyNormalVector(11,[8,17,9,19,16,18])
	
	# # Set all fluids to the same PID
	prec_ents = base.CollectEntities(deck, None, "__ALL_ENTITIES__")
	for ent in prec_ents:
		base.SetEntityCardValues(deck,ent,{"PID": 10})

	base.Compress("")

if __name__ == '__main__':
	main()


