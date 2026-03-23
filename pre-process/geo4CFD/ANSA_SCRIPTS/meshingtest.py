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
import shutil
from ansa import batchmesh
deck = constants.OPENFOAM

def merge_pids(deck,src_pid_list,tgt_pid_list):
	for src_pid, tgt_pid in zip(src_pid_list, tgt_pid_list):
		src = base.GetEntity(deck, "SHELL_PROPERTY", src_pid)
		faces = base.CollectEntities(deck, src, "FACE")
		shells = base.CollectEntities(deck, src, "SHELL")
		for face in faces:
			base.SetEntityCardValues(deck, face, {'PID':tgt_pid})
		for shell in shells:
			base.SetEntityCardValues(deck, shell, {'PID':tgt_pid})	
		base.DeleteEntity(src, True)
		

def main():
	merge_pids(deck,[11,12,13,14,15],[1,7,5,2,3])
	vol = base.GetEntity(constants.NASTRAN, "VOLUME", 1)
	base.SetEntityCardValues(deck, vol, {'PID':11})	
	name =  base.GetEntity(constants.NASTRAN, "SOLID_PROPERTY", 16)
	base.DeleteEntity(name, True)

	
	
	



if __name__ == '__main__':
	main()
