'''
#/*
#==================================================
#*                    HEADER
#==================================================
#
#Developed by:Nikolas Christodoulou
#Date:08/11/2016
#version: v08
#
#
# 	 Copyright (c) 2016 BETA CAE Systems S. A. 
#                              All rights reserved.
#
#=================================================
#*                   DISCLAIMER
#=================================================
#
# BETA CAE Systems assumes no responsibility or 
# liability for any damages, errors, inaccuracies 
# or data loss caused by installation or use of this 
# software.
#
#=================================================
#*                    HISTORY
#=================================================
#
#Date:20170104                 Updater:       n.christodoulou
#Modifications:Works on unmeshed macros and FE mesh
#-------------------------------------------------
#Date:20190411		            Updater:	n.christodoulou
#Modifications: Generates full spherical and half spherical in addition to orthogonal domain
#-------------------------------------------------
#Date:20220221		            Updater:	n.christodoulou
#Modifications: Tolerances in Isolate has been changed in order to support extreme values in Domain size (Half Sphere). Additionally, scaling is performed if the Domain size is extreme in order for Remove_dbl to work fast.
#-------------------------------------------------
#Description:
#
#
#
#*/ 
'''

import ansa
from ansa import *
import random
from random import *

deck =constants.OPENFOAM
@session.defbutton('TOPO_', 'CreateDomain','Creates a domain/windtunnel')
def CreateDomain_main():
	currRow = 0
	rows = 3
	cols = 2
	window = guitk.BCWindowCreate( "CreateDomain", guitk.constants.BCOnExitDestroy)
	grid = guitk.BCGridLayoutCreate( window, rows, cols )
	label = guitk.BCLabelCreate( grid, "Type of Domain" )
	regularCombo = guitk.BCComboBoxCreate( grid, None )
    #Combo box title
	guitk.BCComboBoxInsertItem( regularCombo, "Orthogonal", -1 )
	guitk.BCComboBoxInsertItem( regularCombo, "Spherical", -1 )
	guitk.BCComboBoxInsertItem( regularCombo, "Hemispherical", -1 )
	guitk.BCGridLayoutAddWidget( grid, label, currRow, 0, guitk.constants.BCAlignAuto )
	guitk.BCGridLayoutAddWidget( grid, regularCombo, currRow, 1, guitk.constants.BCAlignAuto )	
	BCButtonGroup_1 = guitk.BCButtonGroupCreate(window, "For Ortho Domain set offset distance from boundary faces (absolute values). For Spherical set Radius", guitk.constants.BCVertical)
	BCBoxLayout_1 = guitk.BCBoxLayoutCreate(BCButtonGroup_1, guitk.constants.BCVertical)
	BCGridLayout = guitk.BCGridLayoutCreate(BCBoxLayout_1, 8, 2)
	GridLayoutWidgets = [None]*21
	GridLayoutWidgets[0]=guitk.BCLabelCreate(BCGridLayout, "+X")
	GridLayoutWidgets[1]=guitk.BCLineEditCreate(BCGridLayout, "0")
	GridLayoutWidgets[2]=guitk.BCLabelCreate(BCGridLayout, "+Y")
	GridLayoutWidgets[3]=guitk.BCLineEditCreate(BCGridLayout, "0")
	GridLayoutWidgets[4]=guitk.BCLabelCreate(BCGridLayout, "+Z")
	GridLayoutWidgets[5]=guitk.BCLineEditCreate(BCGridLayout,"0")
	GridLayoutWidgets[6]=guitk.BCLabelCreate(BCGridLayout, "-X")
	GridLayoutWidgets[7]=guitk.BCLineEditCreate(BCGridLayout, "0")
	GridLayoutWidgets[8]=guitk.BCLabelCreate(BCGridLayout, "-Y")
	GridLayoutWidgets[9]=guitk.BCLineEditCreate(BCGridLayout, "0")
	GridLayoutWidgets[10]=guitk.BCLabelCreate(BCGridLayout, "-Z")
	GridLayoutWidgets[11]=guitk.BCLineEditCreate(BCGridLayout, "0")
	GridLayoutWidgets[12]=guitk.BCLabelCreate(BCGridLayout, "Radius")
	GridLayoutWidgets[13]=guitk.BCLineEditCreate(BCGridLayout, "0")
	GridLayoutWidgets[20]=guitk.BCCheckBoxCreate(window, "Connect model with Symmetry (only for symmetric models)")
	for i in range(0, 14, 2):
		guitk.BCGridLayoutAddWidget(BCGridLayout, GridLayoutWidgets[i], int(i/2), 0, guitk.constants.BCAlignAuto)
		guitk.BCGridLayoutAddWidget(BCGridLayout, GridLayoutWidgets[i+1], int(i/2), 1, guitk.constants.BCAlignAuto)
		guitk.BCLineEditSetAlignment(GridLayoutWidgets[i+1], guitk.constants.BCAlignRight)	
	BCDialogButtonBox_1 = guitk.BCDialogButtonBoxCreate(window)
	data = GridLayoutWidgets
	guitk.BCHide(data[12])
	guitk.BCHide(data[13])
	data[14]=window
	data[15]=0
	guitk.BCWindowSetAcceptFunction(window, _OK, data)
	guitk.BCWindowSetRejectFunction(window, _cancel, 0)		
	guitk.BCComboBoxSetActivatedFunction( regularCombo, _ComboActivated, data )	
	guitk.BCShow( window )

def _ComboActivated(combo, index, data):
		if guitk.BCComboBoxCurrentItem(combo)==0:
			guitk.BCShow(data[0])
			guitk.BCShow(data[1])
			guitk.BCShow(data[2])
			guitk.BCShow(data[3])
			guitk.BCShow(data[4])
			guitk.BCShow(data[5])
			guitk.BCShow(data[6])
			guitk.BCShow(data[7])
			guitk.BCShow(data[8])
			guitk.BCShow(data[9])
			guitk.BCShow(data[10])
			guitk.BCShow(data[11])
			guitk.BCShow(data[20])
			guitk.BCHide(data[12])
			guitk.BCHide(data[13])
			
			data[15]=(guitk.BCComboBoxCurrentItem(combo))
			guitk.BCWindowSetAcceptFunction(data[14], _OK, data)
			guitk.BCWindowSetRejectFunction(data[14], _cancel, 0)
			text = guitk.BCComboBoxGetText( combo, index )
			return 0
			
		elif (guitk.BCComboBoxCurrentItem(combo)==1):
			guitk.BCHide(data[0])
			guitk.BCHide(data[1])
			guitk.BCHide(data[2])
			guitk.BCHide(data[3])
			guitk.BCHide(data[4])
			guitk.BCHide(data[5])
			guitk.BCHide(data[6])
			guitk.BCHide(data[7])
			guitk.BCHide(data[8])
			guitk.BCHide(data[9])
			guitk.BCHide(data[10])
			guitk.BCHide(data[11])
			guitk.BCHide(data[20])
			guitk.BCShow(data[12])
			guitk.BCShow(data[13])
			data[15]=(guitk.BCComboBoxCurrentItem(combo))
			guitk.BCWindowSetAcceptFunction(data[14], _OK, data)
			guitk.BCWindowSetRejectFunction(data[14], _cancel, 0)
			return 0
		elif guitk.BCComboBoxCurrentItem(combo)==2:
			guitk.BCHide(data[0])
			guitk.BCHide(data[1])
			guitk.BCHide(data[2])
			guitk.BCHide(data[3])
			guitk.BCHide(data[4])
			guitk.BCHide(data[5])
			guitk.BCHide(data[6])
			guitk.BCHide(data[7])
			guitk.BCHide(data[8])
			guitk.BCHide(data[9])
			guitk.BCHide(data[10])
			guitk.BCHide(data[11])
			guitk.BCShow(data[12])
			guitk.BCShow(data[13])
			guitk.BCShow(data[20])
			data[15]=(guitk.BCComboBoxCurrentItem(combo))
			guitk.BCWindowSetAcceptFunction(data[14], _OK, data)
			guitk.BCWindowSetRejectFunction(data[14], _cancel, 0)
			return 0
			
def _OK(w, data):
	if data[15]==0:
		ind = 1
		ret = guitk.BCLineEditGetText(data[ind])
		try:
			dxpos = float(ret)
			if dxpos < 0:
				print('Define a valid +X value.')
				return 0
		except:
			print('Define a valid +X value.')
			return 0
		ind += 2
		
		ret = guitk.BCLineEditGetText(data[ind])
		try:
			dypos = float(ret)
			if dypos < 0:
				print('Define a valid +Y value.')
				return 0
		except:
			print('Define a valid +Y value.')
			return 0
		ind += 2
		
		ret = guitk.BCLineEditGetText(data[ind])
		try:
			dzpos = float(ret)
			if dzpos < 0:
				print('Define a valid +Z value.')
				return 0
		except:
			print('Define a valid +Z value.')
			return 0
		ind += 2
		
		ret = guitk.BCLineEditGetText(data[ind])
		try:
			dxneg = float(ret)
			if dxneg < 0:
				print('Define a valid -X value.')
				return 0
		except:
			print('Define a valid -X value.')
			return 0
		ind += 2
		
		ret = guitk.BCLineEditGetText(data[ind])
		try:
			dyneg = float(ret)
			if dyneg < 0:
				print('Define a valid -Y value.')
				return 0
		except:
			print('Define a valid -Y value.')
			return 0
		ind += 2
		
		ret = guitk.BCLineEditGetText(data[ind])
		try:
			dzneg = float(ret)
			if dzneg < 0:
				print('Define a valid -Z value.')
				return 0
		except:
			print('Define a valid -Z value.')
			return 0
		ind += 2
		
		_multibox(dxpos,dypos,dzpos,dxneg,dyneg,dzneg,guitk.BCCheckBoxIsChecked(data[20]))
		return 1
	elif data[15]==1:
		_full_sphere(guitk.BCLineEditGetText(data[13]))
		return 1
	elif data[15]==2:
		_half_sphere(guitk.BCLineEditGetText(data[13]),guitk.BCCheckBoxIsChecked(data[20]))
		return 1
		
		
		
		
		
def _cancel(w, data):	
	print("Cancel button pressed. Script execution terminated!")	
	return 1
	
def _multibox(dxpos,dypos,dzpos,dxneg,dyneg,dzneg,f):
	print ("Creating the domain.Please wait...")
	base.BlockRedraws(True)
	#Collect shells
	collected_entities={}
	faces=base.CollectEntities(deck, None, "FACE",False,True)
	shells=base.CollectEntities(deck, None, "SHELL",False,True)
	collected_entities=faces+shells
	if len(collected_entities)==0:
		print ('No entities exist in this database. Aborting...')
		base.BlockRedraws(False)
		return 0
	# Create a size box from the visible nodes
	SizeBox = base.SizeBoxOrtho(collected_entities,db_or_visible = 'Visible', min_flag = False)
	# Collect the points from the size box
	CornerPoints = base.CollectEntities(deck, SizeBox, "SIZEBOXPOINT")
	# Calculate the COG of the SizeBox
	(x,y,z)=base.Cog(SizeBox)
	# Move the corner points based on dx, dy, dz inputs
	for point in CornerPoints:
		vals = ansa.base.GetEntityCardValues(deck, point, {'X','Y','Z'})
		if vals['X']>= x:
			newX = vals['X']+dxpos
		else:
			newX = vals['X']-dxneg
		ansa.base.SetEntityCardValues(deck,point,{'X': newX})
		#print('X ', vals['X'], newX)
		if vals['Y']>= y:
			newY = vals['Y']+dypos
		else:
			newY = vals['Y']-dyneg
		ansa.base.SetEntityCardValues(deck,point,{'Y': newY})
		#print('Y ', vals['Y'], newY)
		if vals['Z']>= z:
			newZ = vals['Z']+dzpos
		else:
			newZ = vals['Z']-dzneg
		ansa.base.SetEntityCardValues(deck,point,{'Z': newZ})
		
	#Create a Property called Domain
	VolumeBoxProperty=base.CreateEntity(deck, "SHELL_PROPERTY", {'Name':'Domain'})
	id=base.GetEntityCardValues(deck,VolumeBoxProperty,{'PID',})
	# Create a Part called Domain_Part
	VolumeBoxPart = base.CreateEntity(deck, "ANSAPART", {'Name': 'Domain_Part_' + str(randint(1, 101))})

	SizeBoxFaces = base.CollectEntities(deck, SizeBox, 'SIZEBOXFACE')
	print(SizeBoxFaces)

	old_length = base.GetANSAdefaultsValues(('perimeter_length',))
	base.SetANSAdefaultsValues({'perimeter_length': '200'})
	NewFaces = morph.MorphConvert("MorphFacesToGeo", SizeBoxFaces, 1)
	base.SetANSAdefaultsValues({'perimeter_length': old_length['perimeter_length']})

	base.Orient(NewFaces)
	base.SetEntityPart(NewFaces, VolumeBoxPart)
	pid_names = ["inlet","lateralDomainNorth","topDomain","lateralDomainSouth","groundDomain","outlet"]

	# Assign a unique PID to each new face
	for idx, NewFace in enumerate(NewFaces, start=1):
		# Create a new property for each face
		prop_name = pid_names[idx-1]
		VolumeBoxProperty = base.CreateEntity(deck, "SHELL_PROPERTY", {'Name': prop_name})
		pid_values = base.GetEntityCardValues(deck, VolumeBoxProperty, {'PID'})
		base.SetEntityCardValues(deck, NewFace, {'PID': pid_values['PID']})
		#base.SetEntityId(NewFace, idx+1, True, False)
	
	base.Compress({'EMPTY ANSAPART': 1, 'Properties': 1}, deck)
	base.Compress("")
	ansa.base.SizeBoxDelete(SizeBox)
	base.BlockRedraws(False)
	for idx,Face in enumerate(NewFaces,start=4):
		face = base.GetEntity(deck, "SHELL_PROPERTY", idx)
		base.SetEntityId(face, idx-2, True, False)
		print(face)
	
	#Connect with Symmetry
	ids=[]
	if f==True:
		single_cons=[]
		all_cons=base.CollectEntities(deck,None, "CONS",False,False)
		for a in all_cons:
			n=base.GetEntityCardValues(deck,a,{'Number of Pasted Cons'})
			if n['Number of Pasted Cons']==1:
				single_cons.append(a)
		for N in NewFaces:
			id=base.GetEntityCardValues(deck,N,{'ID'})
			ids.append(id['ID'])
		ids.sort()
		if y<0:
			proj_face=base.GetEntity(deck,'FACE',ids[1])
		else:
			proj_face=base.GetEntity(deck,'FACE',ids[3])
		project = base.ConsProjectNormal(single_cons, proj_face, 0.0, True, True,False,True)
	print ("Domain completed.")

def _full_sphere(r):
	base.BlockRedraws(True)
	radius=int(r)
	if radius==0:
		print ('Insert a valid Radius.')
	elif radius<0:
		print ('Insert a valid Radius.')
	else:
		print ("Creating the domain.Please wait...")
		#Collect shells
		collected_entities={}
		faces=base.CollectEntities(deck, None, "FACE",False,True)
		shells=base.CollectEntities(deck, None, "SHELL",False,True)
		collected_entities=faces+shells
		if len(collected_entities)==0:
			print ('No entities exist in this database. Aborting...')
			return 0
		# Create a size box from the visible nodes
		SizeBox = base.SizeBoxOrtho(collected_entities,db_or_visible = 'Visible', min_flag = False)
		# Calculate the COG of the SizeBox
		(x,y,z)=base.Cog(SizeBox)
		#base.CreateVolumeSphere(center_point, radius, part, property, volumes)
		#print (x,y,z)
		VolumeBoxPart=base.CreateEntity(deck, "ANSAPART", {'Name':'Domain_Part_'+'radius='+str(radius)+'_'+str(randint(1,101))})
		VolumeBoxProperty=base.CreateEntity(deck, "SHELL_PROPERTY", {'Name':'Farfield'})
		xyz_list=[]
		xyz_list.append(x)
		xyz_list.append(y)
		xyz_list.append(z)		
		defval=base.GetANSAdefaultsValues({'ctolerance','ntolerance','perimeter_length'})
		ctol=defval['ctolerance']
		ntol=defval['ntolerance']
		res=defval['perimeter_length']
		#Alter Defaults
		base.SetANSAdefaultsValues({'ctolerance':radius/10000,'ntolerance':radius/10000,'perimeter_length':radius/10})	
		sphere=base.CreateVolumeSphere(xyz_list,radius,VolumeBoxPart,VolumeBoxProperty,False)
		base.Compress({'EMPTY ANSAPART':1,'Properties':1},deck)
		ansa.base.SizeBoxDelete(SizeBox)
		#Set Defaults again
		base.SetANSAdefaultsValues({'ctolerance':ctol,'ntolerance':ntol,'perimeter_length':res})
		base.SetViewAngles(f_key="F6")
		base.BlockRedraws(False)
		#Set PID type to Farfield
		pids=base.CollectEntities(deck,None,'SHELL_PROPERTY',False,False)
		for p in pids:
			name=base.GetEntityCardValues(deck,p,{'Name'})
			if name['Name']=='Farfield':
				cur_deck=base.CurrentDeck()
				if cur_deck==12:
					base.SetEntityCardValues(deck,p,{'TYPE':'patch'})
				elif cur_deck==9:
					base.SetEntityCardValues(constants.STAR,p,{'TYPE':'Freestream'})
				elif cur_deck==7:
					base.SetEntityCardValues(constants.FLUENT,p,{'ZONE_TYPE':'pressure-far-field'})
		print ("Domain completed.")
		
	
def _half_sphere(r,f):
	base.BlockRedraws(True)
	radius=int(r)
	if radius==0:
		print ('Insert a valid Radius.')
	elif radius<0:
		print ('Insert a valid Radius.')
	else:
		print ("Creating the domain.Please wait...")
		#Collect shells
		collected_entities={}
		faces=base.CollectEntities(deck, None, "FACE",False,True)
		shells=base.CollectEntities(deck, None, "SHELL",False,True)
		collected_entities=faces+shells
		if len(collected_entities)==0:
			print ('No entities exist in this database. Aborting...')
			return 0
		# Create a size box from the visible nodes
		SizeBox = base.SizeBoxOrtho(collected_entities,db_or_visible = 'Visible', min_flag = False)	
		# Calculate the COG of the SizeBox
		(x,y,z)=base.Cog(SizeBox)
		#base.CreateVolumeSphere(center_point, radius, part, property, volumes)
		#print (x,y,z)
		VolumeBoxPart=base.CreateEntity(deck, "ANSAPART", {'Name':'Domain_Part_'+'radius='+str(radius)+'_'+str(randint(1,101))})
		VolumeBoxProperty=base.CreateEntity(deck, "SHELL_PROPERTY", {'Name':'Farfield'})
		xyz_list=[]
		xyz_list.append(x)
		#xyz_list.append(y)
		xyz_list.append(0)
		xyz_list.append(z)
		#Read Defaults and create params
		defval=base.GetANSAdefaultsValues({'ctolerance','ntolerance','perimeter_length'})
		ctol=defval['ctolerance']
		ntol=defval['ntolerance']
		res=defval['perimeter_length']		
		
		#Alter Defaults
		base.SetANSAdefaultsValues({'ctolerance':radius/10000,'ntolerance':radius/10000,'perimeter_length':radius/10})	
		sphere=base.CreateVolumeSphere(xyz_list,radius,VolumeBoxPart,VolumeBoxProperty,False)
		base.Or(sphere)
		
		
		#Cut half geometry
		base.PlaneCut(0, 0, 0, 0, 0, 1, 1, 0, 0, sphere, produce_plane_faces=True, perform_topology=True)
		
		if radius>100000:
			base.ScalePart(VolumeBoxPart,transformation_mode="MOVE",x=xyz_list[0],y=xyz_list[1],z=xyz_list[2],factor=0.001, connectivity=True  )	
			
		
		if y<0:
			base.RmdblSymmetric ("negative", "delete", 0, 0, 0, 0., 1., 0., radius/10000, 90)
			base.Orient()
		else:
			base.RmdblSymmetric ("positive", "delete", 0, 0, 0, 0., 1., 0., radius/10000, 90)
			base.Orient()
			base.Orient()
		base.Compress({'EMPTY ANSAPART':1,'Properties':1},deck)
		ansa.base.SizeBoxDelete(SizeBox)
		faces_1={}
		face_vis=base.CollectEntities(deck,None,'FACE',False,True)
		isolate=base.IsolateConnectivityGroups(face_vis,1,0,feature_angle=40., feature_type="concave")
		pid_sym=base.CreateEntity(deck, "SHELL_PROPERTY", {'Name':'Symmetry','TYPE':'symmetry'})
		if len(isolate)>1:
			if len(isolate['group_2'])==1 :
				for key in isolate['group_2']:
					pid_target=base.GetEntityCardValues(deck,pid_sym,{'PID'})
					pid_source=base.GetEntityCardValues(deck,key,{'PID'})
					base.SetEntityCardValues(deck,key,{'PID':pid_target['PID']})
			else:
				for key in isolate['group_1']:
					pid_target=base.GetEntityCardValues(deck,pid_sym,{'PID'})
					pid_source=base.GetEntityCardValues(deck,key,{'PID'})
					base.SetEntityCardValues(deck,key,{'PID':pid_target['PID']})
		
		#Set Scale again
		if radius>100000:
			base.ScalePart(VolumeBoxPart,transformation_mode="MOVE",x=xyz_list[0],y=xyz_list[1],z=xyz_list[2],factor=1000, connectivity=True  )	
		
		#Set Defaults again
		base.SetANSAdefaultsValues({'ctolerance':ctol,'ntolerance':ntol,'perimeter_length':res})
		base.All()
		base.SetViewAngles(f_key="F6")
		base.BlockRedraws(False)
		
		
		#Set PID type to Farfield
		pids=base.CollectEntities(deck,None,'SHELL_PROPERTY',False,False)
		for p in pids:
			name=base.GetEntityCardValues(deck,p,{'Name'})
			if name['Name']=='Farfield':
				cur_deck=base.CurrentDeck()
				if cur_deck==12:
					base.SetEntityCardValues(deck,p,{'TYPE':'patch'})
				elif cur_deck==9:
					base.SetEntityCardValues(constants.STAR,p,{'TYPE':'Freestream'})
				elif cur_deck==7:
					base.SetEntityCardValues(constants.FLUENT,p,{'ZONE_TYPE':'pressure-far-field'})
		
		#Connect with Symmetry
		if f==True:
			sym_face=base.CollectEntities(deck,pid_sym,'FACE',False,False)
			single_cons=[]
			all_cons=base.CollectEntities(deck,None, "CONS",False,False)
			for a in all_cons:
				n=base.GetEntityCardValues(deck,a,{'Number of Pasted Cons'})
				if n['Number of Pasted Cons']==1:
					single_cons.append(a)
			project = base.ConsProjectNormal(single_cons, sym_face, 0.0, True, True,False,True)

					
		print ("Domain completed.")
		
			        	


