import pyQvarsi
import numpy as np
from   scipy.interpolate import griddata

## Input parameters
BASEDIR = '/home/benet/Baixades'  #Path where the data is
CASESTR = 'cube_bound_N_4_lineal' #Name of the file

## Slice parameters
direction = [0,0,1]    #Slice direction
origin    = [0,0,0.5]  #Origin of the slice

## Clip area
xmin = -0.1
xmax = 1.1
ymin = -0.1
ymax = 1.1
zmin = -0.1
zmax = 1.1

## Output name
OUTDIR  = '.' #Output directory
outfile = '%s/slice_%s' % (OUTDIR, CASESTR)

omega = 3*np.pi/2
phi   = np.pi/2

## Read data
mesh  = pyQvarsi.MeshSOD2D.read(CASESTR, basedir=BASEDIR)
scaf  = np.cos(omega*mesh.x+phi)*np.sin(omega*mesh.y-phi)
field = pyQvarsi.FieldSOD2D(xyz=mesh.xyz,ptable=mesh.partition_table,f=scaf)

## Clip data
cube = pyQvarsi.Geom.SimpleCube(xmin,xmax,ymin,ymax,zmin,zmax)
cmesh,mask = mesh.clip(cube)
cfield = field.selectMask(mask)

## Create contour 
Smesh = pyQvarsi.plotting.pvslice(cmesh, direction, origin=origin)

## Interpolate field to contours
if Smesh.xyz.shape[0] > 0:
	F = griddata(cmesh.xyz, cfield['f'], Smesh.xyz)
else:
	F=np.zeros(Smesh.xyz.shape[0])
Sfield = pyQvarsi.Field(xyz=Smesh.xyz, ptable=Smesh.partition_table, f=F)

integral = Smesh.integral(Sfield['f'], kind='surf')
analytic = ((np.cos(omega - phi) - np.cos(phi)) * (np.sin(phi) - np.sin(omega + phi))) / omega**2

print(analytic)
print(integral)
print(integral-analytic)

Smesh.write(outfile)
Sfield.write(outfile)

## Print timmings
pyQvarsi.cr_info()