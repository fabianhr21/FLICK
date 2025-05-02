import pyAlya
import numpy as np
from scipy.interpolate import griddata

## Input parameters
BASEDIR = 'examples/data/sod2d' #Path where the data is
CASESTR = 'tgv_cp'              #Name of the file
instant = 2001                  #Instant to load

## Contour parameters
varcont = 'qcrit'               #Variable to do the contour of
contval = 0.001                 #Contour value
varcol  = 'u'                   #Variable to colour the contour with (If it is a vector field, the magnitude will be used to colour the contour)
interp  = 'nn'                  #Interpolation method to colour the contour (nn: Nearest Neighbor [RECOMENDED], fem: Finite Element Method, femnn: Hybrid between FEM and NN)
varlist = [varcont, varcol]

## Clip area
xmin = 3.
xmax = 5.
ymin = 4.
ymax = 6.
zmin = 4.
zmax = 6.

## Output name
OUTDIR  = '.' #Output directory
outfile = '%s/%s_%s_%s' % (OUTDIR, CASESTR, varcont, varcol)

## Read data
mesh  = pyAlya.MeshSOD2D.read(CASESTR, basedir=BASEDIR)
field = pyAlya.FieldSOD2D.read(CASESTR, varlist, instant, mesh.xyz, basedir=BASEDIR)

## Clip data
cube = pyAlya.Geom.SimpleCube(xmin,xmax,ymin,ymax,zmin,zmax)
cmesh,mask = mesh.clip(cube)
cfield = field.selectMask(mask)
#cmesh = mesh
#cfield = field
## Create contour 
#Qmesh, normals  = pyAlya.plotting.pvcontour(cmesh, cfield, varcont, contval)

### Interpolate color variable to contour
#if cmesh.xyz.shape[0] > 0:
#        uq = griddata(cmesh.xyz, cfield['u'][:,0], (Qmesh.xyz[:,0], Qmesh.xyz[:,1], Qmesh.xyz[:,2]), method='linear')
#else:
#        uq = np.zeros((cmesh.xyz.shape[0],), dtype=np.double)
#### Add normals to the field
#Qfield = pyAlya.FieldSOD2D(xyz=Qmesh.xyz, ptable=Qmesh.partition_table, Normals=normals)
#
cmesh.write(outfile)
cfield.write(outfile, exclude_vars=[varcont])
## Print timmings
pyAlya.cr_info()