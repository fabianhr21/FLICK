import pyAlya
import numpy as np

BASEDIR = 'examples/data/sod2d'
CASESTR = 'tgv_lo'
varlist = ['u', 'qcrit']
instant = 2001

mesh  = pyAlya.MeshSOD2D.read(CASESTR, basedir=BASEDIR)
field = pyAlya.FieldSOD2D.read(CASESTR, varlist, instant, mesh.xyz, basedir=BASEDIR)
gradv = mesh.gradient(field['u'])
field['q'] = pyAlya.postproc.QCriterion(gradv)

##PyVista plot
pyAlya.plotting.pvcontour(mesh, field, 'qcrit', 0.001, varcolour='u')

## Output to ParaView
mesh.write('output')
field.write('output',fmt='vtkh5')

## Print timmings
pyAlya.cr_info()