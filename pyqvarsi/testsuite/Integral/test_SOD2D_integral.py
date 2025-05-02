import pyAlya
import numpy as np

BASEDIR = '/home/benet/Dropbox/UNIVERSITAT/PhD/test_cases/mms_parallel/p7'
CASESTR = 'cube_bound_N_8'

## Read mesh and extract boundary condition
mesh = pyAlya.MeshSOD2D.read(CASESTR,basedir=BASEDIR)
    
## Define analitical function for a scalar and a vectorial field
x0    = 0.5
omega = 3*np.pi/2
phi   = np.pi/2-omega*x0
scafi = np.cos(omega*mesh.x+phi)**2*np.sin(omega*mesh.y-phi)**2*np.cos(omega*mesh.z+phi)**2

## Compute the analitical integral
integral = (2*omega-np.sin(2*phi)-np.sin(2*(omega-phi)))*(2*omega-np.sin(2*phi)+np.sin(2*(omega+phi)))**2/(64*omega**3)

## Store variables to a pyAlya field class
field = pyAlya.FieldSOD2D(xyz=mesh.xyz,ptable=mesh.partition_table,scafi=scafi)

## Compute the integral numerically
intnum = mesh.integral(field['scafi'],kind='volume')
pyAlya.pprint(0, 'error in the integral:', np.abs(intnum-integral))