import pyAlya
import numpy as np

BASEDIR = '/home/benet/Dropbox/UNIVERSITAT/PhD/test_cases/mms_parallel/p5'
CASESTR = 'cube_bound_N_8'

## Read mesh and extract boundary condition
mesh = pyAlya.MeshSOD2D.read(CASESTR,basedir=BASEDIR)

## Define analitical function for a scalar and a vectorial field
x0    = 0.5
omega = 3*np.pi/2
phi   = np.pi/2-omega*x0
scaf  = np.cos(omega*mesh.x+phi)*np.sin(omega*mesh.y-phi)*np.cos(omega*mesh.z+phi)
vecf  = np.transpose(np.array([mesh.x*scaf, mesh.y*scaf, mesh.z*scaf]))
scafi = np.cos(omega*mesh.x+phi)**2*np.sin(omega*mesh.y-phi)**2*np.cos(omega*mesh.z+phi)**2

## Compute the analitcal gradient of the scalar field
gradf_x = -omega*np.cos(omega*mesh.z+phi)*np.sin(omega*mesh.y-phi)*np.sin(omega*mesh.x+phi)
gradf_y = omega*np.cos(omega*mesh.y-phi)*np.cos(omega*mesh.x+phi)*np.cos(omega*mesh.z+phi)
gradf_z = -omega*np.cos(omega*mesh.x+phi)*np.sin(omega*mesh.y-phi)*np.sin(omega*mesh.z+phi)  
gradf   = np.transpose(np.vstack((gradf_x,gradf_y,gradf_z)))

## Compute the analitical divergence of the vector field
partial_x = np.cos(omega*mesh.x+phi)*np.cos(omega*mesh.z+phi)*np.sin(omega*mesh.y-phi)-mesh.x*omega*np.cos(omega*mesh.z+phi)*np.sin(omega*mesh.y-phi)*np.sin(omega*mesh.x+phi)
partial_y = mesh.y*omega*np.cos(omega*mesh.y-phi)*np.cos(omega*mesh.x+phi)*np.cos(omega*mesh.z+phi)+np.cos(omega*mesh.x+phi)*np.cos(omega*mesh.z+phi)*np.sin(omega*mesh.y-phi)
partial_z = np.cos(omega*mesh.x+phi)*np.cos(omega*mesh.z+phi)*np.sin(omega*mesh.y-phi)-mesh.z*omega*np.cos(omega*mesh.x+phi)*np.sin(omega*mesh.y-phi)*np.sin(omega*mesh.z+phi)
div_vecf  = partial_x + partial_y + partial_z

## Compute the analitical laplacian of the vector field
partial2_x = -omega**2*np.cos(omega*mesh.x+phi)*np.cos(omega*mesh.z+phi)*np.sin(omega*mesh.y-phi)
partial2_y = -omega**2*np.cos(omega*mesh.x+phi)*np.cos(omega*mesh.z+phi)*np.sin(omega*mesh.y-phi)
partial2_z = -omega**2*np.cos(omega*mesh.x+phi)*np.cos(omega*mesh.z+phi)*np.sin(omega*mesh.y-phi)
lap_vecf   = partial2_x + partial2_y + partial2_z

## Store variables to a pyAlya field class
field = pyAlya.FieldSOD2D(xyz=mesh.xyz,ptable=mesh.partition_table,scaf=scaf,gradf=gradf,vecf=vecf,div_vecf=div_vecf,lap_vecf=lap_vecf,scafi=scafi)

## Compute the gradient numerically
field['numgrad'] = mesh.gradient(field['scaf'])
errx = np.max(field['numgrad'][:,0] - gradf_x)
erry = np.max(field['numgrad'][:,1] - gradf_y)
errz = np.max(field['numgrad'][:,2] - gradf_z)
pyAlya.pprint(0,'max error for df/dx', errx)
pyAlya.pprint(0,'max error for df/dy', erry)
pyAlya.pprint(0,'max error for df/dz', errz)

## Compute the divergence numerically
field['numdiv'] = mesh.divergence(field['vecf'])
err = np.max(field['numdiv'] - div_vecf)
pyAlya.pprint(0,'max error for div(F)', err)

## Compute the laplacian numerically
field['numlap'] = mesh.laplacian(field['scaf'])
err = np.max(field['numlap'] - lap_vecf)
pyAlya.pprint(0,'max error for lap(F)', err)

## Print timmings
pyAlya.cr_info()
