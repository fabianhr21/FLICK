import pyAlya
import numpy as np
np.set_printoptions(precision=16)

BASEDIR = '/home/benet/Dropbox/UNIVERSITAT/PhD/test_cases/mms_cube_periodic/p4'
CASESTR = 'cube_N_8'

## Read mesh and extract boundary condition
mesh = pyAlya.MeshSOD2D.read(CASESTR,basedir=BASEDIR)

## Define analitical function for a scalar and a vectorial field
x0    = 0.5
omega = 3*np.pi/2
phi   = np.pi/2-omega*x0
scaf  = np.cos(omega*mesh.x+phi)**2*np.sin(omega*mesh.y-phi)**2*np.cos(omega*mesh.z+phi)**2
vecf  = np.transpose(np.array([(mesh.x-x0)**2*scaf, (mesh.y-x0)**2*scaf, (mesh.z-x0)**2*scaf]))

## Compute the analitical integral
integral = (2*omega-np.sin(2*phi)-np.sin(2*(omega-phi)))*(2*omega-np.sin(2*phi)+np.sin(2*(omega+phi)))**2/(64*omega**3)

## Compute the analitcal gradient of the scalar field
gradf_x = -2*omega*np.cos(phi+omega*mesh.x)*np.cos(phi+omega*mesh.z)**2*np.sin(phi+omega*mesh.x)*np.sin(phi-omega*mesh.y)**2
gradf_y = -2*omega*np.cos(phi+omega*mesh.x)**2*np.cos(phi-omega*mesh.y)*np.cos(phi+omega*mesh.z)**2*np.sin(phi-omega*mesh.y)
gradf_z = -2*omega*np.cos(phi+omega*mesh.x)**2*np.cos(phi+omega*mesh.z)*np.sin(phi-omega*mesh.y)**2*np.sin(phi+omega*mesh.z)  
gradf   = np.transpose(np.vstack((gradf_x,gradf_y,gradf_z)))

## Compute the analitical divergence of the vector field
partial_x = 2*(mesh.x-x0)*np.cos(omega*mesh.x+phi)**2*np.cos(omega*mesh.z+phi)**2*np.sin(omega*mesh.y-phi)**2-2*omega*(mesh.x-x0)**2*np.cos(omega*mesh.x+phi)*np.cos(omega*mesh.z+phi)**2*np.sin(omega*mesh.y-phi)**2*np.sin(omega*mesh.x+phi)
partial_y = 2*(mesh.y-x0)*np.cos(omega*mesh.x+phi)**2*np.sin(omega*mesh.y-phi)**2*np.cos(omega*mesh.z+phi)**2+(mesh.y-x0)**2*np.cos(omega*mesh.x+phi)**2*2*np.sin(omega*mesh.y-phi)*np.cos(omega*mesh.y-phi)*omega*np.cos(omega*mesh.z+phi)**2
partial_z = 2*(mesh.z-x0)*np.cos(omega*mesh.x+phi)**2*np.sin(omega*mesh.y-phi)**2*np.cos(omega*mesh.z+phi)**2-2*omega*np.sin(omega*mesh.z+phi)*np.cos(omega*mesh.z+phi)*(mesh.z-x0)**2*np.cos(omega *mesh.x+phi)**2*np.sin(omega*mesh.y-phi)**2
div_vecf  = partial_x + partial_y + partial_z

## Compute the analitical laplacian of the vector field
partial2_x = -2*omega**2*np.cos(phi+omega*mesh.x)**2*np.cos(phi+ omega*mesh.z)**2*np.sin(phi-omega*mesh.y)**2+2*omega**2*np.cos(phi+omega*mesh.z)**2*np.sin(phi+omega*mesh.x)**2*np.sin(phi-omega*mesh.y)**2
partial2_y = 2*omega**2*np.cos(phi+omega*mesh.x)**2*np.cos(phi- omega*mesh.y)**2*np.cos(phi+omega*mesh.z)**2-2*omega**2*np.cos(phi+omega*mesh.x)**2*np.cos(phi+omega*mesh.z)**2*np.sin(phi-omega*mesh.y)**2
partial2_z = -2*omega**2*np.cos(phi+omega*mesh.x)**2*np.cos(phi+ omega*mesh.z)**2*np.sin(phi-omega*mesh.y)**2+2*omega**2*np.cos(phi+omega*mesh.x)**2*np.sin(phi-omega*mesh.y)**2*np.sin(phi+omega*mesh.z)**2

lap_vecf   = partial2_x + partial2_y + partial2_z

## Store variables to a pyAlya field class
field = pyAlya.FieldSOD2D(xyz=mesh.xyz,ptable=mesh.partition_table,scaf=scaf,gradf=gradf,vecf=vecf,div_vecf=div_vecf,lap_vecf=lap_vecf)

## Compute the integral numerically
intnum = mesh.integral(field['scaf'],kind='volume')
print(intnum)
print('error in the integral:', np.abs(intnum-integral))

## Compute the gradient numerically
field['numgrad'] = mesh.gradient(field['scaf'])
errx = np.max(field['numgrad'][:,0] - gradf_x)
erry = np.max(field['numgrad'][:,1] - gradf_y)
errz = np.max(field['numgrad'][:,2] - gradf_z)
print('max error for df/dx', errx)
print('max error for df/dy', erry)
print('max error for df/dz', errz)

## Compute the divergence numerically
field['numdiv'] = mesh.divergence(field['vecf'])
err = np.max(field['numdiv'] - div_vecf)
print('max error for div(F)', err)

## Compute the laplacian numerically
field['numlap'] = mesh.laplacian(field['scaf'])
err = np.max(field['numlap'] - lap_vecf)
print('max error for lap(F)', err)

## Output to ParaView
mesh.write('output')
field.write('output',fmt='vtkh5')

## Print timmings
pyAlya.cr_info()
