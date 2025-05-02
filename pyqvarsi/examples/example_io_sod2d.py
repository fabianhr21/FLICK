import pyAlya
import numpy as np
np.set_printoptions(precision=16)

BASEDIR = '/home/benet/Dropbox/UNIVERSITAT/PhD/test_cases/mms_cube/p3'
CASESTR = 'cube_bound_N_8'

## Read mesh and extract boundary condition
mesh = pyAlya.MeshSOD2D.read(CASESTR,basedir=BASEDIR)
print(mesh.leinv)
print(mesh.leinv_linear)

## Define analitical function for a scalar and a vectorial field
x0    = 0.5
omega = 3*np.pi/2
phi   = np.pi/2-omega*x0
scaf  = np.cos(omega*mesh.x+phi)*np.sin(omega*mesh.y-phi)*np.cos(omega*mesh.z+phi)
vecf  = np.transpose(np.array([mesh.x*scaf, mesh.y*scaf, mesh.z*scaf]))
scafi = np.cos(omega*mesh.x+phi)**2*np.sin(omega*mesh.y-phi)**2*np.cos(omega*mesh.z+phi)**2

## Compute the analitical integral
integral = (2*omega-np.sin(2*phi)-np.sin(2*(omega-phi)))*(2*omega-np.sin(2*phi)+np.sin(2*(omega+phi)))**2/(64*omega**3)

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

## Compute the integral numerically
intnum = mesh.integral(field['scafi'],kind='volume')
print('error in the integral:', np.abs(intnum-integral))

## Compute the gradient numerically
field['numgrad'] = mesh.gradient(field['scaf'])
errx = np.max(field['numgrad'][:,0] - gradf_x)
erry = np.max(field['numgrad'][:,1] - gradf_y)
errz = np.max(field['numgrad'][:,2] - gradf_z)
print('max error for df/dx', errx)
print('max error for df/dy', erry)
print('max error for df/dz', errz)

## Interpolate to a point
xp     = 1/3
yp     = 1/3
zp     = 1/3
point  = np.array([[xp, yp, zp]])
interp = mesh.interpolate(point, field, fact=mesh.porder)
print('Error interpolate:', interp['scaf'] - np.cos(omega*xp+phi)*np.sin(omega*yp-phi)*np.cos(omega*zp+phi))

## Interpolate to a plane
p1 = np.array([0.55,0.1,0.1])
p2 = np.array([0.55,0.9,0.1])
p3 = np.array([0.55,0.9,0.9])
p4 = np.array([0.55,0.1,0.9])

n1 = 5
n2 = 5
planem = pyAlya.MeshSOD2D.plane(p1,p2,p4,n1,n2)
planef = mesh.interpolate(planem.xyz, field, fact=mesh.porder**2, method='femnn')
planem.write('plane')
planef.write('plane')

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
field.write('output')

## Print timmings
pyAlya.cr_info()
