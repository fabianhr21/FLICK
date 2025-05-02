import pyAlya
import numpy as np
np.set_printoptions(precision=16)

BASEDIR = '/home/benet/Dropbox/UNIVERSITAT/PhD/test_cases/mms_cube/p7'
CASESTR = 'cube_bound_N_16'

## Read mesh and extract boundary condition
mesh = pyAlya.MeshSOD2D.read(CASESTR,basedir=BASEDIR)
meshbc,_,_ = mesh.extract_bc([1])
meshbc.xyz = meshbc.xyz[:,:2] ## Remove 3rd dimension for derivatives TODO: code a better isoparametric tansformation

## Define analitical function for a scalar and a vectorial field
x0    = 0.5
omega = 3*np.pi/2
phi   = np.pi/2-omega*x0
scaf  = np.cos(omega*meshbc.x+phi)*np.sin(omega*meshbc.y-phi)
vecf  = np.transpose(np.array([scaf, meshbc.x*scaf]))

## Compute the analitical integral
integral = ((np.cos(omega - phi) - np.cos(phi)) * (np.sin(phi) - np.sin(omega + phi))) / omega**2

## Compute the analitcal gradient of the scalar field
gradf_x = omega*np.sin(phi+omega*meshbc.x)*np.sin(phi-omega*meshbc.y)
gradf_y = omega*np.cos(phi+omega*meshbc.x)*np.cos(phi-omega*meshbc.y)  
gradf   = np.transpose(np.vstack((gradf_x,gradf_y)))

## Compute the analitical divergence of the vector field
partial_x = omega*np.sin(phi+omega*meshbc.x)*np.sin(phi-omega*meshbc.y)
partial_y = omega*np.cos(phi+omega*meshbc.x)*np.cos(phi-omega*meshbc.y)*meshbc.x
div_vecf  = partial_x + partial_y

## Compute the analitical laplacian of the vector field
partial2_x = -omega**2 * np.cos(omega*meshbc.x + phi) * np.sin(omega*meshbc.y - phi)
partial2_y = -omega**2 * np.cos(omega*meshbc.x + phi) * np.sin(omega*meshbc.y - phi)
lap_vecf   = partial2_x + partial2_y

## Store variables to a pyAlya field class
field = pyAlya.FieldSOD2D(xyz=meshbc.xyz,ptable=meshbc.partition_table,scaf=scaf,gradf=gradf,vecf=vecf,div_vecf=div_vecf,lap_vecf=lap_vecf)

## Compute the integral numerically
intnum = meshbc.integral(field['scaf'])
print('error in the integral:', np.abs(intnum-integral))

#print('Interpolation error:', interp - np.cos(omega*xp+phi)*np.sin(omega*yp-phi))

## Compute the gradient numerically
field['numgrad'] = meshbc.gradient(field['scaf'])
#Compute the maximum of the errors for each component
errx = np.max(field['numgrad'][:,0] - gradf_x)
erry = np.max(field['numgrad'][:,1] - gradf_y)
#Print gradient errors
print('max error for df/dx', errx)
print('max error for df/dy', erry)

## Compute the divergence numerically
field['numdiv'] = meshbc.divergence(field['vecf'])
#Compute the maximum of the errors
err = np.max(field['numdiv'] - div_vecf)
#Print divergence errors
print('max error for div(F)', err)

## Compute the laplacian numerically
field['numlap'] = meshbc.laplacian(field['scaf'])
#Compute the maximum of the errors
err = np.max(field['numlap'] - lap_vecf)
#Print laplacian errors
print('max error for lap(F)', err)

## Output to ParaView
meshbc.write('out_surf')
field.write('out_surf',fmt='vtkh5')

## Print timmings
pyAlya.cr_info()
