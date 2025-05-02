import pyAlya
import numpy as np
import matplotlib.pyplot as plt

porders = np.array([2,3,4,5,6,7])
xs      = np.linspace(-1,1,1000)

plt.figure()
for porder in porders:
    leg = np.zeros(xs.shape)
    for ix, x in enumerate(xs):
        leg[ix] = pyAlya.FEM.legendre(porder, x)
    plt.plot(xs, leg, label='porder %i' % porder)
plt.title('Legendre polynomials')
plt.legend()

plt.figure()
for porder in porders:
    leg = np.zeros(xs.shape)
    for ix, x in enumerate(xs):
        leg[ix] = pyAlya.FEM.dlegendre(porder, x)
    plt.plot(xs, leg, label='porder %i' % porder)
plt.title('1st Derivative of Legendre polynomials')
plt.legend()

plt.figure()
for porder in porders:
    leg = np.zeros(xs.shape)
    for ix, x in enumerate(xs):
        leg[ix] = pyAlya.FEM.d2legendre(porder, x)
    plt.plot(xs, leg, label='porder %i' % porder)
plt.title('2nd Derivative of Legendre polynomials')
plt.legend()

plt.figure()
for porder in porders:
    leg = np.zeros(xs.shape)
    for ix, x in enumerate(xs):
        leg[ix] = pyAlya.FEM.d3legendre(porder, x)
    plt.plot(xs, leg, label='porder %i' % porder)
plt.title('3rd Derivative of Legendre polynomials')
plt.legend()

for porder in porders:
    xi, wi = pyAlya.FEM.quadrature_GaussLobatto(porder+1)
    print('porder %i' % porder)
    print('xi', xi)
    print('wi', wi)
    print('----------------------------------------------------------------')

plt.figure()
for ixi in range(xi.shape[0]):
    lag = np.zeros(xs.shape)
    for ix, x in enumerate(xs):
        lag[ix] = pyAlya.FEM.lagrange(x, ixi, xi)
    plt.plot(xs, lag, label=ixi)
plt.title('Shape functions for pOrder 7')
plt.legend()

plt.figure()
for ixi in range(xi.shape[0]):
    lag = np.zeros(xs.shape)
    for ix, x in enumerate(xs):
        lag[ix] = pyAlya.FEM.dlagrange(x, ixi, xi)
    plt.plot(xs, lag, label=ixi)
plt.title('Shape function derivatives for pOrder 7')
plt.legend()

pyAlya.cr_info()
plt.show()