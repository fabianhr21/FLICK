import numpy as np
from ..cr                  import cr

def lagrange(r, kpoint, isopoints):
	prod  = 1
	for ipoint, isopoint in enumerate(isopoints):
		if ipoint != kpoint:
			prod = prod*(r - isopoint)/(isopoints[kpoint] - isopoint)
	return prod

def dlagrange(r, kpoint, isopoints):
	sum  = 0
	for ipoint, isopoint in enumerate(isopoints):
		prod = 1
		if ipoint != kpoint:    
			for jpoint, isopoint2 in enumerate(isopoints):
				if jpoint != kpoint and jpoint != ipoint:
					prod *= (r - isopoint2)/(isopoints[kpoint] - isopoint2)
			sum  +=  prod/(isopoints[kpoint] - isopoint)
	return sum

def legendre(p, x):
    # Variable declaration
    lp = 0.0
    # Accumulator loop
    for k in range(p+1):
        lp += np.math.comb(p, k) * np.math.comb(p+k, k) * (0.5*(x-1.0))**k
    # Return
    return lp

def dlegendre(p, x):
    # Avoid zero singularity
    if np.isclose(x, 1.0):
        x += np.finfo(x).eps
    # Variable declaration
    lp = 0.0
    # Accumulator loop
    for k in range(p+1):
        lp += np.math.comb(p, k) * np.math.comb(p+k, k) * 0.5*k*(0.5*(x-1.0))**(k-1.0)
    # Return
    return lp

def d2legendre(p, x):
    # Avoid zero singularity
    if np.isclose(x, 1.0):
        x += np.finfo(x).eps
    # Variable declaration
    lp = 0.0
    # Accumulator loop
    for k in range(p+1):
        lp += np.math.comb(p, k) * np.math.comb(p+k, k) * 0.25*k*(k-1.)*(0.5*(x-1.0))**(k-2.0)
    # Return
    return lp

def d3legendre(p, x):
    # Avoid zero singularity
    if np.isclose(x, 1.0):
        x += np.finfo(x).eps
    # Variable declaration
    lp = 0.0
    # Accumulator loop
    for k in range(p+1):
        lp += np.math.comb(p, k) * np.math.comb(p+k, k) * 0.125*k*(k-1.)*(k-2.)*(0.5*(x-1.0))**(k-3.0)
    # Return
    return lp

def halley(x, fun, dfun, d2fun, tol, niter):
    y = fun(x)
    if abs(y) < tol:
        return x, abs(y)
    for ite in range(niter):
        yp = dfun(x)
        ypp = d2fun(x)
        denominator = 2.0 * yp**2 - y * ypp
        xn = x - 2.0 * y * yp / denominator
        y = fun(xn)
        if abs(y) < tol:
            x = xn
            break
        x = xn
    return x, abs(y)

def quadrature_GaussLobatto(p, tol=1e-16, niter=5):
    # A polynomial of order p has p roots
    xi  = np.zeros(p, dtype=np.double)
    wi  = np.zeros(p, dtype=np.double)
    ts  = np.zeros(p, dtype=np.double)
    # Loop on the roots
    dlp1  = lambda x: dlegendre(p-1, x)
    d2lp1 = lambda x: d2legendre(p-1, x)
    d3lp1 = lambda x: d3legendre(p-1, x)
    for k in range(1,p-1):
        # Functions
        xi[k] = -(1.0 - (3.0 * (p - 2.0)) / (8.0 * (p - 1.0) * (p - 1.0) * (p - 1.0))) *np.cos((4.0 * (k+1) - 3.0)*np.pi / (4.0 * (p - 1.0) + 1.0))
        xi[k], ts[k] = halley(xi[k], dlp1, d2lp1, d3lp1, tol, niter)
    for k in range(1,p-1):
        abxk = np.abs(xi[k])
        for i in range(1,p-1):
            if i != k:
                abxi = np.abs(xi[i])
                if np.round(abxi,10) == np.round(abxk,10):
                    if ts[k] < ts[i]:
                        xi[k] = -xi[i]
    for k in range(1,p-1):
        Pxi   = legendre(p - 1, xi[k])
        wi[k] = 2.0 / (p * (p - 1.0) * Pxi * Pxi)
    # Extremes
    xi[0] = -1.0
    wi[0] = 2.0 / (p * (p - 1.0))
    xi[p-1] = 1.0
    wi[p-1] = 2.0 / (p * (p - 1.0))
    # Round zeros
    xi[np.abs(xi) < 10.0 * np.finfo(xi.dtype).eps] = 0.0
    # Reorder xi and wi so that xi = xi [0], xi[-1], xi[1:-1] as it is done in SOD2D
    xi_reorder = np.hstack((xi[0], xi[-1], xi[1:-1]))
    wi_reorder = np.hstack((wi[0], wi[-1], wi[1:-1]))
    return xi_reorder, wi_reorder
