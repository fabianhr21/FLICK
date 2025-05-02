#!/usr/bin/env python
#
# Example showing the solver functionality
# of pyQvarsi.
#
# Last revision: 24/10/2022
from __future__ import print_function, division

import numpy as np

import pyQvarsi

## Define a linear system
A = np.array([[0.02777778, 0., 0., 0., 0.01388889, 0., 0., 0.01388889, 0.00694444],
	 		  [0., 0.02777778, 0., 0., 0.01388889, 0.01388889, 0., 0., 0.00694444],
 			  [0., 0., 0.02777778, 0., 0., 0.01388889, 0.01388889, 0., 0.00694444],
 			  [0., 0., 0., 0.02777778, 0., 0., 0.01388889, 0.01388889, 0.00694444],
 			  [0.01388889, 0.01388889, 0., 0., 0.05555556, 0.00694444, 0., 0.00694444, 0.02777778],
 			  [0., 0.01388889, 0.01388889, 0., 0.00694444, 0.05555556, 0.00694444, 0., 0.02777778],
 			  [0., 0., 0.01388889, 0.01388889, 0., 0.00694444, 0.05555556, 0.00694444, 0.02777778],
 			  [0.01388889, 0., 0., 0.01388889, 0.00694444, 0., 0.00694444, 0.05555556, 0.02777778],
 			  [0.00694444, 0.00694444, 0.00694444, 0.00694444, 0.02777778, 0.02777778, 0.02777778, 0.02777778, 0.11111111]])
Al = A.sum(axis=1)


## Scalar field
b  = np.array([-0.08333333, 0., 0., 0., -0.08333333, 0., 0., -0.04166667, -0.04166667])

# Brute force solver
x = np.matmul(np.linalg.inv(A),b)
print('Solution = ',x)

# Lumped + approximate inverse
x = pyQvarsi.solvers.solver_lumped(Al,b.copy())
x = pyQvarsi.solvers.solver_approxInverse(A,Al,x,iters=100)
print('Solution = ',x)

# Conjugate gradient
x = pyQvarsi.solvers.solver_conjgrad(A,b.copy(),iters=500)
print('Solution = ',x)


## CSR solvers
Acsr = pyQvarsi.math.csr_convert(A)
Al   = np.squeeze(A.sum(axis=1))

# Lumped + approximate inverse
x = pyQvarsi.solvers.solver_lumped(Al,b.copy())
x = pyQvarsi.solvers.solver_approxInverse(Acsr,Al,x,iters=100)
print('Solution (CSR) = ',x)

# Conjugate gradient
x = pyQvarsi.solvers.solver_conjgrad(Acsr,b.copy(),iters=500)
print('Solution (CSR) = ',x)


## Vector field
b  = np.array([[-0.08333333,-0.08333333], [0.,0.], [0.,0.], [0.,0.], [-0.08333333,-0.08333333], [0.,0.], [0.,0.], [-0.04166667,-0.04166667], [-0.04166667,-0.04166667]])


# Brute force solver
x = np.matmul(np.linalg.inv(A),b)
print('Solution = ',x)
#print('kk = ',cg(A,b))

# Lumped + approximate inverse
x = pyQvarsi.solvers.solver_lumped(Al,b.copy())
x = pyQvarsi.solvers.solver_approxInverse(A,Al,x,iters=100)
print('Solution = ',x)

# Conjugate gradient
x = pyQvarsi.solvers.solver_conjgrad(A,b.copy(),iters=500)
print('Solution = ',x)


## CSR solvers
Acsr = pyQvarsi.math.csr_convert(A)
Al   = np.squeeze(A.sum(axis=1))

# Lumped + approximate inverse
x = pyQvarsi.solvers.solver_lumped(Al,b.copy())
x = pyQvarsi.solvers.solver_approxInverse(Acsr,Al,x,iters=100)
print('Solution (CSR) = ',x)

# Conjugate gradient
x = pyQvarsi.solvers.solver_conjgrad(Acsr,b.copy(),iters=500)
print('Solution (CSR) = ',x)


pyQvarsi.cr_info()