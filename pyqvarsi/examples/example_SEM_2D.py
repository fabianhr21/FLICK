import numpy as np
import pyAlya

gmshCoords = np.array([[0,0],[1,0],[0,1],[1,1]])
ltype      = np.array([12])
lninv      = np.array([0,1,2,3])

element = pyAlya.FEM.LinearQuadrangle(lninv, ngauss=1)

print(element)