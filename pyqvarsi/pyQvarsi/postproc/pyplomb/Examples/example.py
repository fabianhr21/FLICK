#!/usr/bin/env python3
#
# Example using the plomb tool
#
import os, numpy as np, matplotlib.pyplot as plt
import pyplomb

# Obtain path to file
PATH = os.path.dirname(os.path.abspath(__file__))

# Load data and sample output
data = np.genfromtxt(os.path.join(PATH,'data.txt'))
time, u = data[:,0], data[:,1]

data = np.genfromtxt(os.path.join(PATH,'output.txt'))
f, p = data[:,0], data[:,1]

# Compute Lomb-Scargle periodogram
ff, pp = pyplomb.plomb(time,u)
fff, ppp = pyplomb.filter_octave_base2(ff,pp,order=3)

plt.figure()
plt.loglog(f,p,'r')
plt.loglog(ff,pp,'b')
plt.loglog(fff,ppp,'g')
plt.xlabel('f [Hz]')
plt.ylabel('|P1(f)|')

plt.savefig("example.png",dpi=300)