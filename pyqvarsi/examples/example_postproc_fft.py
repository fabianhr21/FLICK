#!/usr/bin/env python
#
# Example how to perform FFT analysis.
#
# Last revision: 21/06/2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import pyQvarsi


## Parameters
Fs = 1000.  # Sampling frequency
T  = 1/Fs   # Sampling period
L  = 1500   # Number of samples


## Build a noisy signal
t = np.linspace(0,np.pi/2,L)#np.arange(L)*T
S = 0.7*np.sin(50*2.*np.pi*t) + np.sin(120*2.*np.pi*t)
X = S + 2*np.random.randn(t.size)


## Compute FFT in different ways
f1,p1 = pyQvarsi.postproc.fft_spectra(t,X,psd=True)
f2,p2 = pyQvarsi.postproc.fft_periodogram(t,X)
f3,p3 = pyQvarsi.postproc.fft_plomb(t,X,freq=np.linspace(0.01,150,500))


## Plot
fig, ax = plt.subplots(2,2,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k',gridspec_kw={'hspace':0.25,'wspace':0.25})

# Signal plot
ax[0,0].plot(1000*t,X,'k')
ax[0,0].set_xlim([0,50])
ax[0,0].set_xlabel('time [msec]')
ax[0,0].set_ylabel('Y(t)')

# FFT spectrum plot
ax[0,1].plot(f1,p1,'k')
ax[0,1].set_xlim([0,150])
ax[0,1].set_xlabel('f [Hz]')
ax[0,1].set_ylabel('|P1(f)|')

# FFT spectrum plot
ax[1,0].plot(f2,p2,'k')
ax[1,0].set_xlim([0,150])
ax[1,0].set_xlabel('f [Hz]')
ax[1,0].set_ylabel('|P1(f)|')

# FFT spectrum plot
ax[1,1].plot(f3,p3,'k')
ax[1,1].set_xlim([0,150])
ax[1,1].set_xlabel('f [Hz]')
ax[1,1].set_ylabel('|P1(f)|')


pyQvarsi.cr_info()
plt.show()
