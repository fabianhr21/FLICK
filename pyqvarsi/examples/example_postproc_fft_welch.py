#!/usr/bin/env python
#
# Example how to perform FFT analysis with the Welch method.
#
# Last revision: 21/06/2021
from __future__ import print_function, division

import os
import numpy as np
import matplotlib.pyplot as plt
import pyQvarsi

# Constants
path = os.path.dirname(os.path.abspath(__file__))
lift_file = os.path.join(path, "data/DRL_signals/lift.csv")

# Read data
# lift
lift_data = np.genfromtxt(lift_file, delimiter=';', skip_header=1, usecols=(0,1,3,4,6,7,9,10))
lift_labels = ["baseline", "2000", "1000", "100"]
Re_dict = {}
for (j, label) in enumerate(lift_labels):
	t = lift_data[:, j * 2]
	y = lift_data[:, j * 2 + 1]
	cut = np.where(np.isnan(t))[0]
	cut = cut[0] if cut.size > 0 else -1
	Re_dict[label] = (t[:cut], y[:cut])

# Typical PS
f0, ps0 = pyQvarsi.postproc.fft_spectra(Re_dict["baseline"][0],Re_dict["baseline"][1],
	windowing=False, lowpass=False, psd=True)
# Custom implementation with low-pass filter
f1, ps1 = pyQvarsi.postproc.freq_spectra_Welch(Re_dict["baseline"][0],Re_dict["baseline"][1], n=4, OL=0.5, 
	use_scipy=False, windowing=True, lowpass=True, psd=True)
# Custom implementation with without filter
f2, ps2 = pyQvarsi.postproc.freq_spectra_Welch(Re_dict["baseline"][0],Re_dict["baseline"][1], n=4, OL=0.5, 
	use_scipy=False, windowing=True, lowpass=False, psd=True)
# Scipy implementation
f3, ps3 = pyQvarsi.postproc.freq_spectra_Welch(Re_dict["baseline"][0],Re_dict["baseline"][1], n=4, OL=0.5, 
	use_scipy=True, windowing=True, lowpass=False, psd=True)

# Plot data
fig, ax = plt.subplots(1,4,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k',gridspec_kw={'hspace':0.25,'wspace':0.25})

# Show lines
ax[0].loglog(f0, ps0, color='blue')
ax[1].loglog(f1, ps1, color='orange')
ax[2].loglog(f2, ps2, color='black')
ax[3].loglog(f3, ps3, color='grey')

# Add plot data
ax[0].set_title("No Welch (w/o low-pass, w/o wind.)")
ax[1].set_title("Welch (with low-pass & wind.)")
ax[2].set_title("Welch (w/o low-pass, with wind.)")
ax[3].set_title("Scipy Welch (w/o low-pass, with wind.)")
ax[0].set_xlabel('f')
ax[1].set_xlabel('f')
ax[2].set_xlabel('f')
ax[3].set_xlabel('f')
ax[0].set_ylabel('|PS(f)|')
ax[1].set_ylabel('|PS(f)|')
ax[2].set_ylabel('|PS(f)|')
ax[3].set_ylabel('|PS(f)|')
ax[0].set_ylim([1e-12,1e5])
ax[1].set_ylim([1e-12,1e5])
ax[2].set_ylim([1e-12,1e5])
ax[3].set_ylim([1e-12,1e5])
ax[0].tick_params(bottom="on", top="on", which='both')
ax[1].tick_params(bottom="on", top="on", which='both')
ax[2].tick_params(bottom="on", top="on", which='both')
ax[3].tick_params(bottom="on", top="on", which='both')


plt.show()

pyQvarsi.cr_info()
