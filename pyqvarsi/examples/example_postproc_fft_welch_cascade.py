#!/usr/bin/env python
#
# Example how to perform FFT analysis with the Welch method on a list of temporal signals.
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
	print(label)
	t = lift_data[:, j * 2]
	y = lift_data[:, j * 2 + 1]
	cut = np.where(np.isnan(t))[0]
	cut = cut[0] if cut.size > 0 else None
	t, y = t[:cut], y[:cut]
	f, ps = pyQvarsi.postproc.freq_spectra_Welch(t, y, n=2, OL=0.5, windowing=True, lowpass=True, psd=True)
	Re_dict[label] = (t, y, f, ps)

# Plot cascade-like for all cases.
fig, ax = plt.subplots(1,1,facecolor='w',edgecolor='k')
for i,(k,v) in enumerate(Re_dict.items()):
	f = v[2]
	ps = v[3] * 10 ** (-3*i)
	ax.loglog(f, ps, lw=1.5, label=k)

ax.set_xlabel('f')
ax.set_ylabel('|PS(f)|')
ax.tick_params(bottom="on", top="on", which='both')
leg = plt.legend(loc='lower left')
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_facecolor('white')
leg.get_frame().set_linewidth(0.5)

plt.show()

pyQvarsi.cr_info()
