#!/usr/bin/env python3
#
# PLOMB: LOMB-SCARGLE PERIODOGRAM
#
import numpy as np

from .utils import _fill_nan

def filter_octave_base2(freq,psd,f_min=None,f_max=None,order=2):
    '''
    '''
    if f_min is None: f_min = freq[0]
    if f_max is None: f_max = freq[-1]

    Xmin = np.log2(f_min)*order
    Xmax = np.log2(f_max)*order
    
    fc = 2.**(np.arange(Xmin,Xmax)/order)
    fd = 2.**(1./(2.*order))
    fu = fc * fd
    fl = fc / fd

    out = np.zeros_like(fc)
    j, a = 0, 0.
    for i in range(len(fc)):
        while freq[j] > fl[i] and freq[j] <= fu[i] or j == 0:
            out[i] += psd[j]
            j      += 1
            a      += 1
            if j == len(freq): break

        out[i] /= a
        a = 0

    return fc, _fill_nan(out)