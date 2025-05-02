
#!/usr/bin/env python3
#
# PLOMB: LOMB-SCARGLE PERIODOGRAM
#
import numpy as np

from .cy_plomb import cy_plomb
from .utils    import _fill_nan


def plomb(t, y, ofac=4, hifac=1):
    '''
    freq,psd = plomb(t,x) returns the Lomb-Scargle power spectral density (PSD) 
    estimate, pxx, of a signal, x, that is sampled at the instants specified in t. 
    t must increase monotonically but need not be uniformly spaced. 
    All elements of t must be nonnegative. pxx is evaluated at the frequencies 
    returned in f.
    '''
    # Ensure that the arrays are C contiguous in memory
    # otherwise it is a disaster
    f,s = cy_plomb(np.ascontiguousarray(t),np.ascontiguousarray(y),ofac=ofac,hifac=hifac)
    # Also make sure to eliminate any NAN coming from the computation
    return f, _fill_nan(s)