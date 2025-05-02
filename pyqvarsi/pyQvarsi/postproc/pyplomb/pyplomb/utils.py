#!/usr/bin/env python3
#
# PLOMB: LOMB-SCARGLE PERIODOGRAM
#
import numpy as np


def _fill_nan(A):
    '''
    Interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    return np.where(np.isfinite(A),A,np.interp(inds,inds[good],A[good]))