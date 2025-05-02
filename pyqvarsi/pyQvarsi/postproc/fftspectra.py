#!/usr/bin/env python
#
# pyQvarsi, utils.
#
# FFT power spectra.
#
# Last rev: 25/10/2022
from __future__ import print_function, division

import numpy as np, nfft
from scipy.signal      import periodogram, welch
from scipy.interpolate import interp1d

from ..cr              import cr

USE_PYPLOM = False
try:
    from .pyplomb.pyplomb import plomb as lombscargle
    from .pyplomb.pyplomb import filter_octave_base2 as fft_filter_octave
    USE_PYPLOM = True
except:
    from scipy.signal     import lombscargle
    from ..utils          import raiseError

    def fft_filter_octave(freq,psd,f_min=None,f_max=None,order=2):
        '''
        '''
        raiseError('Problems with pyplomb!')


@cr('fft.periodogram')
def fft_periodogram(t, y, **kwargs):
    '''
    Performs the fast fourier transform (FFT) from a set of data and
    returns its power spectra using periodogram.

    Usage:
            f,pY = fft_periodogram(t,y)

    Where:
            > t:  is the time vector of the simulation
            > y:  are the values of the function to perform FFT

    Outputs:
            > f:  frequencies
            > px: values of the FFT in the power spectra
    '''
    # Compute sampling frequency
    if 'fs' not in kwargs.keys():
        ts = t[1] - t[0]  # Sampling time
        kwargs['fs'] = 1/ts
    # Run periodogram
    f, px = periodogram(y, **kwargs)
    return f, px


@cr('fft.plomb')
def fft_plomb(t, y, freq=None, **kwargs):
    '''
    Performs the fast fourier transform (FFT) from a set of data and
    returns its power spectra using Lomb-Scargle periodogram for 
    unevenly sampled data.

    Usage:
            f,pY = fft_plomb(t,y)

    Where:
            > t:  is the time vector of the simulation
            > y:  are the values of the function to perform FFT

    Outputs:
            > f:  frequencies
            > px: values of the FFT in the power spectra
    '''
    if 'normalize' not in kwargs.keys(): kwargs['normalize'] = True
    if USE_PYPLOM:
        freq, px = lombscargle(t,y)
    else:
        # Compute sampling frequency
        if freq is None:
            ts   = 2.*np.pi*(t[1] - t[0])  # Sampling time
            freq = np.fft.fftfreq(y.size, ts)[1:]
        # Run Lomb-Scargle periodogram
        px = lombscargle(2.*np.pi*t.copy(), y.copy(), freq, **kwargs)
    return freq, px


@cr('fft.spectra')
def fft_spectra(t, y, **kwargs):
    '''
    Performs the fast fourier transform (FFT) from a set of data and
    returns its frequence spectra. Data is assumed to be equispaced.

    Usage:
            f,pY = fftSpectra(t,y,equispaced=True)

    Where:
            > t:          is the time vector of the simulation
            > y:          are the values of the function to perform FFT
            > equispaced: if dt is not constant switch to NFFT
            > resample:   resample to equispaced values
            > lowpass:    apply a lowpass filter before processing
            > windowing:  apply windowing before processing
            > downsample: downsample values before processing
            > psd:        obtain power spectrum distribution

    Outputs:
            > f:  frequencies
            > ps: values of the FFT in the power spectra
    '''
    # Recover function arguments
    equispaced = kwargs.get('equispaced', False)
    resample = kwargs.get('resample', False)
    lowpass = kwargs.get('lowpass', False)
    windowing = kwargs.get('windowing', False)
    downsample = kwargs.get('downsample', 0)
    psd = kwargs.get('psd', False)
    # Switch between the algorithms
    if equispaced or resample:
        # Here data is assumed to be equispaced or that it
        # will be resampled to become equispaced
        if resample:
            t, dt, y = _resample(t, y)
        else:
            dt = t[1]-t[0]
        # Apply filtering
        if lowpass:
            y = _low_pass_filter(y)  # Signal filtering for high frequencies
        if windowing:
            y = _window(y)          # Windowing
        # Compute power fft and associated frequencies
        yf = np.fft.fft(y)
#		yk = (1/(t_max-t_min))*np.fft.fft(y)
#		ps = np.abs(yk) ** 2 / len(yk)
        f = np.fft.fftfreq(yf.size, d=dt)
    else:
        t -= t[0]  # force to start at t=0
        # Apply filtering
        if lowpass:
            y = _low_pass_filter(y)  # Signal filtering for high frequencies
        if windowing:
            y = _window(y)          # Windowing
        # Compute sampling frequency
        k_left = (t.shape[0]-1.)/2.
        f = (np.arange(t.shape[0], dtype=np.double)-k_left)/t[-1]
        # Compute power spectra using nfft
        x = -0.5 + np.arange(t.shape[0], dtype=np.double)/t.shape[0]
        yf = nfft.nfft_adjoint(x, y, len(t))
    # Compute PSD
    ps = np.real(yf*np.conj(yf))/y.shape[0] if psd else np.abs(yf)**2

    # Downsample averaging if necessary
    if downsample > 0:
        ps = _downsample_avg(ps, downsample)
        f = _downsample_avg(f, downsample)

    # Return only positive frequencies (and associated coefficients)
    ps = ps[f >= 0]
    f = f[f >= 0]
    return f, ps


@cr('fft.spectra_Welch')
def freq_spectra_Welch(t, y, n=8, OL=0.5, use_scipy=False, **kwargs):
	"""
	Returns the FFT of y together with the associated frequency after resampling the signal evenly.
	The Welch methods splits a temporal signal into N overlapping (OL) segments. Overlapping 50%
	means that the N+1 segment contains 50% of the N segment. Recomended 50%-75% overlap.
	An FFT is computed for each segment, and the resulting spectras are averaged.
	The original time signal will be distributed in an equispaced manner, so that segments can be
	produced with equal time length (and the classic FFT, not nFFT, will be applied).

	Usage:
		f, yk = freq_spectra_Welch(t, y, n=8, OL=0.5, use_scipy=False, **kwargs)
	
	Where:
		> t: is the time series
		> y: is the signal value series
		> n: is the number of segments
		> OL: is the overlapping value, 0.0-1.0, were 0.0 means 0% overlapping, and 1.0 means 100% overlapping.
		> use_scipy: Boolean to use scipy.signal.welch (True) of the custom implementation (False)
		> kwargs:
			> lowpass: Boolean to apply a low-pass filter to y (only optional in custom implementation)
			> windowing: Boolean to apply the Hanning windowing function.

	Outputs:
		> f:  frequencies
		> ps: values of the FFT in the power spectra
	"""
	windowing = kwargs.get('windowing', False)
	psd = kwargs.get('psd', False)

	# Re-sample y on a evenly spaced time series (constant dt)
	t,dt,y = _resample(t, y)

	if use_scipy:
		nperseg  = int(y.size/n)
		noverlap = int(nperseg*OL)
		window   = "hann" if windowing else None
		scaling  = "density" if psd else "spectrum"
		f_mean, ps_mean = welch(y, fs=1/dt, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling)
	else:
		y_partial_OL_list = _split_overlap(y, n, OL)
		t_partial_OL_list = _split_overlap(t, n, OL)

		ps_partial_OL_list = []
		f_partial_OL_list = []
		for tup in list(zip(t_partial_OL_list, y_partial_OL_list)):
			f, ps = fft_spectra(tup[0], tup[1], equispaced=True, downsample=False, **kwargs)
			f_partial_OL_list.append(f)
			ps_partial_OL_list.append(ps)

		ps_mean = np.mean(ps_partial_OL_list, axis=0)
		f_mean = np.mean(f_partial_OL_list, axis=0)

	return f_mean, ps_mean


def _split_overlap(y, n, OL):
    """
    Returns a list of n segments of the signal y with OL [0,1] overlap between them.
    Values of the reminder range are not included (same as in scipy.signal.welch). Eg:
            y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            n = 2
            OL = 0.5
    Returns:
            [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]]
    """
    splits_size = int(round(y.size/n))
    nOL = int(round(splits_size * OL))
    skip = splits_size - nOL
    b = [y[i: i + splits_size] for i in range(0, len(y), skip)]
    c = []
    for i, item in enumerate(b):
        if len(item) == splits_size:
            c.append(item)
    return c


def _window(y):
	"""
	Apply the Hanning window to signal y.
	"""
	from scipy.signal.windows import hann
	w = hann(len(y))
	return y * w


def _downsample_avg(arr, n):
    """
    Average every n elements a 1D array.
    """
    end = n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)


def _downsample_simple(arr, n):
    """
    Skip n elements of a 1D array.
    """
    return arr[::n]


def _low_pass_filter(y):
	"""
	Apply a low-pass filter to y.
	"""
	from scipy.signal import butter, filtfilt
	b, a = butter(3, 0.4, 'low') # 2nd arg: Fraction of fs that wants to be filtered
	return filtfilt(b, a, y)


def _resample(t,y):
	"""
	Resample signal y to be equispaced.
	"""
	y = y - np.mean(y)
	y_function = interp1d(t, y, kind='cubic')
	t_min, t_max = np.min(t), np.max(t)
	dt = (t_max - t_min) / len(t)
	t = np.arange(t_min, t_max, dt)[:-1]  # Skip last one because can be problematic if > than actual t_max
	y = y_function(t)
	return t,dt,y