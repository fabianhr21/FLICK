#!/usr/bin/env python
#
# pyQvarsi, postprocessing Module.
#
# Module to compute postprocessing quantities
# from the flow.
#
# Last rev: 17/03/2021

__VERSION__ = 3.1

from .averaging        import midlineAvg, directionAvg, planeAvg
from .vortex_detection import vorticity, QCriterion, Lambda2Criterion, OmegaCriterion, RortexCriterion, OmegaRortexCriterion
from .fftspectra       import fft_spectra, fft_periodogram, fft_plomb, freq_spectra_Welch, fft_filter_octave
from .yplus            import yplus_xyz_3D, computeWallDistancesSOD2D

del averaging, fftspectra, vortex_detection, yplus