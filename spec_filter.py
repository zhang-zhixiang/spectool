import numpy as np
from astropy.constants import c
from . import convol


c = c.value


def gaussian_filter(wave, flux, velocity):
    """Smooth spectrum with gaussian kernel

    Arguments:
        wave {numpy.ndarray(float64)} -- spectrum wave
        flux {numpy.ndarray(float64)} -- spectrum flux
        velocity {float} -- gaussian kernel width,
        in the unit of FWHM (km/s)

    Returns:
        numpy.ndarray(float64) -- the spectrum flux after smooth
    """
    par = velocity / (2.355*c) * 1000
    pararr = np.array([0.0, par])
    return np.array(convol.gauss_filter(wave, flux, pararr))


def gauss_filter_mutable(wave, flux, arrvelocity):
    """Smooth spectrum using gaussian kernel, where the kernel velocity in each wavelength can be different

    Args:
        wave (numpy.ndarray): spectrum wavelength
        flux (numpy.ndarray): spectrum flux
        arrvelocity (numpy.ndarray): kernel velocity at each wavelength

    Returns:
        numpy.ndarray: the spectrum after the smooth
    """
    return np.array(convol.gauss_filter_mutable(wave, flux, arrvelocity))