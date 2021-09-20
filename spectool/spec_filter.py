import numpy as np
from astropy.constants import c
from scipy.interpolate import interp1d
from . import convol


c = c.value


def _get_balances(func_interp, w_start, w_end, interval):
    wave = np.arange(w_start, w_end, interval)
    nflux = func_interp(wave)
    cs_nflux = np.cumsum(nflux)
    norm_csnflux = cs_nflux / cs_nflux[-1]
    cmp = np.abs(norm_csnflux-0.5)
    ret = np.min(cmp)
    return ret


def _get_half_ind(wave, flux):
    cs_flux = np.cumsum(flux)
    norm_csflux = cs_flux / cs_flux[-1]
    cmp = np.abs(norm_csflux-0.5)
    ind = np.argmin(cmp)
    return ind


def filter_use_given_profile_in_wave_space(wave, flux, wave_kernel, profile_kernel):
    """broadening the spectrum using a given profile described in wave space.

    This function try to keep the spectrum not shift but not confirm it.
    The zero point of wave_kernel is meanless, so you can use a wavelength sequence
    like [4500, 4501, 4502...], or [0, 1, 2, 3...]

    Args:
        wave (numpy.ndarray(float64)): the spectral wavelength
        flux (numpy.ndarray(float64)): the spectral flux
        wave_kernel (numpy.ndarray(float64)): the wavelength of the convol kernel
        profile_kernel (numpy.ndarray(float64)): the profile of the convol kernel

    Returns:
        outflux (numpy.ndarray(float64)): the spectral flux after the broadening
    """
    dw = wave[1] - wave[0]
    wend = wave_kernel[-1]
    beginlst = np.linspace(0, dw, 10) + wave_kernel[0]
    funinterp = interp1d(wave_kernel, profile_kernel, kind='cubic')
    balances_lst = np.array([_get_balances(funinterp, w, wend, dw) for w in beginlst])
    ind = np.argmin(balances_lst)
    wb = beginlst[ind]
    nkw = np.arange(wb, wend, dw)
    nkf = funinterp(nkw)
    nkf = nkf / np.sum(nkf)
    ind_half = _get_half_ind(nkw, nkf)
    ind_end = len(nkw) - ind_half -1
    ind_half = ind_half + len(nkw)
    ind_end = ind_end + len(nkw)
    margin_left = np.ones(len(nkw)) * nkf[0]
    margin_right = np.ones(len(nkw)) * nkf[-1]
    nflux = np.concatenate((margin_left, flux, margin_right))
    outflux = np.convolve(nflux, nkf)
    outflux = outflux[ind_half:-ind_end]
    return outflux


def filter_use_given_profile(wave, flux, velocity, profile):
    """Smooth the input spectrum using the given profile

    Args:
        wave (numpy.ndarray(float64)): spectrum wave
        flux (numpy.ndarray(float64)): spectrum flux
        velocity (numpy.ndarray(float64)): kernel velocity used to convol the spectrum
        profile (numpy.ndarray(float64)): kernel profile used to convol the spectrum

    Returns:
        numpy.ndarray(float): the spectrum flux after smooth
    """
    return convol.filter_use_given_profile(wave, flux, velocity, profile)


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


def gaussian_filter_wavespace(wave, flux, fwhm):
    """Smooth spectrum with gaussian kernel in wave space

    Args:
        wave (numpy.ndarray(float64)): -- spectrum wave
        flux (numpy.ndarray(float64)): -- spectrum flux
        fwhm (float): gaussian kernel width, in the unit of FWHM (AA)

    Returns:
        numpy.ndarray(float64): -- the spectrum flux after smooth
    """
    sigma = fwhm / 2.355
    return np.array(convol.gauss_filter_wavespace(wave, flux, sigma))


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
