import math
import numpy as np
from astropy.constants import c
from scipy.interpolate import interp1d
from . import convol
from . import lnspecfilter
from . import rebin


c = c.value


def rotation_filter(wave, flux, vrot, limb=0.5, flag_log=False):
    """
    Rotation filter for spectral data.

    This function applies a rotation filter to a given spectrum using the specified rotational velocity (`vrot`).
    It can either directly work with the input wavelength and flux data or interpolate the flux data onto a logarithmic wavelength scale 
    before applying the filter, depending on the value of the `flag_log` parameter.

    Parameters:
        wave (array-like): The input wavelength array (in Angstroms unit).
        flux (array-like): The input flux array corresponding to the input wavelengths.
        vrot (float): The rotational velocity (in km/s) to apply in the filter.
        limb (float, optional): The limb darkening coefficient (default is 0.5).
        flag_log (bool, optional): A flag to indicate whether to use logarithmic interpolation on the wavelength axis.
                                   If `False`, logarithmic resampling is applied. If `True`, the filter is applied directly
                                   to the input flux data (default is `False`).

    Returns:
        array-like: The flux array after applying the rotation filter.

    Notes:
        - When `flag_log` is `False`, the function interpolates the input flux onto a logarithmic wavelength grid, 
          applies the rotation filter, and then re-bins the result back to the original wavelength grid.
        - When `flag_log` is `True`, the function directly applies the rotation filter to the input flux without interpolation.

    Example:
        wave = np.array([4000, 5000, 6000])
        flux = np.array([1.0, 0.9, 0.8])
        vrot = 100.0
        filtered_flux = rotation_filter(wave, flux, vrot)
    """
    if flag_log == False:
        wave_min, wave_max = np.min(wave), np.max(wave)
        nwave = np.logspace(np.log10(wave_min), np.log10(wave_max), len(wave), endpoint=True)
        nflux = rebin.rebin_padvalue(wave, flux, nwave)
        dll = math.log(nwave[1]/nwave[0])
        flux_broad = np.array(lnspecfilter.rotation_filter(nflux, dll, vrot, limb))
        flux_out = rebin.rebin_padvalue(nwave, flux_broad, wave)
        return flux_out
    dll = math.log(wave[1]/wave[0])
    return np.array(lnspecfilter.rotation_filter(flux, dll, vrot, limb))


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
    """
    Filter the provided flux using a given profile in the wave space, applying a convolution with the specified wave and profile kernels.

    Parameters:
        wave (numpy.ndarray): The wavelength array that defines the wave space.
        flux (numpy.ndarray): The flux values corresponding to the wavelengths.
        wave_kernel (numpy.ndarray): The wavelength kernel used to define the filtering profile.
        profile_kernel (numpy.ndarray): The profile kernel used to define the shape of the filter.

    Returns:
        numpy.ndarray: The filtered flux after applying the convolution with the profile kernel in wave space.

    Note:
        This function applies a cubic interpolation to the profile kernel and convolves the flux with the resulting kernel. The output flux is trimmed based on the convolution margin.
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
        wave (numpy.ndarray(float64)): spectrum wave in angstrom
        flux (numpy.ndarray(float64)): spectrum flux
        velocity (numpy.ndarray(float64)): kernel velocity used to convol the spectrum (unit: km/s)
        profile (numpy.ndarray(float64)): kernel profile used to convol the spectrum

    Returns:
        numpy.ndarray(float): the spectrum flux after smooth
    """
    return convol.filter_use_given_profile(wave, flux, velocity, profile)


def gaussian_filter(wave, flux, velocity):
    """
    Apply a Gaussian filter to the given flux data based on the provided velocity.

    Parameters:
        wave (numpy.ndarray): The wavelength values of the spectrum (unit: AA).
        flux (numpy.ndarray): The flux values corresponding to the wavelength values.
        velocity (float): The velocity value to calculate the Gaussian filter width (unit: km/s).

    Returns:
        numpy.ndarray: The flux data after applying the Gaussian filter.

    Notes:
        The filter width is computed from the velocity using the formula:
            width = velocity / (2.355 * c) * 1000
        where c is the speed of light in km/s.
    """
    par = velocity / (2.355*c) * 1000
    pararr = np.array([0.0, par])
    return np.array(convol.gauss_filter(wave, flux, pararr))


def gaussian_filter_wavespace(wave, flux, fwhm):
    """
    Applies a Gaussian filter in the wave space to smooth the given flux data.

    Parameters:
        wave (array-like): The input wave array (unit: AA).
        flux (array-like): The flux values corresponding to the wave array.
        fwhm (float): Full width at half maximum (FWHM) of the Gaussian filter, which determines the width of the filter (unit: km/s).

    Returns:
        numpy.ndarray: The filtered flux array after applying the Gaussian filter.

    Note:
        The FWHM is converted to the standard deviation (sigma) using the relation:
        sigma = FWHM / 2.355.
    """
    sigma = fwhm / 2.355
    return np.array(convol.gauss_filter_wavespace(wave, flux, sigma))


def gauss_filter_mutable(wave, flux, arrvelocity):
    """
    Applies a Gaussian filter to the given wave and flux data, with a specified velocity array.

    Parameters:
        wave (array-like): The input array representing the wavelength values (unit: AA).
        flux (array-like): The input array representing the flux values corresponding to the wavelengths.
        arrvelocity (array-like): The array of kernel velocities at each wavelength point (unit: km/s).

    Returns:
        numpy.ndarray: The result of applying the Gaussian filter to the input spectrum.
    """
    return np.array(convol.gauss_filter_mutable(wave, flux, arrvelocity))
