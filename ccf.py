import numpy as np
import matplotlib.pyplot as plt
from . import libccf
from . import spec_func


def iccf_spec(wave, flux, wave_ref, flux_ref, shiftlst, mask=None):
    """get the ccf result of a spectrum with the reference spectrum

    Args:
        wave (numpy.ndarray(float64)): spectrum wave
        flux (numpy.ndarray(float64)): spectrum flux
        wave_ref (numpy.ndarray(float64)): reference wave
        flux_ref (numpy.ndarray(float64)): reference flux
        shiftlst (numpy.ndarray(float64) or float list): velocity shift of the spectrum in the unit of km/s
        mask (list [[l1, r1], [l2, r2], ...], optional): mask window when computing ccf result. 
        Attention, we only mask the spectrum (wave, flux), not including
        reference spectrum (wave_ref, flux_ref) Defaults to None.

    Returns:
        Array: CCF result Array.
    """
    arg = spec_func.mask_wave(wave, mask)
    newave = wave[arg]
    newflux = flux[arg]
    result = np.array(libccf.iccf_spec(wave_ref, flux_ref, newave, newflux, shiftlst))
    return result


def get_ccf(wave1, flux1, wave2, flux2, start, end, bins, show=False):
    """get the ccf array

    Arguments:
        wave1 {np.darray} -- used as the reference
        flux1 {np.darray} -- used as the reference
        wave2 {np.darray} -- the target spectrum
        flux2 {np.darray} -- the target spectrum
        start {float} -- the velocity begin of ccf
        end {float} -- the velocity end of ccf
        bins {int} -- the bins number of the ccf

    Keyword Arguments:
        show {bool} -- whether show the ccf (default: {False})

    Returns:
        velocity {np.darray}, ccf result {np.darray} -- the ccf result
    """
    shift = np.linspace(start, end, bins)
    result = np.array(libccf.iccf_spec(wave1, flux1, wave2, flux2, shift))
    argmax = np.argmax(result)
    peak = shift[argmax]
    max008 = 0.8 * np.max(result)
    arg = np.where(result > max008)
    nshift = shift[arg]
    nresult = result[arg]
    center = np.sum(nresult * nshift) / np.sum(nresult)
    if show is True:
        plt.plot(shift, result)
        plt.axvline(peak, color='C9')
        plt.axvline(center, color='C3')
        plt.axhline(max008, color='C3')
        plt.show()
    return shift, result


def get_ccf_info(vel_shift, coef):
    """get the typical info of a ccf

    Arguments:
        vel_shift {np.darray} -- velocity array
        coef {np.darray} -- correlation coef

    Returns:
        rmax, center, peak -- float, float, float
    """
    argmax = np.argmax(coef)
    peak = vel_shift[argmax]
    rmax = np.max(coef)
    max008 = 0.8 * rmax
    arg = np.where(coef > max008)
    nshift = vel_shift[arg]
    nresult = coef[arg]
    center = np.sum(nresult * nshift) / np.sum(nresult)
    return rmax, center, peak