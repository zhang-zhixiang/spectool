import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from lmfit.models import PolynomialModel
import matplotlib.pyplot as plt
from . import rebin
from . import convol


def normalize_wave(wave):
    """normalize the wavelength range to [-1, ..., 1]

    Args:
        wave (numpy.ndarray(float64)): wavelength data

    Returns:
        numpy.ndarray(float64): wavelength after the normalization
    """
    start, end = wave[0], wave[-1]
    scale = 1 / (end - start) * 1.9999
    med = (end + start) / 2
    new_wave = (wave - med) * scale
    return new_wave


def mask_wave(wave, mask=None):
    """return the index not in the mask windows

    Args:
        wave (numpy.ndarray(float64)): wavelength data
        mask (list, optional): mask windows in the format like this [[l1, r1], [l2, r2], ...]. Defaults to None.

    Returns:
        A array : where the wave not in the mask windows
    """
    if mask is None:
        return np.where(wave < np.inf)
    lw, rw = mask[0]
    arg = (wave < lw) | (wave > rw)
    for win in mask[1:]:
        lw, rw = win
        arg_tmp = (wave < lw) | (wave > rw)
        arg = arg & arg_tmp
    return np.where(arg)


def select_wave(wave, select_window=None):
    """return the index in the selected windows

    Args:
        wave (numpy.ndarray(float64)): wavelength data
        select_window (numpy.ndarray(float64), optional): select window in the format like this [[l1, r1], [l2, r2], ...]. Defaults to None.

    Returns:
        A Array: the wave index in the select windows
    """
    if select_window is None:
        return np.where(wave < np.inf)
        lw, rw = select_window[0]
        arg = (wave > lw) & (wave < rw)
    for win in select_window[1:]:
        lw, rw = win
        arg_tmp = (wave > lw) & (wave < rw)
        arg = arg & arg_tmp
    return np.where(arg)


def spec_match(wave, flux, wave_ref, flux_ref, mask=None, degree=20):
    """scale the spectrum of (wave, flux) to match the
    reference spectrum (wave_ref, flux_ref)

    Args:
        wave (numpy.ndarray(float64)): wavelength
        flux (numpy.ndarray(float64)): flux used to be scaled
        wave_ref (numpy.ndarray(float64)): wavelength of the reference spectrum
        flux_ref (numpy.ndarray(float64)): flux of the reference spectrum
        mask (list like ([[l1, r1], [l2, r2], ...add()]), optional): mask window. Defaults to None.
        degree (int, optional) : poly degree used to fit the scale

    Returns:
        numpy.ndarray(float64): flux after the scale to match the ref spectrum
    """
    fref = rebin.rebin(wave_ref, flux_ref, wave)
    wunit = normalize_wave(wave)
    iniscale = fref / flux
    iniscale = median_filter(iniscale, 10)
    mod = PolynomialModel(degree=7)
    arg = mask_wave(wave, mask)
    wunit_m = wunit[arg]
    iniscale_m = iniscale[arg]
    z = np.polyfit(wunit_m, iniscale_m, degree)
    func = np.poly1d(z)
    datascale = func(wunit)
    out = flux * datascale
    return out
    # pars = mod.guess(iniscale_m, x=wunit_m)
    # result = mod.fit(iniscale_m, params=pars, x=wunit_m)
    # datascale = mod.eval(params=result.params, x=wunit)
    # out = flux * datascale
    # return out


def continuum(wave, flux, degree=15, maxiterations=10, plot=False):
    """reduce the spectrum continuum, and return the uniform flux 
    after the continuum correction

    Arguments:
        wave {numpy.ndarray(float64)} -- spectrum wave
        flux {numpy.ndarray(float64)} -- spectrum flux

    Keyword Arguments:
        degree {int} -- legendre Polynomials order used to fit the continuum (default: {5})
        maxiterations {int} -- max iterations (in order to reject absorption line) (default: {10})
        plot {bool} -- whether plot the spectrum and continuum profile (default: {False})

    Returns:
        numpy.ndarray(float64) -- the uniform flux
    """
    # def residual(par, x, data, epsdata=None):
    #     ymodel = np.array(convol.legendre_poly(x, par))
    #     res = ymodel - data
    #     if epsdata is None:
    #         return res
    #     else:
    #         return res / epsdata

    def func(x, *par):
        return np.array(convol.legendre_poly(x, par))

    newave = normalize_wave(wave)
    newave2 = newave.copy()

    binsize = int(flux.size / 50) + 1
    newflux = median_filter(flux, binsize)
    # inipar = [1.0 for i in range(order+1)]
    inipar = np.ones(degree+1)
    # out = leastsq(residual, inipar, args=(newave, newflux))
    popt, pcov = curve_fit(func, newave, newflux, p0=inipar)
    # print(popt)
    # print(out[0])
    scale = func(newave, *popt)
    tmpuniform = newflux / scale
    std = np.std(tmpuniform)
    arg = np.where(tmpuniform > 1 - 2*std)
    newave = newave[arg]
    newflux = newflux[arg]
    size = newave.size
    count = 0
    while count < maxiterations:
        # out = leastsq(residual, out[0], args=(newave, newflux))
        # scale = np.array(convol.legendre_poly(newave, out[0]))
        popt, pcov = curve_fit(func, newave, newflux, p0=popt)
        scale = func(newave, *popt)

        tmpuniform = newflux / scale
        std = np.std(tmpuniform)
        arg = np.where(tmpuniform > 1 - 2*std)
        newave = newave[arg]
        newflux = newflux[arg]
        if newave.size == size:
            break
        else:
            size = newave.size
        count = count + 1
    # tmpscale = convol.legendre_poly(newave2, out[0])
    tmpscale = func(newave2, *popt)
    # print('loop num = ', count)
    # print(popt)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if plot is True:
        ax.plot(wave, tmpscale)
        ax.plot(wave, flux)
        plt.show()
    return flux / tmpscale