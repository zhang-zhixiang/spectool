from typing_extensions import IntVar
import numpy as np
from numpy.ma import clump_unmasked
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy import special
from lmfit.models import PolynomialModel
import matplotlib.pyplot as plt
from astropy.constants import c
from . import rebin
from . import convol
from . import spec_filter


c = c.value


def get_FWHM(wave, flux, winl, winr, plot=False):
    argl = select_wave(wave, winl)
    argr = select_wave(wave, winr)
    wl = np.median(wave[argl])
    fl = np.median(flux[argl])
    wr = np.median(wave[argr])
    fr = np.median(flux[argr])
    k = (fr - fl) / (wr - wl)
    # y = k * x + y0 --> y0 = y - k * x
    y0 = fl - k * wl
    arg = select_wave(wave, [[wl, wr]])
    nwave = wave[arg]
    nflux = flux[arg]
    cont = k * nwave + y0
    fline = nflux - cont
    dense_wave = np.linspace(nwave[0], nwave[-1], len(nwave)*100)
    dense_flux = np.interp(dense_wave, nwave, fline)
    fmax = np.max(dense_flux)
    fmax_2 = fmax / 2
    arg = dense_flux > fmax_2
    w_sel = dense_wave[arg]
    w_sel_l = w_sel[0]
    w_sel_r = w_sel[-1]
    fwhm = w_sel_r - w_sel_l
    if plot is True:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.plot(wave, flux)
        ax1.plot(nwave, cont)
        v1 = winl[0][0]
        v2 = winl[0][1]
        v3 = winr[0][0]
        v4 = winr[0][1]
        ax1.axvline(v1, color='C3', linestyle=':')
        ax1.axvline(v2, color='C3', linestyle=':')
        ax1.axvline(v3, color='C3', linestyle=':')
        ax1.axvline(v4, color='C3', linestyle=':')
        ax1.scatter([wl, wr], [fl, fr], color='red')
        ax2.plot(nwave, fline)
        ax2.axhline(fmax, color='C2', linestyle=':')
        ax2.axhline(fmax_2, color='red', linestyle=':')
        ax2.axhline(0, color='C2', linestyle=':')
        ax2.axvline(w_sel_l, color='C4', linestyle=':')
        ax2.axvline(w_sel_r, color='C4', linestyle=':')
        plt.show()
    return fwhm


def get_linear_continuum(wave, flux, win1, win2):
    """get a linear continuum from the spectrum

    Args:
        wave (numpy.ndarray): wavelength of the spectrum
        flux (numpy.ndarray): flux of the spectrum
        win1 ([[float, float]]): left continuum window
        win2 ([[float, float]]): right continuum window

    Returns:
        numpy.ndarray: the linear continuum of the spectrum
    """
    arg1 = select_wave(wave, win1)
    arg2 = select_wave(wave, win2)
    w1 = np.median(wave[arg1])
    w2 = np.median(wave[arg2])
    f1 = np.median(flux[arg1])
    f2 = np.median(flux[arg2])
    slop = (f2 - f1) / (w2 - w1)
    continuum = slop * (wave - w1) + f1
    return continuum


def get_SNR(flux):
    """estimate the SNR of a spectrum

    Args:
        flux (numpy.ndarray): spectral flux

    Returns:
        float: the SNR of the flux
    """
    signal = np.median(flux)
    n = len(flux)
    noise = 0.6052697 * np.median(np.abs(2*flux[2:n-2] - flux[0:n-4] - flux[4:n]))
    snr = signal / noise
    return snr


def plot_spec(wave, flux, ax=None, select_wins=None, mask_wins=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(wave, flux)
    yl, yr = ax.get_ylim()
    if select_wins is not None:
        for ind, win in enumerate(select_wins):
            if ind == 0:
                ax.fill_between(win, [yl, yl], [yr, yr], color='blue', alpha=0.3, label='select win')
            else:
                ax.fill_between(win, [yl, yl], [yr, yr], color='blue', alpha=0.3)
    if mask_wins is not None:
        for ind, win in enumerate(mask_wins):
            if ind == 0:
                ax.fill_between(win, [yl, yl], [yr, yr], color='grey', alpha=0.3, label='mask win')
            else:
                ax.fill_between(win, [yl, yl], [yr, yr], color='grey', alpha=0.3)
    ax.legend()
    return ax


def scombine(wavelst, fluxlst, errlst=None, new_wave=None, method='weight', reject=None):
    """combine spectra to one spectrum

    Args:
        wavelst ([np.ndarray, ndarray, ...]): the wavelength list of a series of spectra
        fluxlst ([np.ndarray, ndarray, ...]): the flux list of a series of spectra
        errlst ([np.ndarray, ndarray, ...], optional): the err list of a series of spectra.
            If errlst = None, the function will regard all err = 1. 
            Defaults to None.
        new_wave (np.ndarray, optional): the wavelength of the spectrum after the combination. 
            If new_wave = None, wavelst[0] will be used as the base wavelength to rebin. 
            Defaults to None.
        method (str, optional): The method of how to combine the spectra.
            The methods include: 'weight', 'sum', 'mean', 'median'. 
            Defaults to 'weight'.
        reject (str, optional): How to reject the spectra when doing the combination.
            The values allowed include: 'minmax', '3sigma'. Defaults to None.

    Returns:
        outwave, outflux, outerr: the wave, flux and err after the spectra combination
    """
    if new_wave is None:
        new_wave = wavelst[0]
    if errlst is None:
        errlst = []
        for wave in wavelst:
            errlst.append(np.ones(wave.shape, dtype=np.float64))
    nfluxlst = []
    nerrlst = []
    nivarlst = []
    for ind, wave in enumerate(wavelst):
        flux = fluxlst[ind]
        err = errlst[ind]
        nflux = np.array(rebin.rebin_padvalue(wave, flux, new_wave))
        nerr = np.array(rebin.rebin_err(wave, err, new_wave))
        nfluxlst.append(nflux)
        nerrlst.append(nerr)
        # nivarlst.append(1 / nerr)
    nfluxlst = np.array(nfluxlst)
    nerrlst = np.array(nerrlst)
    if method == 'sum':
        sumflux = np.sum(nfluxlst, axis=0)
        sumerr = np.sqrt(np.sum(nerrlst**2, axis=0))
        return new_wave, sumflux, sumerr
    if method == 'weight':
        sumflux = np.sum(nivarlst * nfluxlst, axis=0)
        sumweight = np.sum(nivarlst, axis=0)
        sumweight[sumweight == 0] = 1
        outflux = sumflux / sumweight
        newivar = np.sum(nivarlst, axis=0) / np.sqrt(len(nivarlst))
        arg = np.where(newivar == 0)
        newivar[arg] = 1
        outerr = 1 / newivar
        outerr[arg] = np.inf
    return new_wave, outflux, outerr


def air2vac(wave):
    """convert the wavelength from air to vacuum

    Args:
        wave (numpy.ndarray): wave data

    Returns:
        numpy.ndarray: the wavelength in vacuum
    """
    coef = 6.4328e-5 + 2.94981e-2/(146-(1.0e4/wave)**2) + 2.554e-4/(41-(1.0e4/wave)**2)
    return (1+coef) * wave


def median_reject_cos(flux, size=21):
    """reject the cosmic ray using the median filter

    Args:
        flux (numpy.ndarray): spectral flux
        size (int, optional): median filter size. Defaults to 21.

    Returns:
        new_flux(numpy.ndarray): the spectral flux after removing the cosmic ray
    """
    med_flux = median_filter(flux, size=size)
    res = flux - med_flux
    std = np.std(res)
    arg = np.where(res > 3*std)
    new_flux = flux
    new_flux[arg] = med_flux[arg]
    return new_flux


def shift_wave(wave, shift):
    """shift the wavelength in the unit of km/s

    Arguments:
        wave {numpy.ndarray(float64)} -- wave arr
        shift {float} -- unit = km/s

    Returns:
        numpy.darray(float64) -- the wave after wavelength shift
    """
    waveshift = shift * wave * 1000 / c
    newave = wave + waveshift
    return newave


def shift_wave_mutable(wave, arrpar, shifttype='velocity'):
    """shift the wavelength, the delta_lambda can be different in each wave

    Args:
        wave (numpy.ndarray): wave arr
        arrpar (list or numpy.ndarray): shift parameters
        shifttype (str, optional): in the unit of 'velocity' or 'lambda'. Defaults to 'velocity'.

    Returns:
        numpy.ndarray: the wave after the wavelength shift
    """
    waveshift = np.zeros(wave.size)
    tmpwave = wave / 10000
    for ind, val in enumerate(arrpar):
        waveshift = waveshift + (tmpwave**ind) * val
    if shifttype == 'velocity':
        waveshift = waveshift * wave * 1000 / c
    newave = wave + waveshift
    return newave


def legendre_polynomial(x, arrpar):
    """get legendre polynomial values, x should be in the range of [-1, 1]

    Args:
        x (numpy.ndarray): Array
        arrpar (list or numpy.ndarray): parameter of each degree

    Returns:
        numpy.ndarray: legendre polynomial values
    """
    return np.array(convol.legendre_poly(x, arrpar))


def get_unit(data):
    exp = int(np.log10(np.median(data)))
    return 10**exp


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


def get_scale(wave, scalepar):
    """return the Legendre polynomial, which can be used to scale a spectrum

    Args:
        wave (numpy.ndarray): wavelength
        scalepar (list like array): the coefficients of the Legendre polynomial, 
          the format is like [1.0, -0.5, 0.25, -0.125], the length of scalepar 
          is the degree of Legendre polynomial

    Returns:
        numpy.ndarray: the scale array calculated using Legendre polynomial
    """
    norm_wave = normalize_wave(wave)
    scale = np.zeros(norm_wave.shape)
    for nn, value in enumerate(scalepar):
        scale = scale + special.eval_legendre(nn, norm_wave) * value
    return scale


def mask_wave(wave, mask=None):
    """return the index not in the mask windows

    Args:
        wave (numpy.ndarray(float64)): wavelength data
        mask (list, optional): mask windows in the format like this [[l1, r1], [l2, r2], ...]. Defaults to None.

    Returns:
        A array : where the wave not in the mask windows
    """
    if mask is None:
        return wave < np.inf
    lw, rw = mask[0]
    arg = (wave < lw) | (wave > rw)
    for win in mask[1:]:
        lw, rw = win
        arg_tmp = (wave < lw) | (wave > rw)
        arg = arg & arg_tmp
    return arg


def get_delta_wave(wave):
    """return the bin width of each wavelength points

    Args:
        wave (numpy.ndarray): wavelength array

    Returns:
        widths(numpy.ndarray): bin width array
    """
    dif = np.diff(wave)
    dif1 = np.append([dif[0]], dif) / 2
    dif2 = np.append(dif, [dif[-1]]) / 2
    return dif1 + dif2


def select_wave(wave, select_window=None):
    """return the index in the selected windows

    Args:
        wave (numpy.ndarray(float64)): wavelength data
        select_window (numpy.ndarray(float64), optional): select window in the format like this [[l1, r1], [l2, r2], ...]. Defaults to None.

    Returns:
        A Array: the wave index in the select windows
    """
    if select_window is None:
        return wave < np.inf
    lw, rw = select_window[0]
    arg = (wave > lw) & (wave < rw)
    for win in select_window[1:]:
        lw, rw = win
        arg_tmp = (wave > lw) & (wave < rw)
        arg = arg | arg_tmp
    return arg


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
    # mod = PolynomialModel(degree=7)
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


def normalize_spec_gaussian_filter(wave, flux,
                                 fwhm=100,
                                 mask_windows=None,
                                 plot=False):
    """normalize the spectrum using the gaussian filter method

    Args:
        wave (numpy.ndarray): wavelength of the spectrum
        flux (numpy.ndarray): flux of the spectrum
        fwhm (float, optional): the width of the gaussian kernel. Defaults to 50.
        mask_windows (list)], optional): The mask windows before to do the normalization, which is in the format of 
        [
            [[mask_w1, mask_w2], [left_cont_w1, left_cont_w2], [right_cont_w1, right_cont_w2]],
            [[mask_w1, mask_w2], [left_cont_w1, left_cont_w2], [right_cont_w1, right_cont_w2]],
            ...
        ]. Defaults to None.
        plot (bool, optional): whether plot. Defaults to False.

    Returns:
        numpy.ndarray: the normalized flux
    """
    nflux = flux.copy()
    if mask_windows is not None:
        for wins in mask_windows:
            mask_win, cont_win1, cont_win2 = wins
            wcl = cont_win1[1]
            wcr = cont_win2[0]
            if wcl < np.min(wave) or wcr > np.max(wave):
                continue
            cont = get_linear_continuum(wave, nflux, [cont_win1], [cont_win2])
            arg = select_wave(wave, [mask_win])
            nflux[arg] = cont[arg]
    cont = spec_filter.gaussian_filter_wavespace(wave, nflux, fwhm)
    normflux = flux / cont
    if plot is True:
        plt.plot(wave, flux)
        plt.plot(wave, cont)
        if mask_windows is not None:
            for wins in mask_windows:
                mask_win, cont_win1, cont_win2 = wins
                wcl = cont_win1[1]
                wcr = cont_win2[0]
                if wcl < np.min(wave) or wcr > np.max(wave):
                    continue
                w1, w2 = mask_win
                plt.axvline(w1, linestyle=':', color='red')
                plt.axvline(w2, linestyle=':', color='red')
                w1, w2 = cont_win1
                plt.axvline(w1, linestyle=':', color='blue')
                plt.axvline(w2, linestyle=':', color='blue')
                w1, w2 = cont_win2
                plt.axvline(w1, linestyle=':', color='blue')
                plt.axvline(w2, linestyle=':', color='blue')
        plt.show()
    return normflux


def continuum(wave, flux, degree=7, maxiterations=10, plot=False, rejectemission=False, mask_window=None):
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
    if mask_window is not None:
        arg_mask = mask_wave(wave, mask_window)
    else:
        arg_mask = np.ones(wave.size, dtype=bool)
    if plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(wave, flux)

    def func(x, *par):
        return np.array(convol.legendre_poly(x, par))

    arg = np.where(np.isfinite(wave) & np.isfinite(flux))
    wave = wave[arg]
    flux = flux[arg]

    newave = normalize_wave(np.log10(wave))
    # newave = normalize_wave(wave)
    newave2 = newave.copy()

    binsize = int(flux.size / 50)
    newflux = median_filter(flux, binsize)
    if plot is True:
        ax.plot(wave, newflux)
    newflux = np.log10(newflux)
    # inipar = [1.0 for i in range(order+1)]
    # inipar = np.zeros(degree+1)
    # inipar[0] = 1.0
    inipar = 1/np.arange(1, degree+1, 1.0)
    # inipar = np.random.random(degree+1)
    # out = leastsq(residual, inipar, args=(newave, newflux))
    popt, _ = curve_fit(func, newave[arg_mask], newflux[arg_mask], p0=inipar)
    if maxiterations < 1:
        maxiterations = 1
    count = 0
    ideg = 1
    popt = np.ones(ideg)
    arg_protect = np.ones(newflux.size, dtype=bool)
    tmp_size = arg_protect.size
    arg_protect[arg_protect < 0.1 * tmp_size] = False
    arg_protect[arg_protect > 0.9 * tmp_size] = False
    while ideg < degree + 1:
    # while count < maxiterations:
        tmppopt, _ = curve_fit(func, newave[arg_mask], newflux[arg_mask], p0=popt)
        ideg = ideg + 1
        popt = np.zeros(ideg)
        popt[:-1] = tmppopt
        scale = func(newave, *popt)
        tmpuniform = newflux / scale
        std = np.std(tmpuniform)
        arg = np.where((tmpuniform < 1 - 2*std) & arg_protect)
        newflux[arg] = newflux[arg] + scale[arg] * std
        if rejectemission is True:
            arg = np.where((tmpuniform > 1 + 2*std) & arg_protect)
            newflux[arg] = newflux[arg] - scale[arg] * std
        keep_size = arg[0].size
        # if keep_size == len(newave):
        #     break
        # count = count + 1

    # tmpscale = convol.legendre_poly(newave2, out[0])
    tmpscale = func(newave2, *popt)
    tmpscale = 10**tmpscale
    # print('loop num = ', count)
    # print(popt)

    if plot is True:
        ax.plot(wave, 10**newflux)
        ax.plot(wave, tmpscale)
        if mask_window is not None:
            yl, yr = ax.get_ylim()
            for ind, win in enumerate(mask_window):
                if ind == 0:
                    ax.fill_between(win, [yl, yl], [yr, yr], color='grey', alpha=0.3, label='mask window')
                else:
                    ax.fill_between(win, [yl, yl], [yr, yr], color='grey', alpha=0.3)
            ax.set_ylim(yl, yr)
            ax.legend()
        
        plt.show()
        # print('flag')
    return flux / tmpscale
