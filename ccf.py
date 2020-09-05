import math
import numpy as np
import matplotlib.pyplot as plt
from . import libccf
from . import spec_func
from . import rebin


def shiftspec(flux, shift):

    sp = flux

    ln = len(sp)
    nsp = sp

    # Take the inverse Fourier transform and multiply by length to put it in IDL terms
    fourtr = np.fft.ifft(nsp) * len(nsp)   
    sig = np.arange(ln)/float(ln) - .5
    sig = np.roll(sig, int(ln/2))
    sh = sig*2. * np.pi * shift

    count=0
    shfourtr = np.zeros( (len(sh), 2) )
    complexarr2 = np.zeros( len(sh), 'complex' )
    for a,b in zip(np.cos(sh), np.sin(sh)):
        comps = complex(a,b)
        complexarr2[count] = comps
        count+=1

    shfourtr = complexarr2 * fourtr

    # Take the Fourier transform
    newsp = np.fft.fft(shfourtr) / len(shfourtr)
    newsp = newsp[0:ln]

    return newsp


def find_radial_velocity(wave, flux, wave_ref, flux_ref, mult=True, plot=False, ccfleft=-800, ccfright=800, velocity_resolution=1.0):
    """find the radial velocity using ccf method

    Args:
        wave (numpy.ndarray): spectral wave
        flux (numpy.ndarray): spectral flux
        wave_ref (numpy.ndarray): the spectral wave of template
        flux_ref (numpy.ndarray): the spectral flux of template
        mult (bool, optional): use multiplication to cal the ccf value, else use diff. Defaults to True.
        plot (bool, optional): whether plot the ccf profile. Defaults to False.
        ccfleft (int, optional): the left edge of ccf funtion, in the unit of km/s. Defaults to -800.
        ccfright (int, optional): the right edge of ccf function, in the unit of km/s. Defaults to 800.
        velocity_resolution (float, optional): the velocity resolution of ccf, in the unit of km/s. Defaults to 1.0.

    Returns:
        velocity(float, km/s): the velocity of the spectrum compared with the template. Here positive value means red shift,
        negative value means blue shift.
    """
    c = 299792.458 # km/s
    logwave = np.log(wave)
    logwave_ref = np.log(wave_ref)
    log_delta_w = np.min(np.diff(logwave))
    logwbegin = min(logwave[0], logwave_ref[0])
    logwend = max(logwave[-1], logwave_ref[-1])
    lognewave = np.arange(logwbegin, logwend, log_delta_w)
    newave = np.exp(lognewave)
    newflux = np.array(rebin.rebin(wave, flux, newave))
    newflux_ref = np.array(rebin.rebin(wave_ref, flux_ref, newave))
    cont = spec_func.continuum(newave, newflux)
    cont_ref = spec_func.continuum(newave, newflux_ref)
    norm_cont = (cont - np.mean(cont)) / np.std(cont)
    norm_cont_ref = (cont_ref - np.mean(cont_ref)) / np.std(cont_ref)
    # arrvelovity = np.arange(-800, 800, 1.0)
    # here the code is modified to first and second loops, in order to save the computation
    shiftleft = int(math.floor(ccfleft / c / log_delta_w))
    shiftright = int(math.ceil(ccfright / c / log_delta_w))
    shiftlst = np.arange(shiftleft, shiftright+1)
    select_range = max(abs(shiftlst[0]), abs(shiftlst[-1]))
    ccf_valuelst = []
    for shift in shiftlst:
        norm_cont_shift = shiftspec(norm_cont, shift)
        if mult is True:
            mul_val = norm_cont_shift[select_range:-select_range] * norm_cont_ref[select_range:-select_range]
            ccf_val = np.abs(np.sum(mul_val))
            # ccf_valuelst.append(ccf_val)
        else:
            dif_val = norm_cont_shift[select_range:-select_range] - norm_cont_ref[select_range:-select_range]
            ccf_val = np.sum(np.real(dif_val * dif_val))
        ccf_valuelst.append(ccf_val)
    if plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(shiftlst, ccf_valuelst)
        # plt.show()
    if mult is True:
        index = np.argmax(ccf_valuelst)
    else:
        index = np.argmin(ccf_valuelst)
    measure_shift = shiftlst[index]
    delta_shift = velocity_resolution / c / log_delta_w
    if delta_shift > 1:
        if plot is True:
            plt.show()
        return -measure_shift * log_delta_w * c

    shiftlst = np.arange(measure_shift-1, measure_shift+1, delta_shift)
    ccf_valuelst = []
    for shift in shiftlst:
        norm_cont_shift = shiftspec(norm_cont, shift)
        if mult is True:
            mul_val = norm_cont_shift[select_range:-select_range] * norm_cont_ref[select_range:-select_range]
            ccf_val = np.abs(np.sum(mul_val))
            # ccf_valuelst.append(ccf_val)
        else:
            dif_val = norm_cont_shift[select_range:-select_range] - norm_cont_ref[select_range:-select_range]
            ccf_val = np.sum(np.real(dif_val * dif_val))
        ccf_valuelst.append(ccf_val)
    if mult is True:
        index = np.argmax(ccf_valuelst)
    else:
        index = np.argmin(ccf_valuelst)
    measure_shift = shiftlst[index]
    velocity = measure_shift * log_delta_w * c
    if plot is True:
        ax.plot(shiftlst, ccf_valuelst)
        plt.show()
    return -velocity


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