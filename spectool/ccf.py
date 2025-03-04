import math
import time
import numpy as np
import matplotlib.pyplot as plt
from . import libccf
from . import spec_func
from . import rebin
from . import liblogccf
from . import pyrebin


def shiftspec(flux, shift):

    ln = len(flux)
    nsp = flux

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
    newflux = np.array(rebin.rebin_padvalue(wave, flux, newave))
    newflux_ref = np.array(rebin.rebin_padvalue(wave_ref, flux_ref, newave))
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
    count_pixels = norm_cont[select_range:-select_range].size
    for shift in shiftlst:
        norm_cont_shift = shiftspec(norm_cont, -shift)
        if mult is True:
            mul_val = norm_cont_shift[select_range:-select_range] * norm_cont_ref[select_range:-select_range]
            ccf_val = np.abs(np.sum(mul_val)) / count_pixels
            # ccf_valuelst.append(ccf_val)
        else:
            dif_val = norm_cont_shift[select_range:-select_range] - norm_cont_ref[select_range:-select_range]
            ccf_val = np.sum(np.real(dif_val * dif_val)) / count_pixels
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
        return measure_shift * log_delta_w * c

    shiftlst = np.arange(measure_shift-1, measure_shift+1, delta_shift)
    ccf_valuelst = []
    for shift in shiftlst:
        norm_cont_shift = shiftspec(norm_cont, -shift)
        if mult is True:
            mul_val = norm_cont_shift[select_range:-select_range] * norm_cont_ref[select_range:-select_range]
            ccf_val = np.abs(np.sum(mul_val)) / count_pixels
            # ccf_valuelst.append(ccf_val)
        else:
            dif_val = norm_cont_shift[select_range:-select_range] - norm_cont_ref[select_range:-select_range]
            ccf_val = np.sum(np.real(dif_val * dif_val)) / count_pixels
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
    return velocity


def find_radial_velocity2(wave, flux, wave_ref, flux_ref, 
                          mult=True, plot=False, 
                          ccfleft=-800, ccfright=800, velocity_resolution=1.0, 
                          maskwindow=None, returnrmax=False, fig=None, 
                          do_continuum=True, degree=7):
    """
    Find the radial velocity by cross-correlating the input spectrum with a reference spectrum.

    Parameters:
    -----------
    wave : array_like
        Wavelength values of the input spectrum.
        
    flux : array_like
        Flux values corresponding to the wavelengths in `wave`.
        
    wave_ref : array_like
        Wavelength values of the reference spectrum (template).
        
    flux_ref : array_like
        Flux values corresponding to the wavelengths in `wave_ref`.
        
    mult : bool, optional, default=True
        If True, the resulting radial velocity is multiplied by -1 to get the convention for radial velocity.
        
    plot : bool, optional, default=False
        If True, a plot of the spectra and the cross-correlation function (CCF) is generated.
        
    ccfleft : float, optional, default=-800
        Left velocity range in km/s for the cross-correlation function.
        
    ccfright : float, optional, default=800
        Right velocity range in km/s for the cross-correlation function.
        
    velocity_resolution : float, optional, default=1.0
        Velocity resolution in km/s for the cross-correlation.
        
    maskwindow : list of tuples, optional, default=None
        A list of wavelength ranges to be masked in both spectra. Each range is a tuple of (start, end).
        
    returnrmax : bool, optional, default=False
        If True, returns both the radial velocity and the maximum value of the cross-correlation function (rmax).
        
    fig : matplotlib.figure.Figure, optional, default=None
        If provided, the plot will be drawn on this figure. Otherwise, a new figure will be created.
        
    do_continuum : bool, optional, default=True
        If True, the spectra are normalized by their continuum. If False, no continuum normalization is applied.
        
    degree : int, optional, default=7
        The degree of the polynomial used for continuum fitting if `do_continuum` is True.

    Returns:
    --------
    velocity : float
        The computed radial velocity in km/s.
        
    rmax : float, optional
        The maximum value of the cross-correlation function. Returned only if `returnrmax` is True.
    """
    wl = max(wave[0], wave_ref[0])
    wr = min(wave[-1], wave_ref[-1])
    arg = np.where((wave<wr) & (wave>wl))
    wave = wave[arg]
    flux = flux[arg]
    arg = np.where((wave_ref<wr) & (wave_ref>wl))
    wave_ref = wave_ref[arg]
    flux_ref = flux_ref[arg]
    c = 299792.458 # km/s
    logwave = np.log(wave)
    logwave_ref = np.log(wave_ref)
    log_delta_w = np.min(np.diff(logwave))
    if log_delta_w < 0.1 / c:
        log_delta_w = 0.1 / c
    logwbegin = min(logwave[0], logwave_ref[0])
    logwend = max(logwave[-1], logwave_ref[-1])
    lognewave = np.arange(logwbegin, logwend, log_delta_w)
    newave = np.exp(lognewave)
    newflux = np.array(pyrebin.rebin_padvalue(wave, flux, newave, interp_kind='cubic'))
    newflux_ref = np.array(pyrebin.rebin_padvalue(wave_ref, flux_ref, newave, interp_kind='cubic'))
    if do_continuum is True:
        cont = spec_func.continuum(newave, newflux, degree=degree)
        cont_ref = spec_func.continuum(newave, newflux_ref, degree=degree)
    else:
        cont = newflux
        cont_ref = newflux_ref
    if maskwindow is not None:
        cont_old = cont.copy()
        cont_ref_old = cont_ref.copy()
        arg = spec_func.select_wave(newave, maskwindow)
        # print(arg)
        cont[arg] = 1.0
        cont_ref[arg] = 1.0
    norm_cont = (cont - np.mean(cont)) / np.std(cont)
    norm_cont_ref = (cont_ref - np.mean(cont_ref)) / np.std(cont_ref)
    shiftleft = int(math.floor(ccfleft / c / log_delta_w))
    shiftright = int(math.ceil(ccfright / c / log_delta_w))
    delta_shift = velocity_resolution / c / log_delta_w
    shift, rmax = liblogccf.get_shift(norm_cont, norm_cont_ref, shiftleft, shiftright, delta_shift, True)
    velocity = shift * log_delta_w * c
    if plot is True:
        shiftlst, rlst = liblogccf.get_ccf(norm_cont, norm_cont_ref, shiftleft, shiftright, delta_shift, True)
        shiftlst = np.array(shiftlst)
        rlst = np.array(rlst)
        arg = np.argsort(shiftlst)
        shiftlst = shiftlst[arg]
        rlst = rlst[arg]
        velocitylst = shiftlst * log_delta_w * c
        if fig is None:
            fig = plt.figure(figsize=(13, 4))
        else:
            fig.clf()
        ax1 = fig.add_axes([0.05, 0.05+0.08, 0.6, 0.85])
        # ax2 = fig.add_axes([0.05, 0.53+0.02, 0.6, 0.4])
        ax3 = fig.add_axes([0.68, 0.05+0.08, 0.28, 0.85])
        # ax = fig.add_subplot(111)
        ax1.set_xlabel('Wavelength ($\mathrm{\AA}$)')
        ax1.set_ylabel('Normalized Flux')
        if maskwindow is None:
            ax1.plot(newave, cont, label='spec')
            ax1.plot(newave, cont_ref, label='template')
            ax1.legend()
        else:
            ax1.plot(newave, cont_old, label='spec')
            ax1.plot(newave, cont_ref_old, label='template')
            ax1.legend()
            yl, yr = ax1.get_ylim()
            xl, xr = newave[0], newave[-1]
            for win in maskwindow:
                if (xl < win[0] and xr > win[0]) or (xl < win[1] and xr > win[1]):
                    ax1.fill_between(win, yl, yr, color='C7', alpha=0.3)
            ax1.set_ylim(yl, yr)
        # ax2.legend()
        ax3.plot(velocitylst, rlst)
        ax3.set_xlabel('Velocity (km/s)')
        ax3.yaxis.set_label_position('right')
        ax3.set_ylabel('CCF')
        ax3.axvline(velocity, color='red', linestyle='--', label='vel = %.2f km/s' % velocity)
        ax3.axhline(rmax, color='C3', linestyle=':', label='rmax = %.2f' % rmax)
        ax3.legend()
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # fig.show()
    if returnrmax is True:
        return velocity, rmax
    return velocity


def find_radial_velocity2_mc(wave, flux, fluxerr, wave_ref, flux_ref, mcnum=1000,
                             mult=True, ccfleft=-800, ccfright=800, velocity_resolution=1,
                             maskwindow=None,
                             do_continuum=True, degree=7):
    """
        Estimate the radial velocity by cross-correlating the given spectrum with a reference spectrum using Monte Carlo simulations.

        Parameters:
        -----------
        wave : array_like
            The wavelength values of the observed spectrum.
        flux : array_like
            The flux values corresponding to the wavelengths in `wave`.
        fluxerr : array_like
            The flux error values corresponding to the flux data.
        wave_ref : array_like
            The wavelength values of the reference spectrum.
        flux_ref : array_like
            The flux values corresponding to the wavelengths in `wave_ref`.
        mcnum : int, optional
            The number of Monte Carlo simulations to run (default is 1000).
        mult : bool, optional
            Whether to multiply the continuum by the scale (default is True).
        ccfleft : float, optional
            The left limit (in km/s) for the cross-correlation function (default is -800 km/s).
        ccfright : float, optional
            The right limit (in km/s) for the cross-correlation function (default is 800 km/s).
        velocity_resolution : float, optional
            The velocity resolution (in km/s) for the output (default is 1 km/s).
        maskwindow : list of tuples, optional, default=None
            A list of wavelength ranges to be masked in both spectra. Each range is a tuple of (start, end).
        do_continuum : bool, optional
            Whether to remove the continuum from the spectra before cross-correlation (default is True).
        degree : int, optional
            The degree of the polynomial used for continuum fitting (default is 7).

        Returns:
        --------
        velocity_lst : ndarray
            An array of radial velocity estimates obtained from the Monte Carlo simulations (in km/s).
        
        Notes:
        ------
        This function works by rebinning both the observed and reference spectra to a common wavelength grid, 
        performing Monte Carlo simulations to introduce random noise into the flux values, and cross-correlating 
        the noisy spectra with the reference to compute the radial velocity.
        """
    wl = max(wave[0], wave_ref[0])
    wr = min(wave[-1], wave_ref[-1])
    arg = np.where((wave<=wr) & (wave>=wl))
    wave = wave[arg]
    flux = flux[arg]
    fluxerr = fluxerr[arg]
    arg = np.where((wave_ref<=wr) & (wave_ref>=wl))
    wave_ref = wave_ref[arg]
    flux_ref = flux_ref[arg]
    c = 299792.458 # km/s
    logwave = np.log(wave)
    logwave_ref = np.log(wave_ref)
    log_delta_w = np.min(np.diff(logwave))
    logwbegin = min(logwave[0], logwave_ref[0])
    logwend = max(logwave[-1], logwave_ref[-1])
    lognewave = np.arange(logwbegin, logwend, log_delta_w)
    newave = np.exp(lognewave)
    newflux = np.array(rebin.rebin_padvalue(wave, flux, newave))
    newflux_ref = np.array(rebin.rebin_padvalue(wave_ref, flux_ref, newave))
    if do_continuum is True:
        cont = spec_func.continuum(newave, newflux, degree=degree, mask_window=maskwindow)
        cont_ref = spec_func.continuum(newave, newflux_ref, degree=degree, mask_window=maskwindow)
    else:
        cont = newflux
        cont_ref = newflux_ref
    if maskwindow is not None:
        arg = spec_func.select_wave(newave, maskwindow)
        cont[arg] = 1.0
        cont_ref[arg] = 1.0
    scale = cont / newflux
    scale_ref = cont_ref / newflux_ref
    velocity_lst = []
    for ind in range(mcnum):
        fluxc = flux.copy()
        err = np.random.normal(size=fluxc.size) * fluxerr
        fluxc = fluxc + err
        arg = np.unique(np.random.choice(np.arange(fluxc.size), size=fluxc.size, replace=True))
        swave = wave[arg]
        sflux = fluxc[arg]
        argref = np.unique(np.random.choice(np.arange(flux_ref.size), size=flux_ref.size, replace=True))
        swave_ref = wave_ref[argref]
        sflux_ref = flux_ref[argref]
        newflux = np.array(rebin.rebin_padvalue(swave, sflux, newave))
        newflux_ref = np.array(rebin.rebin_padvalue(swave_ref, sflux_ref, newave))
        cont = newflux * scale
        cont_ref = newflux_ref * scale_ref

        norm_cont = (cont - np.mean(cont)) / np.std(cont)
        norm_cont_ref = (cont_ref - np.mean(cont_ref)) / np.std(cont_ref)
        shiftleft = int(math.floor(ccfleft / c / log_delta_w))
        shiftright = int(math.ceil(ccfright / c / log_delta_w))
        delta_shift = velocity_resolution / c / log_delta_w
        shift, rmax = liblogccf.get_shift(norm_cont, norm_cont_ref, shiftleft, shiftright, delta_shift, True)
        velocity = shift * log_delta_w * c
        velocity_lst.append(velocity)
    return np.array(velocity_lst)


def find_radial_velocity_mc(wave, flux, wave_ref, flux_ref, mult=True, plot=False, ccfleft=-800, ccfright=800, velocity_resolution=1.0, mcnumber=50, incratio=0.5):
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
    newflux = np.array(rebin.rebin_padvalue(wave, flux, newave))
    newflux_ref = np.array(rebin.rebin_padvalue(wave_ref, flux_ref, newave))
    # t1 = time.time()
    # binsize = int(newflux.size / 50) + 1
    cont = spec_func.continuum(newave, newflux, maxiterations=1)
    cont_ref = spec_func.continuum(newave, newflux_ref, maxiterations=1)
    # t2 = time.time()
    # print('reduce continuum time spend =', t2 - t1)
    norm_cont = (cont - np.mean(cont)) / np.std(cont)
    norm_cont_ref = (cont_ref - np.mean(cont_ref)) / np.std(cont_ref)
    shiftleft = int(math.floor(ccfleft / c / log_delta_w))
    shiftright = int(math.ceil(ccfright / c / log_delta_w))
    delta_shift = velocity_resolution / c / log_delta_w
    argc = np.isfinite(norm_cont) == False
    argt = np.isfinite(norm_cont_ref) == False
    norm_cont[argc] = 0.0
    norm_cont_ref[argt] = 0.0
    if len(norm_cont[argc]) > 0 or len(norm_cont_ref[argt] > 0):
        print('Caution !!, NaN or inf ocured')
    # t1 = time.time()
    bestshiftlst, rmaxlst = liblogccf.get_shift_mc(norm_cont, norm_cont_ref, shiftleft, shiftright, delta_shift, mcnumber, incratio, True)
    # t2 = time.time()
    # print('ccf mc time spend =', t2-t1)
    bestshiftlst = np.array(bestshiftlst)
    rmaxlst = np.array(rmaxlst)
    velocitylst = bestshiftlst * log_delta_w * c
    if plot is True:
        shiftlst, rlst = liblogccf.get_ccf(norm_cont, norm_cont_ref, shiftleft, shiftright, delta_shift, True)
        shiftlst = np.array(shiftlst)
        rlst = np.array(rlst)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(newave, norm_cont)
        ax1.plot(newave, norm_cont_ref)
        arg = np.argsort(shiftlst)
        shiftlst = shiftlst[arg]
        rlst = rlst[arg]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(shiftlst, rlst)
        fig1.show()
        fig.show()
        # plt.show()
    return velocitylst, rmaxlst


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
