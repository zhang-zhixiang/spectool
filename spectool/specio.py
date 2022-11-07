import os
import re
import numpy as np
from astropy.io import fits
from PyAstronomy.pyasl import read1dFitsSpec
from PyAstronomy.pyasl import hmsToDeg, dmsToDeg
from . import spec_func


def read_iraf_spec(fn, aper=1):
    """read the spectrum from a fits file generated by iraf

    Args:
        fn (str): fits name
        aper (int, optional): the aperture of the spectrum in fn (aper start from 1). Defaults to 1.

    Returns:
        wave, flux, err: numpy.ndarray
    """

    def get_band(fitname, aper):
        apername = 'APNUM' + str(int(aper))
        aperband = fits.getval(fitname, apername)
        band = int(aperband.strip().split()[0]) - 1
        return band
    hdul = fits.open(fn)
    data = hdul[0].data
    head = hdul[0].header
    size = head['NAXIS1']
    step = head['CD1_1']
    start = head['CRVAL1']
    crpix1 = head['CRPIX1']
    wave = (np.arange(size) + 1.0 - crpix1) * step + start
    if len(data.shape) == 2:
        flux = data[0, :]
        err = data[1, :]
    else:
        band = get_band(fn, aper)
        flux = data[0, band, :].astype('float64')
        err = data[3, band, :].astype('float64')
    return wave, flux, err


def read_iraf_echelle(fn):
    """read the spectra file created by iraf package echell,

    Args:
        fn (fitname): fits name 

    Returns:
        [[wave, flux], [wave, flux],...]: A spectra list
        if the fits file contains sigma information, then return
        [[wave, flux, sigma], [wave, flux, sigma],...]
    """
    # from pyraf import iraf
    # import tempfile
    fit = fits.open(fn)
    head = fit[0].header
    data = fit[0].data
    ind = 1
    wtext = ''
    while True:
        keyname = 'WAT2_{:03}'.format(ind)
        if keyname in head:
            keytext = head[keyname]
            if len(keytext) < 68:
                size_space = 68 - len(keytext)
                keytext = keytext + ' ' * size_space
            wtext = wtext + keytext
        else:
            break
        ind = ind + 1
    lis = re.findall(r'"(.+?)"', wtext)
    # specnames = re.findall(r'spec.{1,3}=', wtext)
    specs = []
    for ind, line in enumerate(lis):
        if len(data.shape) == 2:
            flux = data[ind].astype(float)
            sigma = None
        else:
            shape = data.shape
            flux = data[0, ind].astype(float)
            if shape[0] < 4:
                sigma = None
            else:
                sigma = data[3, ind].astype(float)
        lines = line.split()
        wbegin = float(lines[3])
        step = float(lines[4])
        size = int(lines[5])
        wave = np.arange(size) * step + wbegin
        if sigma is None:
            specs.append((wave, flux))
        else:
            specs.append((wave, flux, sigma))
    return specs


def read_lte_spec(fn):
    hdul = fits.open(fn)
    data = hdul[1].data
    wave = data['WAVELENGTH'].astype('float64')
    flux = data['FLUX'].astype('float64')
    bflux = data['BBFLUX'].astype('float64')
    return wave, flux, bflux


def read_lamost_low(fn):
    """read the LAMOST low resolution spectrum

    Args:
        fn (string): spectral file name

    Returns:
        wave, flux, error (numpy.ndarray): data of the spectrum
    """
    hdul = fits.open(fn)
    data = hdul[1].data
    wave = data['WAVELENGTH'][0].astype('float64')
    flux = data['FLUX'][0].astype('float64')
    ivar = data['IVAR'][0].astype('float64')
    arg = np.where(ivar == 0)
    ivar[arg] = 1
    err = 1 / ivar**0.5
    err[arg] = np.inf
    argsort = np.argsort(wave)
    wave = wave[argsort]
    flux = flux[argsort]
    err = err[argsort]
    return wave, flux, err


def read_lamost_med(fn, hduid):
    data = fits.getdata(fn, ext=hduid)
    flux = data['flux'].astype('float64')
    if len(flux.shape) == 2:
        wave = data['WAVELENGTH'].astype('float64')
        invar = data['IVAR'].astype('float64')
        wave = wave[0, :]
        flux = flux[0, :]
        invar = invar[0, :]
    else:
        wave = 10**data['loglam'].astype('float64')
        invar = data['IVAR'].astype('float64')
    arg = invar == 0.0
    invar[arg] = 1.0
    invar2 = invar**0.5
    err = 1 / invar2
    err[arg] = np.inf
    arg = np.argsort(wave)
    wave = wave[arg]
    flux = flux[arg]
    err = err[arg]
    return wave, flux, err


def read_txt_spec(fn):
    data = np.loadtxt(fn)
    wave = data[:, 0]
    flux = data[:, 1]
    if data.shape[1] >= 3:
        err = data[:, 2]
    else:
        err = None
    return wave, flux, err


def read_sdss_spec(fn):
    hdul = fits.open(fn)
    data = hdul[1].data
    wave = data['loglam'].astype('float64')
    flux = data['flux'].astype('float64')
    ivar = data['ivar'].astype('float64')
    return wave, flux, ivar


def read_spec(fn):
    data = 123


def get_ra_dec(fn):
    strra = fits.getval(fn, 'RA')
    strdec = fits.getval(fn, 'DEC')
    if isinstance(strra, float):
        return strra, strdec
    if ':' in strra:
        hh, mm, ss = [float(i) for i in strra.split(':')]
        ra = hmsToDeg(hh, mm, ss)
        dd, mm, ss = [float(i) for i in strdec.split(':')]
        dec = dmsToDeg(dd, mm, ss)
        return ra, dec
    else:
        return float(strra), float(strdec)


class Spectrum(object):
    def __init__(self):
        self.wave = None
        self.flux = None
        self.error = None
